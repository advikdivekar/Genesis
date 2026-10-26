"""GENESIS live viewer server.

Runs one seeded GenesisEnv in a background thread and streams its state to browser clients over a
WebSocket (aiohttp). The client (viewer/static, three.js from viewer/node_modules) only renders; it never
changes the simulation except through logged commands.

    .venv/bin/python -m viewer.server --seed 0 --policy heuristic      # then open http://localhost:8787

Logging: every session writes a run directory under runs/viewer/ (resolved config, manifest, metrics.csv,
chunked event logs, summary.json) plus controls.jsonl, which records every command (pause, speed,
disturbances, restarts, selections) with the tick it was applied at. Commands are applied between ticks,
so a session is reproducible from (seed, config, controls.jsonl); playback speed never changes results.
Disturbances go through env.trigger_*, so they also appear in env.manual_log and the event stream.

Agents are driven by the non-learning baselines (agents/baselines.py) until Stage 2 learners exist;
the client labels this.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web
from scipy import ndimage
from scipy.spatial import cKDTree

from agents.baselines import make_baseline
from sim import GenesisEnv, load_config
from sim import constants as C
from sim.constants import DEATH_NAMES
from sim.layout import (A_CARRY_AMT, A_ENERGY, A_HEALTH, A_HYDRATION, A_TEMP_EFF, A_WARMTH, F_ALIVE, F_FLEE,
                        F_HEALTH, F_SPECIES, F_X, F_Y, I_AGE, I_ALIVE, I_CARRY_TYPE, I_GEN, I_HIST_PTR,
                        I_LAST_ACTION, I_LIFESPAN, I_PARENT, I_SHELTERED, I_UID, I_X, I_Y, STAT_NAMES)
from sim.logger import RunLogger, append_agent_log
from sim.recorder import TrajectoryRecorder

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
THREE = HERE / "node_modules" / "three"

BASE_TPS = 10.0                      # "1x" = 10 ticks/s (a sim day of 120 ticks takes 12 s)
POLICIES = ("heuristic", "greedy", "random")
FRAME_HZ = 30.0
SAMPLE_EVERY = 20                    # ticks between dashboard samples (6 per sim day)
HISTORY = 720                        # dashboard samples kept server-side (~120 sim days)
FEED_TYPES = (C.EV_DEATH, C.EV_BIRTH, C.EV_KILL_LARGE, C.EV_FIRE_START, C.EV_FLOOD_START,
              C.EV_DROUGHT_START, C.EV_FAMINE_START, C.EV_COLD_SNAP_START)
FX_TYPES = (C.EV_KILL_SMALL, C.EV_ATTACK_LARGE, C.EV_PREDATOR_ATTACK, C.EV_SIGNAL)
_DT = {np.dtype(np.uint8): "u8", np.dtype(np.uint16): "u16", np.dtype(np.uint32): "u32",
       np.dtype(np.int32): "i32", np.dtype(np.float32): "f32"}


# ---------------------------------------------------------------------------------------------
# wire format: [u32 header_len][header JSON, space padded to 4][buffers, each padded to 4 bytes]
def pack(header: dict, buffers: dict | None = None) -> bytes:
    metas, chunks, off = [], [], 0
    for name, arr in (buffers or {}).items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        metas.append({"name": name, "dtype": _DT[arr.dtype], "offset": off, "length": int(arr.size)})
        pad = (-len(raw)) % 4
        chunks.append(raw + b"\0" * pad)
        off += len(raw) + pad
    header = dict(header, bufs=metas)
    hj = json.dumps(header, separators=(",", ":"), default=_json_default).encode()
    hj += b" " * ((-len(hj)) % 4)
    return len(hj).to_bytes(4, "little") + hj + b"".join(chunks)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------------------------
def scene_heights(env: GenesisEnv):
    """Render heights (world units, 1 cell = 1 unit) for the terrain mesh, plus water-surface levels:
    `water` for permanent water (NaN elsewhere, extended one cell onto the shore so the surface meets
    the bank) and `flood` for floodplain cells (surface shown while a flood covers the cell).
    Purely visual: the simulation itself is a flat grid."""
    W = env.world
    t, e = env.terrain, env.elev.astype(np.float64)
    sea = env.cfg.world.sea_level
    ocean, lake = W["ocean"], W["lake"]
    river = (W["river"] | W["stream"]) & ~ocean & ~lake
    hl = np.clip((e - sea) / (1.0 - sea), 0.0, 1.0)
    h0 = 0.6 + 24.0 * hl ** 1.8
    h0 = np.where(t == C.MOUNTAIN, h0 + 5.0, h0)
    h0 = np.where(ocean, -1.2 - 9.0 * np.clip((sea - e) / sea, 0.0, 1.0), h0)
    hs = ndimage.gaussian_filter(h0, 1.2)
    wl = np.full(t.shape, np.nan)
    wl[ocean] = 0.0
    wl[river] = hs[river] - 0.10
    lab, n = ndimage.label(lake)
    for k in range(1, n + 1):
        m = lab == k
        ring = ndimage.binary_dilation(m, iterations=2) & ~m & ~ocean
        level = np.percentile(hs[ring], 15) - 0.1 if ring.any() else hs[m].mean()
        wl[m] = level
    h = hs.copy()
    h[river] = wl[river] - 0.45
    deep = t == C.DEEP_WATER
    h[lake] = wl[lake] - 0.7 - 1.4 * deep[lake]
    h[ocean] = np.minimum(hs[ocean], -0.6)
    h = ndimage.gaussian_filter(h, 0.6)
    water = ~np.isnan(wl)
    h[water] = np.minimum(h[water], wl[water] - 0.25)
    wl_max = ndimage.maximum_filter(np.where(water, wl, -1e9), size=3)
    land = ~water
    h[land] = np.maximum(h[land], wl_max[land] + 0.08)
    h[land & ocean] = np.maximum(h[land & ocean], 0.1)
    # jagged peaks
    rng = np.random.default_rng([env.seed, 0x5CE7E])
    h += (t == C.MOUNTAIN) * rng.random(t.shape) * 1.5
    # water surface map extended one cell onto banks
    wsurf = np.where(water, wl, np.nan)
    shore = land & (wl_max > -1e8)
    wsurf[shore] = wl_max[shore]
    flood = np.where(env.floodplain, hs + 0.2, np.nan)
    return h.astype(np.float32), wsurf.astype(np.float32), flood.astype(np.float32)



# ---------------------------------------------------------------------------------------------
class ViewerRunLogger(RunLogger):
    """RunLogger whose event log is flushed to disk in chunks (viewer sessions can run for hours)."""
    CHUNK = 400_000

    def __init__(self, run_dir, env, meta=None):
        super().__init__(run_dir, env, meta)
        self._rows = 0
        self._chunks = 0
        self.controls = open(self.dir / "controls.jsonl", "a")

    def collect_events(self):
        if self.env.event_log:
            self._rows += sum(len(e) for e in self.env.event_log)
        super().collect_events()
        if self._rows >= self.CHUNK:
            self._flush()

    def _flush(self):
        if self._events:
            np.save(self.dir / f"events_{self._chunks:04d}.npy", np.concatenate(self._events))
            self._chunks += 1
            self._events.clear()
        self._rows = 0

    def log_control(self, rec: dict):
        self.controls.write(json.dumps(rec, default=_json_default) + "\n")
        self.controls.flush()

    def close(self, summary=None):
        self.collect_events()
        self._flush()
        if self._csv:
            self._csv.close()
        self.controls.close()
        s = dict(final=self.env.metrics(), manual_interventions=self.env.manual_log, event_chunks=self._chunks,
                 event_columns=["tick", "type", "a_uid", "b_uid", "x", "y", "value"],
                 finished=dt.datetime.now().isoformat(timespec="seconds"), **(summary or {}))
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2, default=float))
        return s


# ---------------------------------------------------------------------------------------------
class Sim:
    """Owns the environment; all mutation happens on its thread. Clients read packed frames."""

    def __init__(self, seed: int, policy: str, n_agents: int | None, config: str | None,
                 overrides: dict | None = None, log_root: Path | None = None, speed: float = 1.0,
                 agent_log: bool = True, record: bool = True, record_obs: bool | None = None,
                 checkpoint: str | None = None):
        self.config_path = config
        self.agent_log = agent_log
        self.checkpoint = checkpoint
        self.policies = POLICIES + (("ppo",) if checkpoint else ())
        self.record = record
        self.record_obs = record_obs
        self.base_overrides = overrides or {}
        self.log_root = Path(log_root or ROOT / "runs" / "viewer")
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.speed = speed               # multiple of BASE_TPS; 0 = as fast as possible
        self.paused = False
        self.selected: set[int] = set()
        self.version = 0
        self.stop_flag = threading.Event()
        self.ready = threading.Event()
        self.frame: bytes | None = None
        self.frame_id = 0
        self.fields: bytes | None = None
        self.fields_id = 0
        self.details: dict[int, dict] = {}
        self.status = "starting"
        self._build(seed, policy, n_agents)

    # ---------------------------------------------------------------- setup / teardown
    def _build(self, seed, policy, n_agents):
        if policy not in self.policies:
            raise ValueError(f"policy must be one of {self.policies}")
        ov = json.loads(json.dumps(self.base_overrides))
        if n_agents:
            ov.setdefault("agents", {})["n_initial"] = int(n_agents)
        self.status = f"generating world (seed {seed})"
        t0 = time.perf_counter()
        cfg = load_config(self.config_path, overrides=ov or None)
        env = GenesisEnv(cfg, seed=seed)
        self.env = env
        self.policy_name = policy
        if policy == "ppo":
            from agents.learned import LearnedPolicy   # torch only when a learner is used
            self.policy = LearnedPolicy(self.checkpoint, env, seed=seed)
            self.policy_desc = f"PPO learner · {Path(self.checkpoint).parent.parent.name} iter {self.policy.iter}"
        else:
            self.policy = make_baseline(policy, env, seed=seed)
            self.policy_desc = f"{policy} baseline · non-learning"

        self.obs = env.observe()
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = ViewerRunLogger(self.log_root / f"{stamp}_s{seed}_{policy}", env,
                                      meta={"source": "viewer", "policy": policy, "base_tps": BASE_TPS})
        self.recorder = (TrajectoryRecorder(env, self.logger.dir, obs=self.record_obs,
                                            meta={"source": "viewer", "policy": policy})
                         if self.record and cfg.logging.get("trajectory", True) else None)
        self.build_s = time.perf_counter() - t0
        self.agent_ticks = 0
        self.scrounge = 0
        self.history: collections.deque = collections.deque(maxlen=HISTORY)
        self.feed: collections.deque = collections.deque(maxlen=400)
        self.fx: list = []
        self.new_samples: list = []
        self.tps_window: collections.deque = collections.deque(maxlen=64)
        self.land_cells = int(np.isin(env.terrain, [C.GRASS, C.MEADOW, C.FOREST, C.SAND, C.BARREN, C.ROCK,
                                                   C.CAVE, C.SHALLOW]).sum())
        self._init_msg = self._make_init()
        self._sample()
        self.version += 1
        self.status = "running"

    def _close_run(self):
        env = self.env
        run_ticks = env.tick - env.spinup_ticks        # env.tick includes the ecological spin-up
        years = run_ticks / env.year_len
        deaths = int(env.deaths_by_cause.sum())
        dpy = deaths / max(1e-9, self.agent_ticks / env.year_len)
        rec = self.recorder.close(dict(policy=self.policy_name)) if self.recorder else None
        s = self.logger.close(dict(trajectory=dict(ticks=rec["ticks"], agent_rows=rec["agent_rows"], bytes=rec["bytes"])
                                   if rec else None, ticks=run_ticks, spinup_ticks=env.spinup_ticks, agent_ticks=self.agent_ticks,
                                   deaths_per_agent_year=dpy, policy=self.policy_name))
        if self.agent_log and run_ticks >= env.tpd:   # only sessions of at least one sim day
            causes = {n: int(v) for n, v in zip(DEATH_NAMES, env.deaths_by_cause) if v}
            append_agent_log(
                f"Live viewer session ({self.policy_name}, seed {env.seed})",
                [f"Viewer run `{_rel(self.logger.dir)}`: {run_ticks} ticks ({years:.2f} sim years), "
                 f"{env.cfg.agents.n_initial} agents, policy **{self.policy_name}** (non-learning baseline).",
                 f"- deaths/agent-year {dpy:.2f}; deaths by cause {causes}; "
                 f"population at end {env.population()}.",
                 f"- manual interventions: {len(env.manual_log)} "
                 f"({', '.join(sorted({m['kind'] for m in env.manual_log})) or 'none'}). Sessions with "
                 "interventions are not comparable to `run_baselines.py` numbers.",
                 f"- eating food another agent killed/dropped: {self.scrounge} meals "
                 "(a co-feeding count, not evidence of intentional sharing)."])
        return s

    # ---------------------------------------------------------------- messages
    def _make_init(self):
        env = self.env
        h, wsurf, flood = scene_heights(env)
        cfg = env.cfg
        hdr = dict(
            type="init", W=env.W, H=env.H, seed=env.seed, policy=self.policy_name, policies=list(self.policies),
            policy_desc=self.policy_desc,
            action_names=env.action_names, tpd=env.tpd, season_len=env.season_len, year_len=env.year_len,
            n_initial=cfg.agents.n_initial, max_agents=env.N, base_tps=BASE_TPS,
            fauna_initial=[cfg.fauna.small_prey.initial, cfg.fauna.large_prey.initial, cfg.fauna.predators.initial],
            fauna_max=[int(v) for v in env.fauna_max], fruit_max=cfg.resources.fruit_max,
            capabilities={k: v for k, v in cfg.capabilities.items() if isinstance(v, bool)},
            terrain_names=C.TERRAIN_NAMES, death_names=DEATH_NAMES, run_dir=str(_rel(self.logger.dir)),
            signal_idx=env.signal_idx, spinup_ticks=env.spinup_ticks, build_s=round(self.build_s, 2), sample_every=SAMPLE_EVERY,
        )
        bufs = {
            "terrain": env.terrain.astype(np.uint8), "height": h, "water": wsurf, "flood": flood,
            "fertility": env.fertility.astype(np.float32),
            "tree_x": env.tree_x.astype(np.uint16), "tree_y": env.tree_y.astype(np.uint16),
            "tree_season": env.tree_season.astype(np.uint8),
        }
        return hdr, bufs

    def init_message(self) -> bytes:
        """Static world + server-side history, so a late client gets populated charts and feed."""
        with self.lock:
            hdr, bufs = self._init_msg
            return pack(dict(hdr, history=list(self.history), feed=list(self.feed)[-30:]), bufs)

    def _pack_fields(self):
        env = self.env
        f = np.empty((env.H, env.W, 4), np.uint8)
        f[..., 0] = np.clip(env.plant * 255.0, 0, 255)
        f[..., 1] = np.clip(env.snow * 255.0, 0, 255)
        burn = float(env.cfg.disturbances.fire_burn_ticks)
        f[..., 2] = np.where(env.fire > 0, np.clip(80 + 175 * env.fire / burn, 0, 255), 0)
        flags = (env.flood > 0).astype(np.uint8)
        flags |= (env.fresh & ~env.fresh_now).astype(np.uint8) << 1
        flags |= (env.food_item > 0.02).astype(np.uint8) << 2
        flags |= ((env.terrain == C.FOREST) & (env.wood < 0.15)).astype(np.uint8) << 3
        flags |= (env.built > 0).astype(np.uint8) << 4
        f[..., 3] = flags
        fruit = np.clip(env.tree_fruit / max(1.0, env.cfg.resources.fruit_max) * 255, 0, 255).astype(np.uint8)
        alive_trees = (env.tree_state == 0).astype(np.uint8) if env.tree_state.size else np.zeros(0, np.uint8)
        self.fields = pack({"type": "fields", "tick": env.tick}, {"field": f.reshape(-1), "fruit": fruit,
                                                                  "tree_alive": alive_trees})
        self.fields_id += 1

    def _clark_evans(self, xy: np.ndarray) -> float:
        n = len(xy)
        if n < 3:
            return float("nan")
        d, _ = cKDTree(xy).query(xy, k=2)
        return float(d[:, 1].mean() / (0.5 * math.sqrt(self.land_cells / n)))

    def _sample(self):
        env = self.env
        al = env.alive
        A = env.A[al]
        xy = np.stack([env.I[al, I_X], env.I[al, I_Y]], 1).astype(np.float64)
        ec = env.ev_counts
        s = dict(
            t=env.tick, pop=int(al.sum()),
            e=float(A[:, A_ENERGY].mean()) if len(A) else 0, hy=float(A[:, A_HYDRATION].mean()) if len(A) else 0,
            hp=float(A[:, A_HEALTH].mean()) if len(A) else 0, w=float(A[:, A_WARMTH].mean()) if len(A) else 0,
            sp=int(env.fcount[0]), lp=int(env.fcount[1]), pr=int(env.fcount[2]),
            plant=float(env.plant.sum()), fruit=float(env.tree_fruit.sum()), ce=self._clark_evans(xy),
            age=float(env.I[al, I_AGE].mean() / env.tpd) if al.any() else 0, temp=round(env.global_temp(), 2),
            deaths=int(env.deaths_by_cause.sum()), dc=[int(v) for v in env.deaths_by_cause],
            dpy=float(env.deaths_by_cause.sum() / max(1e-9, self.agent_ticks / env.year_len)),
            eat=int(ec[C.EV_EAT_PLANT] + ec[C.EV_EAT_FRUIT] + ec[C.EV_EAT_MEAT]), drink=int(ec[C.EV_DRINK]),
            ks=int(ec[C.EV_KILL_SMALL]), kl=int(ec[C.EV_KILL_LARGE]), pa=int(ec[C.EV_PREDATOR_ATTACK]),
            scr=self.scrounge,
        )
        s = {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else (None if isinstance(v, float) else v))
             for k, v in s.items()}
        self.history.append(s)
        self.new_samples.append(s)

    def _harvest_events(self):
        ev = self.env.last_events
        if not len(ev):
            return
        typ = ev[:, 1].astype(np.int64)
        scr = (typ == C.EV_EAT_MEAT) & (ev[:, 3] != 0) & (ev[:, 3] != ev[:, 2])
        self.scrounge += int(scr.sum())
        tpd = self.env.tpd
        for r in ev[np.isin(typ, FEED_TYPES)]:
            k = int(r[1])
            d = dict(t=int(r[0]), k=C.EVENT_NAMES[k], x=int(r[4]), y=int(r[5]))
            if k == C.EV_DEATH:
                d.update(uid=int(r[2]), cause=DEATH_NAMES[int(r[6])], age=round(r[3] / tpd, 1))
            elif k == C.EV_BIRTH:
                d.update(uid=int(r[2]), parent=int(r[3]))
            elif k == C.EV_KILL_LARGE:
                d.update(uid=int(r[2]), n=int(r[3]))
            else:
                d.update(v=float(r[6]))
            self.feed.append(d)
            self.fx.append(d)
        sel = np.isin(typ, FX_TYPES) | scr
        if sel.any() and len(self.fx) < 400:
            for r, is_scr in zip(ev[sel][:200], scr[sel][:200]):
                self.fx.append(dict(k="scrounge" if is_scr else C.EVENT_NAMES[int(r[1])], x=int(r[4]), y=int(r[5]),
                                    uid=int(r[2])))

    def _pack_frame(self):
        env = self.env
        al = np.flatnonzero(env.alive)
        I, A, G = env.I, env.A, env.G
        n = al.size
        ag = np.zeros((n, 8), np.uint16)
        uid = I[al, I_UID]
        ag[:, 0] = I[al, I_X]
        ag[:, 1] = I[al, I_Y]
        ag[:, 2] = uid & 0xFFFF
        ag[:, 3] = (uid >> 16) & 0xFFFF
        carry = np.where(A[al, A_CARRY_AMT] > 0, I[al, I_CARRY_TYPE], 0)
        ag[:, 4] = (I[al, I_LAST_ACTION] & 0xFF) | (carry.astype(np.int64) << 8)
        q = lambda v: np.clip(v * 255, 0, 255).astype(np.uint16)  # noqa: E731
        ag[:, 5] = q(A[al, A_ENERGY]) | (q(A[al, A_HYDRATION]) << 8)
        ag[:, 6] = q(A[al, A_HEALTH]) | (q(A[al, A_WARMTH]) << 8)
        hue = np.clip((G[al, 3] + 1) * 0.5 * 255, 0, 255).astype(np.uint16)
        tone = np.clip((G[al, 4] + 1) * 0.5 * 15, 0, 15).astype(np.uint16)
        size = np.clip((G[al, 1] + 1) * 0.5 * 15, 0, 15).astype(np.uint16)
        ag[:, 7] = hue | (tone << 8) | (size << 12)
        fa = np.flatnonzero(env.FI[:, F_ALIVE] == 1)
        fn = np.zeros((fa.size, 4), np.uint16)
        fn[:, 0] = env.FI[fa, F_X]
        fn[:, 1] = env.FI[fa, F_Y]
        hp = np.clip(env.FF[fa, F_HEALTH], 0, 1)
        fn[:, 2] = env.FI[fa, F_SPECIES] | ((env.FI[fa, F_FLEE] > 0).astype(np.int64) << 4) | \
            ((hp * 15).astype(np.int64) << 8)
        fn[:, 3] = fa & 0xFFFF
        now = time.perf_counter()
        tw = self.tps_window
        tps = (tw[-1][1] - tw[0][1]) / (tw[-1][0] - tw[0][0]) if len(tw) > 1 and tw[-1][0] > tw[0][0] else 0.0
        if self.paused or (tw and now - tw[-1][0] > 1.0):
            tps = 0.0
        hdr = dict(
            type="frame", tick=env.tick, tod=env.time_of_day, season=env.season, day=env.day, year=env.year,
            weather=int(env.weather), temp=round(env.global_temp(), 2), night=round(env.night_level(), 3),
            drought=env.drought_left, cold_snap=env.cold_snap_left, famine=env.famine_left,
            fires=int(env.n_fire), paused=self.paused, speed=self.speed, tps=round(tps, 1),
            target_tps=(BASE_TPS * self.speed) if self.speed else None, version=self.version,
            samples=self.new_samples, fx=self.fx, wall=now,
            rec=dict(rows=self.recorder.rows, ticks=self.recorder.ticks, mb=round(self.recorder.bytes / 1e6, 1),
                     obs=self.recorder.rec_obs) if self.recorder else None,
        )
        self.new_samples = []
        self.fx = []
        self.frame = pack(hdr, {"agents": ag.reshape(-1), "fauna": fn.reshape(-1)})
        self.frame_id += 1
        self.details = {u: self._detail(u) for u in list(self.selected)}

    def _detail(self, uid: int) -> dict:
        env = self.env
        slots = np.flatnonzero((env.I[:, I_UID] == uid) & env.alive)
        if not slots.size:
            return {"uid": uid, "alive": False}
        i = int(slots[0])
        I, A, S = env.I, env.A, env.S
        hl = env.HIST.shape[1]
        ptr = int(I[i, I_HIST_PTR])
        hist = [int(a) for a in np.roll(env.HIST[i], -ptr) if a >= 0]
        return dict(
            uid=uid, alive=True, slot=i, x=int(I[i, I_X]), y=int(I[i, I_Y]), gen=int(I[i, I_GEN]),
            parent=int(I[i, I_PARENT]), age_days=round(I[i, I_AGE] / env.tpd, 2),
            lifespan_days=round(I[i, I_LIFESPAN] / env.tpd, 1), energy=float(A[i, A_ENERGY]),
            hydration=float(A[i, A_HYDRATION]), health=float(A[i, A_HEALTH]), warmth=float(A[i, A_WARMTH]),
            temp_eff=round(float(A[i, A_TEMP_EFF]), 1), sheltered=bool(I[i, I_SHELTERED]),
            carry=["none", "food", "wood", "stone"][int(I[i, I_CARRY_TYPE])], carry_amt=round(float(A[i, A_CARRY_AMT]), 3),
            action=env.action_names[int(I[i, I_LAST_ACTION])] if 0 <= I[i, I_LAST_ACTION] < env.n_actions else "?",
            history=[env.action_names[a] for a in hist[-hl:]], genes=[round(float(g), 3) for g in env.G[i]],
            stats={n: round(float(v), 3) for n, v in zip(STAT_NAMES, S[i])},
            mind=self._mind(i),
        )

    def _mind(self, i: int):
        """A learner's internals for slot i at its last decision: value estimate and action probabilities."""
        last = getattr(self.policy, "last", None)
        if not last:
            return None
        return dict(value=round(float(last["value"][i]), 4),
                    probs={n: round(float(p), 4) for n, p in zip(self.env.action_names, last["probs"][i])})

    # ---------------------------------------------------------------- commands
    def _apply(self, cmd: dict):
        env = self.env
        c = cmd.get("cmd")
        rec = dict(wall=dt.datetime.now().isoformat(timespec="milliseconds"), tick=env.tick, **cmd)
        restart = None
        if c == "pause":
            self.paused = True
        elif c == "play":
            self.paused = False
        elif c == "speed":
            self.speed = max(0.0, float(cmd["value"]))
        elif c == "step":
            self._step()
        elif c == "select":
            self.selected = {int(u) for u in cmd.get("uids", [])}
            return                                   # UI state only; not logged (high rate)
        elif c in ("fire", "flood"):
            x, y = int(cmd["x"]), int(cmd["y"])
            if not (0 <= x < env.W and 0 <= y < env.H):
                return
            if c == "fire":
                env.trigger_fire(x, y)
            else:
                env.trigger_flood(x, y, radius=int(cmd.get("radius", 30)))
        elif c == "drought":
            env.trigger_drought(int(cmd.get("days", 10)))
        elif c == "cold_snap":
            env.trigger_cold_snap(int(cmd.get("days", 5)))
        elif c == "famine":
            env.trigger_famine(int(cmd.get("days", 16)))
        elif c == "restart":
            restart = (int(cmd.get("seed", env.seed)), cmd.get("policy", self.policy_name),
                       cmd.get("n_agents") or env.cfg.agents.n_initial)
        else:
            return
        self.logger.log_control(rec)
        if c in ("fire", "flood", "drought", "cold_snap", "famine"):
            self._pack_fields()
        if restart:
            with self.lock:
                self._close_run()
                self._build(*restart)
                self._pack_fields()
                self._pack_frame()

    # ---------------------------------------------------------------- main loop
    def _step(self):
        env = self.env
        rec = self.recorder
        if rec:
            rec.pre(self.obs)
        actions = self.policy.act(self.obs, env.alive)
        self.obs, rewards, _, info = env.step(actions)
        if rec:
            rec.post(actions, rewards, info, policy_state=getattr(self.policy, "mode", None),
                     extras=getattr(self.policy, "last", None))
        if info["born"].any():
            self.policy.reset(info["born"])
        self.agent_ticks += int(info["alive"].sum())
        self._harvest_events()
        if env.tick % SAMPLE_EVERY == 0:
            self._sample()
        if env.tick % env.cfg.logging.metrics_every == 0:
            self.logger.log_metrics()
            self.logger.collect_events()
        self.tps_window.append((time.perf_counter(), env.tick))

    def run(self):
        self._pack_fields()
        self._pack_frame()
        self.ready.set()
        next_step = time.perf_counter()
        next_frame = next_fields = time.perf_counter()
        while not self.stop_flag.is_set():
            try:
                while True:
                    self._apply(self.commands.get_nowait())
            except queue.Empty:
                pass
            now = time.perf_counter()
            if not self.paused:
                if self.speed == 0:
                    with self.lock:
                        self._step()
                else:
                    tps = BASE_TPS * self.speed
                    if now - next_step > 0.25:
                        next_step = now
                    n = 0
                    while next_step <= now and n < 400:
                        with self.lock:
                            self._step()
                        next_step += 1.0 / tps
                        n += 1
            now = time.perf_counter()
            if now >= next_frame:
                with self.lock:
                    self._pack_frame()
                next_frame = now + 1.0 / (FRAME_HZ if not self.paused else 10.0)
            if now >= next_fields:
                with self.lock:
                    self._pack_fields()
                next_fields = now + 0.5
            if self.paused or (self.speed and next_step > time.perf_counter()):
                wait = min(next_frame, next_step if not self.paused else next_frame) - time.perf_counter()
                time.sleep(max(0.0, min(wait, 0.01)))
        with self.lock:
            self._close_run()


# ---------------------------------------------------------------------------------------------
BCAST = web.AppKey("bcast", asyncio.Task)


def make_app(sim: Sim) -> web.Application:
    app = web.Application()
    clients: dict = {}                      # ws -> per-client state

    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=0, compress=False)
        await ws.prepare(request)
        st = {"version": -1, "frame": -1, "fields": -1, "selected": None}
        clients[ws] = st
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        cmd = json.loads(msg.data)
                    except ValueError:
                        continue
                    if cmd.get("cmd") == "select":
                        st["selected"] = cmd.get("uids", [None])[0] if cmd.get("uids") else None
                        sel = {s["selected"] for s in clients.values() if s["selected"] is not None}
                        sim.commands.put({"cmd": "select", "uids": sorted(sel)})
                    else:
                        sim.commands.put(cmd)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            clients.pop(ws, None)
        return ws

    async def broadcaster(app):
        try:
            while True:
                await asyncio.sleep(1.0 / FRAME_HZ)
                if not clients or not sim.ready.is_set():
                    continue
                for ws, st in list(clients.items()):
                    if ws.closed:
                        continue
                    try:
                        if st["version"] != sim.version:
                            await ws.send_bytes(sim.init_message())
                            st["version"] = sim.version
                            st["fields"] = st["frame"] = -1
                        if st["fields"] != sim.fields_id and sim.fields is not None:
                            st["fields"] = sim.fields_id
                            await ws.send_bytes(sim.fields)
                        if st["frame"] != sim.frame_id and sim.frame is not None:
                            st["frame"] = sim.frame_id
                            await ws.send_bytes(sim.frame)
                            sel = st["selected"]
                            if sel is not None and sel in sim.details:
                                await ws.send_str(json.dumps({"type": "detail", **sim.details[sel]}))
                    except (ConnectionResetError, RuntimeError):
                        pass
        except asyncio.CancelledError:
            pass

    async def on_startup(app):
        app[BCAST] = asyncio.create_task(broadcaster(app))

    async def on_shutdown(app):
        # close open WebSockets first, otherwise aiohttp waits for them (up to shutdown_timeout) before exiting
        app[BCAST].cancel()
        for ws in list(clients):
            await ws.close(code=1001, message=b"server shutdown")

    async def on_cleanup(app):
        app[BCAST].cancel()

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/vendor/three", THREE)
    app.router.add_static("/", STATIC)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)
    return app


def start_sim_thread(sim: Sim) -> threading.Thread:
    th = threading.Thread(target=sim.run, name="genesis-sim", daemon=True)
    th.start()
    return th


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="heuristic", choices=POLICIES + ("ppo",))
    ap.add_argument("--checkpoint", default=None, help="trained learner (runs/stage2/<name>/checkpoints/*.pt); enables --policy ppo")
    ap.add_argument("--agents", type=int, default=None, help="initial agents (default: config agents.n_initial)")
    ap.add_argument("--config", default=None, help="experiment YAML deep-merged over configs/default.yaml")
    ap.add_argument("--speed", type=float, default=1.0, help=f"multiple of {BASE_TPS:g} ticks/s; 0 = max")
    ap.add_argument("--no-agent-log", action="store_true", help="don't append a session summary to logs/AGENT_LOG.md")
    ap.add_argument("--no-record", action="store_true", help="disable the per-tick trajectory recorder")
    ap.add_argument("--record-obs", action="store_true", help="also store every observation (~2 KB/agent/tick)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    if not (THREE / "build" / "three.module.js").exists():
        raise SystemExit("three.js missing: run `make viewer-setup` (npm install in viewer/)")
    print(f"GENESIS viewer: building world (seed {a.seed}, policy {a.policy}) ...", flush=True)
    sim = Sim(a.seed, a.policy, a.agents, a.config, speed=a.speed, agent_log=not a.no_agent_log,
              record=not a.no_record, record_obs=True if a.record_obs else None, checkpoint=a.checkpoint)
    th = start_sim_thread(sim)
    sim.ready.wait()
    print(f"world ready in {sim.build_s:.1f}s; {sim.env.population()} agents; logging to "
          f"{_rel(sim.logger.dir)}", flush=True)
    print(f"open http://{a.host}:{a.port}", flush=True)
    try:
        web.run_app(make_app(sim), host=a.host, port=a.port, print=None, shutdown_timeout=3)
    finally:
        sim.stop_flag.set()
        th.join(timeout=10)
        print(f"session closed; run saved to {_rel(sim.logger.dir)}", flush=True)


if __name__ == "__main__":
    main()
