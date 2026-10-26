"""Flight recorder: every agent's complete state, action and outcome at every tick.

One tick is the simulation's atomic time step (every agent acts exactly once per tick; nothing happens between
ticks), so this is the finest-grained record that exists. Per tick the recorder stores:

  agents   one row per agent alive at the START of the tick: the state it acted from (position, body, age,
           carrying, shelter, felt temperature, all lifetime counters, the policy's internal state), the action it
           took, the reward it received and whether/why it died during the tick  -> exact (s_t, a_t, r_t, done_t)
  births   one row per agent ever created (founders at run start, respawns, offspring): uid, parent, generation,
           birth tick and place, lifespan, full gene vector
  world    one row per tick: clock, season, weather, temperature, light, view radius, disturbances, population,
           fauna counts, vegetation/fruit/food totals, fire/flood/snow extent, event count
  fauna    one row per animal per tick (species, position, energy, health, age, fleeing)
  events   every event of the tick (eats, drinks, kills, attacks, births, deaths, disturbances; sim.constants)
  manual   every manual intervention, with everything needed to replay it
  obs      optional (logging.trajectory_obs): each agent's exact observation (grid/self/neighbors/heard)

Observations are not stored by default because they are a deterministic function of the recorded run:
`analysis.replay` re-simulates from (seed, config, recorded actions, manual interventions), checks every tick
against this record, and can return any agent's exact observation at any tick.

Files: <run_dir>/trajectory/{run.json, schema.json, chunk_000000.npz, ..., index.json}. Chunks are compressed
and written by a background thread so the simulation never waits on the disk.
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from numba import njit

from .constants import DEATH_NAMES, EV_DEATH, EVENT_NAMES, OBS_CHANNELS, SELF_FEATURES, NEIGHBOR_FEATURES
from .layout import (A_CARRY_AMT, A_ENERGY, A_HEALTH, A_HYDRATION, A_TEMP_EFF, A_WARMTH, F_AGE, F_ALIVE, F_ENERGY,
                     F_FLEE, F_HEALTH, F_SPECIES, F_X, F_Y, I_AGE, I_CARRY_TYPE, I_GEN, I_LIFESPAN, I_PARENT,
                     I_REPRO_COOLDOWN, I_SHELTERED, I_UID, I_X, I_Y, STAT_NAMES)

FORMAT_VERSION = 1

AGENT_FIELDS = [
    ("tick", "u4", "absolute env tick at which the action was taken (includes spin-up; run tick = tick - spinup_ticks)"),
    ("uid", "u4", "agent id (unique for the whole run)"),
    ("slot", "u2", "array slot the agent occupies"),
    ("x", "u2", "cell column at the start of the tick"),
    ("y", "u2", "cell row at the start of the tick"),
    ("age", "u4", "age in ticks"),
    ("gen", "u2", "generation (0 = founder or respawn)"),
    ("energy", "f4", "body energy [0,1]"),
    ("hydration", "f4", "body hydration [0,1]"),
    ("health", "f4", "health [0,1]"),
    ("warmth", "f4", "warmth [0,1]"),
    ("temp_eff", "f4", "effective (felt) temperature, deg C, computed last tick"),
    ("carry_type", "u1", "0 none, 1 food, 2 wood, 3 stone"),
    ("carry_amt", "f4", "amount carried"),
    ("sheltered", "u1", "1 if sheltered (cave/built) last tick"),
    ("repro_cooldown", "u4", "ticks until it may reproduce again"),
    ("action", "u1", "action taken this tick (index into action_names)"),
    ("reward", "f4", "reward received for this tick (own homeostatic drive change + death penalty)"),
    ("died", "u1", "1 if the agent died during this tick"),
    ("death_cause", "i1", "index into death_names if died, else -1"),
    ("policy_state", "i2", "policy's internal mode (heuristic: 0 none, 1 drinking, 2 eating; -1 if none)"),
] + [(f"stat_{n}", "f4", f"lifetime counter '{n}' at the start of the tick") for n in STAT_NAMES]

BIRTH_FIELDS = [("uid", "u4", ""), ("parent", "u4", "0 = none (founder/respawn)"), ("gen", "u2", ""),
                ("tick", "u4", "absolute tick of birth (founders: run start)"), ("x", "u2", ""), ("y", "u2", ""),
                ("lifespan", "u4", "heritable lifespan in ticks"), ("founder", "u1", "1 if present at run start")]

WORLD_FIELDS = [
    ("tick", "u4", ""), ("day", "u4", "absolute day"), ("season", "u1", "0 spring .. 3 winter"),
    ("time_of_day", "f4", "0.25 = sunrise"), ("weather", "u1", "0 clear, 1 rain, 2 fog"),
    ("temp", "f4", "sea-level temperature, deg C"), ("night", "f4", "0 day .. 1 night"),
    ("view_radius", "u1", "agents' current view radius"), ("drought_left", "u2", "days"),
    ("cold_snap_left", "u2", "days"), ("famine_left", "u2", "days"), ("fire_cells", "u4", ""),
    ("flood_cells", "u4", ""), ("snow_cells", "u4", "cells with snow > 0.2"), ("population", "u2", "at tick start"),
    ("small_prey", "u2", ""), ("large_prey", "u2", ""), ("predators", "u2", ""), ("plant_total", "f4", ""),
    ("fruit_total", "f4", ""), ("food_items", "f4", "carcass/dropped food on the ground"), ("n_events", "u4", ""),
]

FAUNA_FIELDS = [("tick", "u4", ""), ("slot", "u4", "fauna slot (identity while alive)"), ("species", "u1",
                "0 small prey, 1 large prey, 2 predator"), ("x", "u2", ""), ("y", "u2", ""), ("energy", "f4", ""),
                ("health", "f4", ""), ("age", "u4", "ticks"), ("flee", "u1", "fleeing timer > 0")]


def _dtype(fields):
    return np.dtype([(n, t) for n, t, _ in fields])


AGENT_DTYPE = _dtype(AGENT_FIELDS)
BIRTH_DTYPE = _dtype(BIRTH_FIELDS)
WORLD_DTYPE = _dtype(WORLD_FIELDS)
FAUNA_DTYPE = _dtype(FAUNA_FIELDS)


@njit(cache=True)
def world_totals(fire, flood, snow, plant, food_item):
    """One pass over the map: (fire cells, flood cells, snow cells > 0.2, plant total, food on ground)."""
    nf = nfl = ns = 0
    p = f = 0.0
    h, w = fire.shape
    for y in range(h):
        for x in range(w):
            if fire[y, x] > 0:
                nf += 1
            if flood[y, x] > 0:
                nfl += 1
            if snow[y, x] > 0.2:
                ns += 1
            p += plant[y, x]
            f += food_item[y, x]
    return nf, nfl, ns, p, f


_I_COLS = np.array([I_UID, I_X, I_Y, I_AGE, I_GEN, I_CARRY_TYPE, I_SHELTERED, I_REPRO_COOLDOWN])
_A_COLS = np.array([A_ENERGY, A_HYDRATION, A_HEALTH, A_WARMTH, A_TEMP_EFF, A_CARRY_AMT])


def _build_rows(tick, slot, Icols, Acols, S, action, reward, died, cause, pstate):
    """Assemble structured agent rows from raw per-tick column copies (vectorised, once per chunk)."""
    r = np.zeros(len(slot), AGENT_DTYPE)
    r["tick"], r["slot"] = tick, slot
    for k, n in enumerate(("uid", "x", "y", "age", "gen", "carry_type", "sheltered", "repro_cooldown")):
        r[n] = Icols[:, k]
    for k, n in enumerate(("energy", "hydration", "health", "warmth", "temp_eff", "carry_amt")):
        r[n] = Acols[:, k]
    for k, n in enumerate(STAT_NAMES):
        r[f"stat_{n}"] = S[:, k]
    r["action"], r["reward"], r["died"], r["death_cause"], r["policy_state"] = action, reward, died, cause, pstate
    return r


class TrajectoryRecorder:
    """Usage (per tick):  rec.pre(obs) -> actions = policy(obs) -> env.step -> rec.post(actions, rewards, info)."""

    def __init__(self, env, run_dir, chunk_ticks: int | None = None, obs: bool | None = None,
                 fauna: bool | None = None, meta: dict | None = None):
        lc = env.cfg.logging
        self.env = env
        self.dir = Path(run_dir) / "trajectory"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_ticks = int(chunk_ticks or lc.get("trajectory_chunk", 480))
        self.rec_obs = bool(lc.get("trajectory_obs", False) if obs is None else obs)
        self.rec_fauna = bool(lc.get("trajectory_fauna", True) if fauna is None else fauna)
        self.n_genes = env.G.shape[1]
        self._agents, self._world, self._fauna, self._events, self._births, self._obs = [], [], [], [], [], []
        self._manual_seen = 0
        self._extras: dict[str, list] = {}
        self.skipped = 0
        self._manual: list[dict] = []
        self._ticks_in_chunk = 0
        self.chunks: list[dict] = []
        self.rows = 0
        self.ticks = 0
        self.bytes = 0
        self._pre = None
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._err: list[BaseException] = []
        self._writer = threading.Thread(target=self._write_loop, name="trajectory-writer", daemon=True)
        self._writer.start()
        self.start_tick = int(env.tick)
        run = dict(format_version=FORMAT_VERSION, seed=env.seed, config=env.cfg.to_dict(), start_tick=env.tick,
                   spinup_ticks=env.spinup_ticks, max_agents=env.N, ticks_per_day=env.tpd, year_len=env.year_len,
                   action_names=env.action_names, death_names=DEATH_NAMES, event_names=EVENT_NAMES,
                   event_columns=["tick", "type", "a_uid", "b_uid", "x", "y", "value"],
                   obs_recorded=self.rec_obs, fauna_recorded=self.rec_fauna, state_hash_start=env.state_hash(),
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {}))
        (self.dir / "run.json").write_text(json.dumps(run, indent=2, default=_json))
        schema = dict(agents=AGENT_FIELDS, births=BIRTH_FIELDS + [("genes", f"f4[{self.n_genes}]", "gene vector")],
                      world=WORLD_FIELDS, fauna=FAUNA_FIELDS,
                      events="float64 [n, 7]: tick, type (event_names), a_uid, b_uid, x, y, value",
                      obs=dict(grid=f"u1 [rows, {2 * env.R + 1}, {2 * env.R + 1}, {len(OBS_CHANNELS)}] channels "
                                    + ",".join(OBS_CHANNELS),
                               self="f4 " + ",".join(SELF_FEATURES), neighbors="f4 [rows, k, " +
                               ",".join(NEIGHBOR_FEATURES) + "]", heard="f4 [rows, n_symbols]",
                               note="row i of obs_* belongs to row i of agents in the same chunk"))
        (self.dir / "schema.json").write_text(json.dumps(schema, indent=1))
        al = np.flatnonzero(env.alive)
        self._add_births(al, founder=True)

    # ------------------------------------------------------------------------------------------
    def _add_births(self, slots, founder=False):
        if not len(slots):
            return
        env = self.env
        b = np.zeros(len(slots), BIRTH_DTYPE)
        I = env.I
        b["uid"] = I[slots, I_UID]
        b["parent"] = I[slots, I_PARENT]
        b["gen"] = I[slots, I_GEN]
        b["tick"] = env.tick if founder else env.tick - 1
        b["x"], b["y"] = I[slots, I_X], I[slots, I_Y]
        b["lifespan"] = I[slots, I_LIFESPAN]
        b["founder"] = 1 if founder else 0
        self._births.append((b, env.G[slots].astype(np.float32)))

    def pre(self, obs: dict | None = None):
        """Snapshot the state every live agent acts from. Call right before env.step()."""
        env = self.env
        al = np.flatnonzero(env.alive)
        raw = (al, env.I[al][:, _I_COLS], env.A[al][:, _A_COLS], env.S[al].astype(np.float32))
        w = np.zeros(1, WORLD_DTYPE)
        w["tick"], w["day"], w["season"] = env.tick, env.day, env.season
        w["time_of_day"], w["weather"], w["temp"] = env.time_of_day, env.weather, env.global_temp()
        w["night"], w["view_radius"] = env.night_level(), env.view_radius_now()
        w["drought_left"], w["cold_snap_left"], w["famine_left"] = env.drought_left, env.cold_snap_left, env.famine_left
        nf, nfl, ns, pt, ft = world_totals(env.fire, env.flood, env.snow, env.plant, env.food_item)
        w["fire_cells"], w["flood_cells"], w["snow_cells"], w["plant_total"], w["food_items"] = nf, nfl, ns, pt, ft
        w["population"] = al.size
        w["small_prey"], w["large_prey"], w["predators"] = env.fcount[0], env.fcount[1], env.fcount[2]
        w["fruit_total"] = env.tree_fruit.sum()
        fa = None
        if self.rec_fauna:
            j = np.flatnonzero(env.FI[:, F_ALIVE] == 1)
            fa = np.zeros(j.size, FAUNA_DTYPE)
            fa["tick"], fa["slot"] = env.tick, j
            fa["species"], fa["x"], fa["y"] = env.FI[j, F_SPECIES], env.FI[j, F_X], env.FI[j, F_Y]
            fa["energy"], fa["health"] = env.FF[j, F_ENERGY], env.FF[j, F_HEALTH]
            fa["age"], fa["flee"] = env.FI[j, F_AGE], env.FI[j, F_FLEE] > 0
        ob = None
        if self.rec_obs:
            if obs is None:
                raise ValueError("trajectory_obs is on: pass the observation dict to pre()")
            ob = {k: obs[k][al].copy() for k in ("grid", "self", "neighbors", "heard")}
        self._pre = (raw, w, fa, ob)

    def skip(self, info):
        """A tick that is NOT recorded (sampled recording): still log births and interventions so the
        births table and manual log stay complete."""
        self._add_births(np.flatnonzero(info["born"]))
        if len(self.env.manual_log) > self._manual_seen:
            self._manual.extend(dict(m) for m in self.env.manual_log[self._manual_seen:])
            self._manual_seen = len(self.env.manual_log)
        self.skipped += 1

    def post(self, actions, rewards, info, policy_state=None, extras: dict | None = None):
        """Complete the tick with what the agents did and what happened to them. Call right after env.step()."""
        if self._pre is None:
            raise RuntimeError("post() without pre()")
        env = self.env
        raw, w, fa, ob = self._pre
        al = raw[0]
        self._pre = None
        died = np.asarray(info["died"])[al].astype(np.uint8)
        cause = np.full(al.size, -1, np.int8)
        ev = env.last_events
        w["n_events"] = len(ev)
        if died.any() and len(ev):
            d = ev[ev[:, 1] == EV_DEATH]
            cmap = {int(u): int(c) for u, c in zip(d[:, 2], d[:, 6])}
            idx = np.flatnonzero(died)
            cause[idx] = [cmap.get(int(u), -1) for u in raw[1][idx, 0]]
        ps = np.asarray(policy_state)[al] if policy_state is not None else np.full(al.size, -1)
        r = (env.tick - 1, raw, np.asarray(actions)[al], np.asarray(rewards)[al], died, cause, ps)
        for k, v in (extras or {}).items():      # learner internals per agent row (value, action probabilities, ...)
            self._extras.setdefault(k, []).append(np.asarray(v)[al])
        born = np.flatnonzero(info["born"])
        self._add_births(born)
        if len(env.manual_log) > self._manual_seen:
            new = env.manual_log[self._manual_seen:]
            self._manual_seen = len(env.manual_log)
            self._manual.extend(dict(m) for m in new)
        self._agents.append(r)
        self._world.append(w)
        if fa is not None:
            self._fauna.append(fa)
        if len(ev):
            self._events.append(ev.copy())
        if ob is not None:
            self._obs.append(ob)
        self.rows += al.size
        self.ticks += 1
        self._ticks_in_chunk += 1
        if self._ticks_in_chunk >= self.chunk_ticks:
            self.flush()

    # ------------------------------------------------------------------------------------------
    def flush(self):
        if not self._agents:
            return
        if self._err:
            raise RuntimeError("trajectory writer failed") from self._err[0]
        k = len(self.chunks)
        A = self._agents
        n = [len(x[1][0]) for x in A]
        agents = _build_rows(np.repeat([x[0] for x in A], n), np.concatenate([x[1][0] for x in A]),
                             np.concatenate([x[1][1] for x in A]), np.concatenate([x[1][2] for x in A]),
                             np.concatenate([x[1][3] for x in A]), np.concatenate([x[2] for x in A]),
                             np.concatenate([x[3] for x in A]), np.concatenate([x[4] for x in A]),
                             np.concatenate([x[5] for x in A]), np.concatenate([x[6] for x in A]))
        data = dict(agents=agents, world=np.concatenate(self._world),
                    events=np.concatenate(self._events) if self._events else np.zeros((0, 7)),
                    manual=np.frombuffer(json.dumps(self._manual, default=_json).encode(), np.uint8))
        if self._births:
            data["births"] = np.concatenate([b for b, _ in self._births])
            data["births_genes"] = np.concatenate([g for _, g in self._births])
        if self._fauna:
            data["fauna"] = np.concatenate(self._fauna)
        for name, v in self._extras.items():
            data[f"x_{name}"] = np.concatenate(v)
        self._extras = {}
        if self._obs:
            for key in ("grid", "self", "neighbors", "heard"):
                data[f"obs_{key}"] = np.concatenate([o[key] for o in self._obs])
        meta = dict(chunk=k, file=f"chunk_{k:06d}.npz", tick_first=int(data["world"]["tick"][0]),
                    tick_last=int(data["world"]["tick"][-1]), agent_rows=int(len(data["agents"])),
                    births=int(len(data.get("births", []))), events=int(len(data["events"])),
                    manual=len(self._manual), state_hash_after=self.env.state_hash())
        self.chunks.append(meta)
        self._q.put((self.dir / meta["file"], data, meta))
        self._agents, self._world, self._fauna, self._events, self._births, self._obs = [], [], [], [], [], []
        self._manual = []
        self._ticks_in_chunk = 0

    def _write_loop(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            path, data, meta = item
            try:
                tmp = path.with_suffix(".tmp.npz")
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:   # np.load-compatible
                    for name, arr in data.items():
                        buf = io.BytesIO()
                        np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
                        zf.writestr(name + ".npy", buf.getvalue())
                tmp.replace(path)
                meta["bytes"] = path.stat().st_size
                self.bytes += meta["bytes"]
            except BaseException as e:   # surfaced on the next flush/close
                self._err.append(e)

    def close(self, summary: dict | None = None):
        self.flush()
        self._q.put(None)
        self._writer.join()
        if self._err:
            raise RuntimeError("trajectory writer failed") from self._err[0]
        idx = dict(format_version=FORMAT_VERSION, ticks=self.ticks, skipped_ticks=self.skipped, agent_rows=self.rows,
                   bytes=self.bytes,
                   start_tick=self.start_tick, end_tick=int(self.env.tick), chunks=self.chunks,
                   state_hash_end=self.env.state_hash(), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **(summary or {}))
        (self.dir / "index.json").write_text(json.dumps(idx, indent=1, default=_json))
        return idx


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
