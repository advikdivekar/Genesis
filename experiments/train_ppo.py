"""Train the Stage-2 recurrent PPO learner (parameter-shared, survival reward only).

    python experiments/train_ppo.py --name gru_s0 --iters 600                 # GRU memory
    python experiments/train_ppo.py --name ff_s0 --iters 600 --no-memory      # memoryless ablation

Worlds: training uses seeds 1000+ (a new world every --world-every iterations); evaluation seeds 0-4 (the
baseline seeds) are never trained on. Fixed population of --agents (respawn), all capabilities off.
Logs: runs/stage2/<name>/{train.jsonl, checkpoints/, trajectory_*/}. Rollouts are recorded by the flight recorder
(sampled: 1 of every --record-every rollouts, all ticks of it; births/interventions always) with the learner's
value, action probabilities and log-prob per agent per tick.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.ppo import PPO, PPOConfig, Rollout  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.logger import append_agent_log, git_commit  # noqa: E402
from sim.recorder import TrajectoryRecorder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_env(seed, n):
    return GenesisEnv(load_config(overrides={"agents": {"n_initial": n, "max_agents": n}}), seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="learner seed (network init, sampling, world sequence)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--world-every", type=int, default=40, help="iterations per training world")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--record-every", type=int, default=1, help="record 1 of k rollouts in full (0 = off)")
    ap.add_argument("--min-free-gb", type=float, default=15.0, help="stop recording below this free disk")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = ROOT / "runs" / "stage2" / a.name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig(recurrent=not a.no_memory)
    torch.set_num_threads(5)
    probe = make_env(1000 + a.seed * 100, a.agents)
    ppo = PPO(probe.n_actions, probe.obs_shapes, cfg, a.device, seed=a.seed)
    start_it = 0
    if a.resume and (out / "checkpoints" / "last.pt").exists():
        ck = ppo.load(out / "checkpoints" / "last.pt")
        start_it = ck["iter"] + 1
    (out / "config.json").write_text(json.dumps(dict(args=vars(a), ppo=cfg.__dict__, git_commit=git_commit(),
                                                     torch=torch.__version__, n_actions=probe.n_actions,
                                                     action_names=probe.action_names), indent=2))
    N, T = a.agents, cfg.rollout
    roll = Rollout(T, N, probe.obs_shapes)
    log = open(out / "train.jsonl", "a")
    env = rec = None
    world = -1
    t_start = time.perf_counter()
    for it in range(start_it, a.iters):
        w = it // a.world_every
        if w != world:
            if rec:
                rec.close()
            world = w
            env = probe if (it == 0 and w == 0) else make_env(1000 + a.seed * 100 + w, a.agents)
            probe = None
            obs = env.observe()
            h = torch.zeros(N, cfg.hidden, device=ppo.device)
            reset = np.ones(N, np.float32)
            rec = (TrajectoryRecorder(env, out / f"world_{w:03d}", meta=dict(source="train_ppo", learner=a.name,
                                      world_index=w, note="sampled rollouts; policy changes every iteration"))
                   if a.record_every else None)
        recording = bool(rec) and it % a.record_every == 0 and shutil.disk_usage(ROOT).free / 1e9 > a.min_free_gb
        h0 = h.detach().clone()
        t0 = time.perf_counter()
        deaths = 0
        causes = np.zeros(8, np.int64)
        for t in range(T):
            roll.grid[t], roll.selff[t], roll.nb[t], roll.heard[t] = obs["grid"], obs["self"], obs["neighbors"], obs["heard"]
            roll.reset[t] = reset
            alive = env.alive
            roll.alive[t] = alive
            act, logp, val, h, probs = ppo.act(obs, h, reset)
            if recording:
                rec.pre(obs)
            dc0 = env.deaths_by_cause.copy()
            obs, r, died, info = env.step(act)
            if rec:
                if recording:
                    rec.post(act, r, info, extras=dict(value=val, logp=logp, probs=probs))
                else:
                    rec.skip(info)
            roll.action[t], roll.logp[t], roll.value[t] = act, logp, val
            roll.reward[t], roll.done[t] = r, died
            reset = info["born"].astype(np.float32)
            deaths += int(died.sum())
            causes += env.deaths_by_cause - dc0
        t_collect = time.perf_counter() - t0
        with torch.no_grad():
            _, last_v, _ = ppo.net.step(*(ppo.t(obs[k]) for k in ("grid", "self", "neighbors", "heard")), h,
                                        ppo.t(reset))
        stats = ppo.update(roll, h0, last_v.cpu().numpy() * (1.0 - roll.done[-1]))
        t_all = time.perf_counter() - t0
        agent_years = roll.alive.sum() / env.year_len
        row = dict(iter=it, world=w, tick=env.tick, agent_steps=(it + 1) * T * N, mean_reward=float(roll.reward[roll.alive > 0].mean()),
                   deaths=deaths, deaths_per_agent_year=deaths / max(agent_years, 1e-9),
                   causes={n: int(c) for n, c in zip(DEATH_NAMES, causes) if c},
                   mean_energy=float(env.A[env.alive, 0].mean()), mean_hydration=float(env.A[env.alive, 1].mean()),
                   recorded=recording, collect_s=round(t_collect, 2), total_s=round(t_all, 2),
                   agent_steps_per_s=round(T * N / t_all), wall_s=round(time.perf_counter() - t_start, 1), **stats)
        log.write(json.dumps(row, default=float) + "\n")
        log.flush()
        if it % 10 == 0:
            print(f"it {it:4d} w{w} deaths/agent-yr {row['deaths_per_agent_year']:.2f} r {row['mean_reward']:+.5f} "
                  f"ent {stats['ent']:.2f} kl {stats['kl']:.4f} {row['agent_steps_per_s']} sps", flush=True)
        if it % 25 == 0 or it == a.iters - 1:
            ppo.save(out / "checkpoints" / "last.pt", dict(iter=it))
            if it % 100 == 0 or it == a.iters - 1:
                ppo.save(out / "checkpoints" / f"iter_{it:05d}.pt", dict(iter=it))
    if rec:
        rec.close()
    log.close()
    rows = [json.loads(line) for line in open(out / "train.jsonl")]
    k = max(1, len(rows) // 10)
    first, last = rows[:k], rows[-k:]
    f = lambda rr: np.mean([x["deaths_per_agent_year"] for x in rr])  # noqa: E731
    append_agent_log(f"Stage 2 training: {a.name}", [
        f"PPO ({'GRU' if cfg.recurrent else 'memoryless'}), {a.agents} agents, {len(rows)} iterations × {T} ticks "
        f"({rows[-1]['agent_steps']:,} agent-steps), training worlds 1000+ (eval worlds 0-4 held out).",
        f"- deaths/agent-yr during training: first {k} iters {f(first):.2f} → last {k} iters {f(last):.2f} "
        "(training-time rate with exploration; not an evaluation).",
        f"- last-iteration causes: {rows[-1]['causes']}; entropy {rows[-1]['ent']:.2f}.",
        f"- logs: `runs/stage2/{a.name}/train.jsonl`, checkpoints in `runs/stage2/{a.name}/checkpoints/`."])


if __name__ == "__main__":
    main()
