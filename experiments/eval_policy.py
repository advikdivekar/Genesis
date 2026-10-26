"""Evaluate a trained learner exactly like the baselines (same seeds, length, population, statistics).

    python experiments/eval_policy.py runs/stage2/gru_s0/checkpoints/last.pt --name gru_s0

Seeds 0-4 are the baseline seeds and are never used in training (train_ppo.py uses 1000+). Every evaluation run is
fully recorded by the flight recorder, including the learner's value, action probabilities and log-prob.
Compares against reports/stage1/baselines.json (deaths/agent-year, KM survival at 1 yr, Hedges' g).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.learned import LearnedPolicy  # noqa: E402
from analysis.survival import bootstrap_ci, hedges_g, km_at, lifetimes  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.layout import I_UID  # noqa: E402
from sim.logger import RunLogger, append_agent_log  # noqa: E402
from sim.recorder import TrajectoryRecorder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run_one(ckpt, seed, years, n, out, greedy):
    env = GenesisEnv(load_config(overrides={"agents": {"n_initial": n, "max_agents": n}}), seed=seed)
    pol = LearnedPolicy(ckpt, env, greedy=greedy, seed=seed)
    log = RunLogger(out / f"s{seed}", env, meta=dict(policy="ppo", checkpoint=str(ckpt), experiment="stage2_eval"))
    rec = TrajectoryRecorder(env, out / f"s{seed}", meta=dict(policy="ppo", checkpoint=str(ckpt)))
    initial = env.I[env.alive, I_UID].copy()
    obs = env.observe()
    t0 = time.perf_counter()
    for k in range(int(years * env.year_len)):
        rec.pre(obs)
        a = pol.act(obs, env.alive)
        obs, r, died, info = env.step(a)
        rec.post(a, r, info, extras=pol.last)
        if info["born"].any():
            pol.reset(info["born"])
        if k % env.cfg.logging.metrics_every == 0:
            log.log_metrics({"policy": "ppo"})
        if k % 1000 == 0:
            log.collect_events()
    rec.close()
    log.collect_events()
    ev = np.concatenate(log._events) if log._events else np.zeros((0, 7))
    dur, obsd, cause = lifetimes(ev, initial, env.tick, start_tick=env.spinup_ticks)
    res = dict(policy="ppo", seed=seed, wall_s=round(time.perf_counter() - t0, 1),
               deaths_per_agent_year=int(obsd.sum()) / (dur.sum() / env.year_len),
               km_survival_1yr=km_at(dur, obsd, env.year_len), median_lifetime_days=float(np.median(dur) / env.tpd),
               deaths_by_cause={DEATH_NAMES[c]: int((cause == c).sum()) for c in range(len(DEATH_NAMES)) if (cause == c).any()})
    log.close(summary=dict(result=res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    a = ap.parse_args()
    out = ROOT / "runs" / "stage2_eval" / a.name
    res = []
    for s in range(a.seeds):
        r = run_one(a.checkpoint, s, a.years, a.agents, out, a.greedy)
        res.append(r)
        print(f"seed {s}: deaths/agent-yr {r['deaths_per_agent_year']:.2f} S(1yr) {r['km_survival_1yr']:.3f} "
              f"{r['deaths_by_cause']} ({r['wall_s']}s)", flush=True)
    base = json.loads((ROOT / "reports" / "stage1" / "baselines.json").read_text())
    d = [r["deaths_per_agent_year"] for r in res]
    summ = dict(deaths_per_agent_year=bootstrap_ci(d), km_survival_1yr=bootstrap_ci([r["km_survival_1yr"] for r in res]))
    comp = {}
    for p in ("random", "greedy", "heuristic"):
        bd = [r["deaths_per_agent_year"] for r in base["per_run"] if r["policy"] == p]
        comp[p] = dict(baseline=bootstrap_ci(bd), hedges_g_learner_minus_baseline=hedges_g(d, bd))
    outj = ROOT / "reports" / "stage2" / f"eval_{a.name}.json"
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(dict(args=vars(a), per_run=res, summary=summ, vs_baselines=comp), indent=2))
    m = summ["deaths_per_agent_year"]
    lines = [f"Checkpoint `{a.checkpoint}`, {a.seeds} held-out seeds × {a.years} yr, {a.agents} agents, "
             f"{'greedy' if a.greedy else 'sampled'} actions. Same protocol and statistics as `run_baselines.py`.",
             "", "| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |", "|---|---|---|",
             f"| **learner ({a.name})** | **{m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}]** | |"]
    for p, c in comp.items():
        b = c["baseline"]
        lines.append(f"| {p} | {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] | {c['hedges_g_learner_minus_baseline']:+.2f} |")
    km = summ["km_survival_1yr"]
    causes = {}
    for r in res:
        for k, v in r["deaths_by_cause"].items():
            causes[k] = causes.get(k, 0) + v
    lines += ["", f"KM survival at 1 yr {km[0]:.3f} [{km[1]:.3f}, {km[2]:.3f}]; deaths by cause (summed) {causes}.",
              f"Fully recorded: `runs/stage2_eval/{a.name}/s*/trajectory/`."]
    append_agent_log(f"Stage 2 evaluation: {a.name}", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
