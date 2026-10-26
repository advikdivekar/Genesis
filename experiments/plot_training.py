"""Learning curves from runs/stage2/<name>/train.jsonl, with the baselines' evaluated mortality for reference.

    python experiments/plot_training.py gru_s0 ff_s0       -> reports/stage2/fig_training.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COL = {"random": "#999", "greedy": "#e8a33d", "heuristic": "#3d7fd6"}


def smooth(x, k=15):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main(names):
    base = json.loads((ROOT / "reports" / "stage1" / "baselines.json").read_text())["summary"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    for name in names:
        rows = [json.loads(line) for line in open(ROOT / "runs" / "stage2" / name / "train.jsonl")]
        steps = np.array([r["agent_steps"] for r in rows]) / 1e6
        d = smooth([r["deaths_per_agent_year"] for r in rows])
        s = steps[len(steps) - len(d):]
        ax[0].plot(s, d, label=name)
        thirst = smooth([r["causes"].get("thirst", 0) / max(1, r["deaths"]) for r in rows])
        ax[1].plot(s, thirst, label=name)
        ax[2].plot(steps, [r["ent"] for r in rows], label=name)
        ax[3].plot(s, smooth([r["mean_hydration"] for r in rows]), label=f"{name} hydration")
        ax[3].plot(s, smooth([r["mean_energy"] for r in rows]), "--", label=f"{name} energy")
    for p, c in COL.items():
        ax[0].axhline(base[p]["deaths_per_agent_year"][0], color=c, ls=":", label=f"{p} (evaluated)")
    ax[0].set(title="Mortality during training (15-iter mean)", xlabel="agent-steps (M)", ylabel="deaths / agent-year")
    ax[1].set(title="Share of deaths from thirst", xlabel="agent-steps (M)", ylim=(0, 1))
    ax[2].set(title="Policy entropy (nats; uniform = 2.40)", xlabel="agent-steps (M)")
    ax[3].set(title="Mean body state of living agents", xlabel="agent-steps (M)", ylim=(0, 1))
    for a in ax:
        a.legend(frameon=False, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Training-time rates use the current (exploring) policy on training worlds; held-out evaluation is separate.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "reports" / "stage2" / "fig_training.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["gru_s0"])
