# Stage 2 report: first learning agents (2026-10-26)

**Bottom line:** recurrent and memoryless PPO agents, rewarded only for their own homeostasis, learned in about
20M agent-steps to survive on unseen worlds about 2× better than random and greedy baselines. They are **not yet
distinguishable** from the hand-written heuristic. They solved thirst, predators and fire, and now hit the world's
**energy ceiling**, dying of starvation. At this budget, adding memory (GRU) **did not** improve overall survival. It cut
thirst deaths (large effect, CIs touching) but increased starvation. Hypotheses were preregistered
(`docs/PREREGISTRATION.md`, commit `4c84dec`) before any evaluation.

## Setup
- **Learner** (`agents/ppo.py`): one parameter-shared actor-critic for all agents. The input is the 11×11×16 view plus the agent's
  own body, its neighbours and heard signals. Layers: conv(32) → conv(64, stride 2) → fc 256, plus an fc 64 for the vector inputs, then fc 256 →
  **GRU 256** (or a feed-forward layer in the memoryless ablation) → policy and value heads, about 1.1M parameters. Each agent has its own
  memory state, reset when a new agent fills a slot.
- **Training** (`experiments/train_ppo.py`): PPO with γ 0.995, λ 0.95, clip 0.2, lr 3e-4, entropy 0.01, 4 epochs, 128-tick
  rollouts × 256 agents, 600 iterations (**19.7M agent-steps**, about 50 min each on an M3 Pro, MPS).
  - Fixed population, all capabilities off, reward = own drive reduction + death penalty only.
  - A new training world every 40 iterations (seeds 1000+). The evaluation worlds (0–4) are never seen in training.
  - Every training rollout is flight-recorded, including value, action probabilities and log-prob (about 2 GB per run).
- **Evaluation** (`experiments/eval_policy.py`): identical to the baselines, with seeds 0–4, 2 years, 256 agents and sampled actions.
  It reports deaths/agent-year, KM survival, and bootstrap CIs over seeds. All runs are fully recorded.

## Results (held-out worlds, 5 seeds × 2 years)
| policy | deaths / agent-yr (95% CI) | KM survival 1 yr | g vs heuristic |
|---|---|---|---|
| random | 3.83 [3.53, 4.19] | 0.001 | |
| greedy | 3.31 [2.92, 3.77] | 0.006 | |
| heuristic | 1.50 [1.04, 2.11] | 0.214 | |
| **GRU learner** (`gru_s0`) | **1.80 [1.52, 2.10]** | 0.062 | +0.50 |
| **memoryless learner** (`ff_s0`) | **1.58 [1.35, 1.83]** | 0.101 | +0.15 |

Death rate by cause, per agent-year (95% CI):
| cause | GRU | memoryless | heuristic (counts, 5 seeds) |
|---|---|---|---|
| thirst | **0.09 [0.04, 0.14]** | 0.24 [0.11, 0.36] | 2,460 deaths (dominant) |
| starvation | 1.42 [1.17, 1.62] | 1.13 [0.96, 1.28] | 347 |
| cold | 0.27 [0.10, 0.44] | 0.19 [0.07, 0.31] | 442 |
| predator | 0.006 | 0.010 | 289 |

## Hypotheses (preregistered)
- **H1** (beats random and greedy): **supported** for both learners. The CIs don't overlap (g from −3.0 to −5.5).
- **H2** (beats heuristic): **not supported.** The CIs overlap and both point estimates are slightly worse.
- **H3** (memory helps with water): **mixed.**
  - Thirst deaths are 2.7× lower with GRU (g = −1.15), but the CIs touch, so by our rule this is not distinguishable at n = 5.
  - Overall mortality is *not* lower: memoryless did better on all 5 worlds (all-cause g = +0.59 for GRU−FF).
  - GRU agents traded thirst deaths for starvation.
- **H4** (no social behaviour): not yet measured with the Stage 3 metrics. Nothing in the diet analysis suggests sharing.
  Learners ate meat almost never (0.04% of energy) and made 0 small-prey kills on seed 0.

## What the learners do (seed 0 recordings)
- Actions: eat 21%, rest 16%, drink 15%, moves 37%, hunt 1.1%, pickup and drop 7%. Mean hydration while alive is 0.87, against roughly 0.6 for the heuristic.
- Energy intake is about 0.00133 per agent-tick for both learner and heuristic, which roughly equals basal metabolism. Almost all of it comes from plants, at 0.03 energy per bite.
  **Energy is the binding constraint**: once water and danger are handled, agents starve slowly, especially through winter.
- Predator and fire deaths fell about 20-fold compared with the heuristic. The mechanism (active avoidance, or simply staying in open land
  near water) is **not yet established**.

## Learning curves (`fig_training.png`)
Policy entropy falls steadily from 2.40 (uniform) to about 1.1, which is the clearest sign of learning. Training-time mortality oscillates
on a ~40-iteration cycle, which matches the world switch every 40 iterations plus the seasons. **It is not a usable learning curve.**
A fixed-world evaluation every N iterations is needed (next step).

## Limitations
- One training seed per learner, so the variance between training runs is unmeasured. 20M agent-steps is small.
- The memory comparison is at an equal step budget. Recurrent nets often need longer to pay off, so H3 may be budget-limited.
- MPS (GPU) arithmetic is not guaranteed bit-identical. Seeds, sampling and the CPU path are deterministic and tested.
- Actions are sampled, not greedy. Greedy evaluation has not been run.

## Next steps (proposed)
1. **Periodic held-out evaluation during training**, and longer runs (100M+ agent-steps) with 3 training seeds per arm. This decides H2 and H3.
2. **Energy economy:** measure whether hunting or fruit could pay. If the world simply doesn't allow a surplus, report it. Don't tune
   the world for the learner.
3. Stage 3 social metrics on the learner recordings (co-feeding, proximity, transfer of food), with baseline nulls.

Reproduce: `make train NAME=gru_s0`, `make train NAME=ff_s0 MEMORYLESS=1`, then `make eval NAME=...`, then
`python experiments/plot_training.py gru_s0 ff_s0`.
