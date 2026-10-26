# Preregistered hypotheses

Hypotheses are written here **before** the runs that test them, with date and time. Results go to `logs/AGENT_LOG.md`
and reports; entries here are never edited after the fact (corrections are appended).

## Stage 2, first slice: recurrent PPO vs baselines (written 2026-10-26 21:20 IST, before any evaluation)

**Setup.** Parameter-shared actor-critic, survival reward only. 256 agents, fixed population, all capabilities off.
It trains on worlds 1000+ for 600 iterations × 128 ticks (about 19.7M agent-steps, about 50 min on the M3 Pro).
Evaluation uses held-out seeds 0–4 × 2 years, exactly like `run_baselines.py`. Two learners: `gru_s0` (GRU memory) and
`ff_s0` (memoryless ablation, otherwise identical).

- **H1 (learning works).** The GRU learner's deaths/agent-yr on held-out worlds is below random (3.83) and greedy (3.31).
- **H2 (beats the hand-written rules).** The GRU learner is below the heuristic (1.50). *My expectation: uncertain at this
  training budget.* 20M agent-steps is small for a 1.1M-parameter recurrent policy. A failure here is a budget result,
  not a verdict on learning.
- **H3 (memory helps with water).** GRU vs memoryless: GRU has fewer thirst deaths per agent-year, and its overall mortality is lower
  or equal. This is the prediction logged in AGENT_LOG at Stage 1 (greedy dies of thirst because water leaves its view).
- **H4 (null for social behaviour).** No measurable social behaviour is expected at this stage. "Ate others' kill" will be
  co-feeding at the level the environment produces, and there's no reason for agents to drop food near others.
  Any apparent grouping is expected to be explained by shared water and food sources.

**Decision rule.** "Beats" means the 95% bootstrap CIs over the 5 held-out seeds don't overlap, plus a report of Hedges' g.
Overlapping CIs are reported as "not distinguishable at n = 5", never as a win.

**Known limitations, stated in advance:** a single training seed per learner (seed-to-seed variance of training is unmeasured);
GPU (MPS) arithmetic isn't guaranteed bit-identical across runs, although seeds, sampling and the CPU path are deterministic and tested.
