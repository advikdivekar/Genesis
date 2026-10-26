# GENESIS — Plan, milestones, acceptance criteria

GENESIS is an artificial-life testbed: minimally specified, continually learning RL agents
in a foraging ecology, rewarded **only** for homeostatic survival. Everything social
(sharing, grouping, signalling, norms, culture) must emerge or be reported as absent.

## Machine reality check (Apple M4, 10 CPU cores, 16 GB unified memory, no CUDA)

| Requirement | Feasible here? | Honest alternative |
|---|---|---|
| 256²+ world, 1,000+ agents, 100k steps/min headless | Yes for world dynamics (Numba, multi-core). Measured in Stage 1. | — |
| 1,000+ *learning* agents with independent networks | No (memory + compute) | Parameter-shared policy with per-agent recurrent state and per-agent heritable trait/hyperparameter vectors; independent learners for small populations (≤64). Independent-vs-shared is an ablation. |
| Dreamer-style world model trained online for every agent | Partially | One shared world model (MPS/CPU), per-agent latent states; per-lineage models only in small runs. |
| 50–200 seed rerun-the-tape | Only for short/small runs | 50 seeds × reduced horizon/population; full-scale runs on ≤5 seeds. Scale of every claim stated explicitly. |
| 4K video export | Yes, offline (render at 4K from replay, not live) | Live viewer targets 1080p–1440p at 60 fps; 4K via record mode. |
| Web search for citations (Stage 4) | Yes | Anything not verified is flagged `UNVERIFIED`. |

## Hard rules (enforced in code and tests)
1. Reward = homeostasis only (energy, hydration, health, warmth) + death penalty (+ optional reproduction bonus). Never a social term. `tests/test_reward_purity.py` will check reward is a function of the agent's own body state only.
2. Capabilities are switches, off by default: `memory, world_model, communication, social_learning, niche_construction, reproduction, reputation` (+ `aggression`, `huddle_warmth` documented ecology switches).
3. Every run seeded and reproducible; YAML configs; resolved config + manifest + metrics + events logged per run.
4. No scripted social behaviour. Animals have simple scripted dynamics (they are the environment); agents never do.
5. Baselines (random, greedy forager, fixed heuristic) appear in every comparison.

## Repo layout
`/sim` world engine · `/agents` baselines + learners · `/analysis` metrics · `/viewer` live client/server ·
`/configs` YAML · `/experiments` runnable scripts · `/reports` figures + write-ups · `/tests` · `/docs` ·
`/logs/AGENT_LOG.md` running log of how agents are doing · `CLAUDE.md` project memory for future sessions.

---

## Stage 1 — World engine (headless, fast)
**Build:** procedural world from seed (elevation fBm, priority-flood lakes, flow-accumulation rivers,
forests/meadows/plains/barren/rock/mountain/caves); plants with seasonal logistic regrowth; fruiting trees;
small prey, large prey (needs ≥2 simultaneous attackers — an ecological affordance, switchable), predators;
day/night, 4 seasons, weather Markov chain (rain, fog, drought, cold snap, snow), disturbances (fire, flood,
famine) both random and triggerable; agents with egocentric occluded perception and survival physiology;
counter-based hash RNG so results are identical regardless of thread count; batched API + PettingZoo
`ParallelEnv`; run logger; three baselines.

**Accept when:**
- [x] Same seed ⇒ bit-identical world and trajectory hash; different seeds differ (tests).
- [x] World rules unit-tested: blocking terrain, drinking only near water, eating raises energy, seasonal plant growth, occlusion, large-prey rule, fire spread/extinguish, death.
- [x] `pettingzoo.test.parallel_api_test` passes.
- [x] Benchmark: ≥1,000 agents on 256², ≥100k env steps/min for world dynamics (124k); with-observation throughput reported honestly (72k, below target).
- [~] Heuristic survives far longer than random (0.87 vs 3.07 deaths/agent-yr). Greedy does NOT beat random (thirst; water out of view, no memory) — reported as a negative result.
- [x] Evidence: terrain maps for several seeds, seasonal biomass, climate/weather timeline, fauna dynamics, baseline survival curves (`reports/stage1/`).

## Stage 2 — Agent minds
**Build:** recurrent actor-critic (PPO, GRU core) in PyTorch (MPS when faster, else CPU); shared vs independent
parameters; memory ablations (none / GRU / external episodic memory); online Dreamer-lite world model (RSSM-style
latent dynamics, reward + obs decoder) with imagination rollouts; continual online learning with replay; EWC and
plasticity-injection ablations; lifetimes + heritable trait vectors + mutation (Baldwin experiments); social switches
(model-of-others probe head, reputation obs, imitation channel from elders, costly limited-bandwidth symbols).

**Accept when:**
- [~] PPO-GRU agents beat all three baselines on survival with CIs over ≥5 seeds: beats random and greedy; **not yet
      distinguishable from the heuristic** at 20M agent-steps (`reports/stage2/STAGE2_REPORT.md`).
- [ ] World-model prediction error decreases online; imagined rollouts logged.
- [ ] Forgetting/plasticity metrics computed on a fixed probe task set across seasons.
- [x] Integration test: a short training loop runs end-to-end deterministically; checkpoint round-trip (`tests/test_ppo.py`).

## Stage 3 — Emergence measurement
**Build:** `/analysis` suite for individual, social, communication, culture, norms, resilience metrics
(list in task spec), rerun-the-tape driver, bootstrap CIs + effect sizes, and `docs/PREREGISTRATION.md`
written and timestamped **before** Stage 7 runs.

**Accept when:**
- [ ] Every metric has a unit test on synthetic data with known ground truth (e.g. planted clustering, planted signal–context MI, planted sharing events).
- [ ] Metrics on baselines produce the expected nulls (e.g. random agents: MI≈0, no ratchet).

## Stage 4 — Human-comparison module
**Build:** `analysis/human_timeline.yaml` with citations (anthropology, archaeology, developmental psychology,
Hockett's design features); comparator that aligns each run's emergence order + triggering conditions with it.

**Accept when:**
- [ ] Each entry has a verified citation or is flagged `UNVERIFIED`.
- [ ] Report language frames comparisons as structural analogies only.

## Stage 5 — Live viewer
**Build:** Python WebSocket server streaming delta-encoded state at 30–60 Hz; PixiJS v8 client (**changed 2026-10-26
at the user's request: three.js, perspective "diorama" camera — see `docs/VIEWER.md`**): autotiled
painterly-pixel terrain, animated water, wind-swayed vegetation, day/night lighting + bloom, weather + seasons,
procedural animated agent sprites keyed to genes, signal rings, social links, interpolation, overlays, inspector,
dashboard sparklines, time controls + replay scrub, experiment controls, auto-director, record mode.

**Accept when:**
- [~] 60 fps at 1440p with 1,000 agents: measured 75–90 fps at 1600×900 (pixel ratio 1) with 1,000 agents on an
      M3 Pro (headless Chromium/Metal). 1440p not yet measured in a real browser.
- [x] Every control is logged (`runs/viewer/*/controls.jsonl`; disturbances also via `manual_log` + event stream).
- [x] Screenshots: day, night, rain, snow/winter, fire, flood, inspector, overlays (`reports/viewer/`).
- [ ] Still open: replay scrub + bookmarks, side-by-side seeds, mid-run resource changes, spawn/remove agents,
      audio, 4K offline export, social/signal overlays and imagined futures (need learners).

## Stage 6 — Engineering quality
- [ ] One-command setup (`make setup`) and run (`make viewer`, `make bench`, `make test`).
- [ ] Local experiment tracking (MLflow-compatible file store or JSONL), checkpoints, resumable runs.
- [x] Smoke test: headless sim + viewer server boot + one frame received (`tests/test_viewer.py`).
- [ ] Profiling report: steps/s, FPS, memory, bottleneck fixes.

## Stage 7 — First experiments
A. baseline vs memory vs world-model agents × patchy vs uniform resources
B. social learning on/off across generations (ratchet?)
C. communication on/off (displacement? dialect drift?)
D. famine/flood resilience across A–C
E. 50+ seed rerun-the-tape (reduced scale, stated)

**Deliver:** `reports/FIRST_RESULTS.md` with figures, CIs, negative results, limitations, human-comparison section,
how to launch the viewer, three most promising next experiments.

---

## Status
| Stage | Status |
|---|---|
| 1 World engine | done, plus Stage 1.1 tuning (food-limited fauna, ≥100k steps/min with obs at 1,000 agents) — awaiting review ([report](reports/stage1/STAGE1_REPORT.md)) |
| 5 Live viewer | first slice done (three.js, 2026-10-26): live world, dashboard, inspector, controls, logging — see `docs/VIEWER.md` |
| 2 Agent minds | first slice done: recurrent PPO + memoryless ablation, held-out evaluation (`reports/stage2/`) |
| 3–4, 6–7 | not started |
