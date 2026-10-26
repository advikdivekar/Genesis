# GENESIS session log

Running record of development sessions: what was done, what was found, what was decided. Newest last.
(`logs/AGENT_LOG.md` is for how agents perform; this file is for engineering work.)

## 2026-10-26 — Session 3: live three.js viewer, survival-analysis bug fix

**Machine.** This session ran on a MacBook Pro with an M3 Pro (11 cores: 5 performance and 6 efficiency, 18 GB), not the
M4 Air the earlier benchmarks were measured on. Throughput numbers from different machines must not be compared.

**Setup.** This was a fresh clone with no `.venv`. `make setup` worked, and all 41 tests passed before any change.

**Answered for the user:** the default run has **256 agents** (2,048 slots, fixed population with respawn), plus
400 small prey, 48 large prey and 12 predators at the start, which are food-limited after that. No learning agents exist yet. Everything
on screen is a non-learning baseline (heuristic by default).

**Built: live viewer (Stage 5, first slice).** The user asked for three.js instead of the PixiJS named in PLAN.md.
- `viewer/server.py`: aiohttp WebSocket server, sim thread, binary protocol, command queue applied between ticks,
  and per-session run logging (`controls.jsonl`, chunked events, metrics, summary). An agent-log entry is added
  at session close.
- `viewer/static`: ES modules with no build step, and three 0.170.0 served from `viewer/node_modules`. It has heightmap terrain with
  plant, snow, fire, flood, drought and season shading; animated water whose surfaces rise with floods and vanish
  when streams dry; instanced wind-swayed trees, grass and flowers; agents with walk, rest, eat, drink and hunt animation,
  gene-coloured; three fauna species; predator eyes; need icons; day/night with moon and stars; rain, snow,
  fireflies and leaves; fire with flames, smoke and light; event bursts; bloom, tilt-shift and grading; a dashboard with
  7 live charts and deaths by cause; an inspector, follow mode, trail and perception ring; a minimap; a director;
  disturbance tools; restart; screenshots and recording.
- `tests/test_viewer.py`: smoke test covering server boot, init/fields/frame received, and a command applied and
  logged, with the run files written on close.
- `make viewer-setup`, `make viewer`; docs in `docs/VIEWER.md`.

**Bugs found and fixed during the build:**
- Port 8765 is used by another local service (`nap_whisper`), so the viewer defaults to 8787.
- Client registry held unhashable tuples, which made WebSocket handshakes fail. It's now a dict.
- Test sessions appended to `AGENT_LOG.md`. Those entries were reverted, `--no-agent-log` was added, and session years
  now exclude the spin-up.
- The event feed was invisible at high death rates, because items were replaced faster than their fade-in. Deaths are
  now batched every 1.5 s.

**Bug found in existing analysis (important):** `analysis/survival.py:lifetimes()` started the initial cohort's
lifetimes at tick 0. Since Stage 1.1, `env.tick` already includes the 1-year ecological spin-up, so every founding agent
was credited with an extra year. On seed 0 with the random policy over 1 year this gives 1.60 deaths/agent-yr against 3.20 when
fixed, and KM S(1 yr) 0.340 against 0.003. The Stage 1.1 baseline table (2.55 / 2.21 / 1.00) was affected. The fix adds
`start_tick=env.spinup_ticks` plus a regression test, and the baselines were re-run (see AGENT_LOG).

**Experiment (same session):** density sweep `experiments/density_sweep.py` (hypotheses in the docstring, written
before running). The heuristic dies 4.94 per agent-year at 1,000 agents against 1.6–1.7 at 256–512, almost all of it thirst, and does worse than
random at 1,000. The water-congestion mechanism is untested. Details are in AGENT_LOG.

**Evidence:** `reports/viewer/*.jpg` (day, overview, inspector, flood, wildfire, night, autumn, density overlay,
winter), captured with headless Chromium on Metal through scratch Playwright scripts (not in the repo). Render rate was 85–115 fps
with 256 agents and 75–90 fps with 1,000 agents at 1600×900.

**State at end:** 43 tests pass. Nothing is committed, and the user has not asked for a commit. The viewer server was left running on
http://localhost:8787 (seed 0, 256 heuristic agents, AGENT_LOG summary on close). Viewer test run directories from this session
remain in `runs/viewer/`; a safety check blocked a broad `rm`, so they were left for the user.

### Session 3, continued: flight recorder, teaching guide (2026-10-26, ~20:45–21:05)
- Committed the viewer work as `41c19c2` (not pushed).
- **Reproduction check:** with `capabilities.reproduction` on, heuristic populations crash 60–85% every winter and random agents go extinct (see AGENT_LOG).
- **User request: "record every agent's every activity at the atomic level."** Built `sim/recorder.py` and `analysis/trajectories.py`.
  - Per tick it stores every agent's full state, action, reward and death cause, plus the policy's internal mode and all lifetime counters.
  - It also stores world state per tick, every animal per tick, every event, every birth with genes, and every manual intervention.
  - Optionally it stores every observation.
  - Exact replay re-simulates and checks every tick, and can rebuild any agent's observation at any tick. A full sim year replays in 2.5 s.
  - **Cost:** the first version cut speed to 35% (structured rows assembled per tick, map reductions, slow zlib). After optimising it
    runs at 70%: 1,250 vs 1,780 ticks/s, 63 MB per sim year at 256 agents.
  - It is on by default in the viewer and the experiment scripts.
  - Engine change: manual floods now log `radius` and `days_given`, so floods whose duration was drawn at random replay exactly.
  - Tests: `tests/test_recorder.py`, and `test_viewer.py` now replays the viewer session. 45 tests pass.
- **Viewer shutdown bug:** SIGTERM left the server waiting up to 60 s for open WebSockets. The fix closes them in `on_shutdown` and sets
  `shutdown_timeout=3`. The 20:38 and 20:58 sessions both still closed cleanly, with summaries and AGENT_LOG entries.
- Wrote `TEACHER.md`: RL concepts explained through GENESIS, a glossary, and an FAQ built from the user's questions.

### Session 3, continued: PR #1, Stage 2 first slice (2026-10-26, ~21:05–23:10)
- **PR #1** opened on HeetRanpura/Genesis (branch `session3-viewer-recorder` on the fork advikdivekar/Genesis-advik).
  Author: Advik Divekar only, no co-authors.
- **Stage 2 learner** on branch `stage2-ppo`:
  - `agents/ppo.py` (recurrent PPO, shared network, per-agent GRU state reset at birth)
  - `experiments/train_ppo.py` (training worlds 1000+, full flight recording, disk guard)
  - `experiments/eval_policy.py` (same protocol as the baselines)
  - `agents/learned.py`, and viewer `--checkpoint` (the inspector shows the learner's value estimate and action probabilities)
  - `docs/PREREGISTRATION.md`, committed before any evaluation
- **Bugs caught:**
  - minibatch shuffle used numpy's unseeded global RNG. Fixed, and the run was restarted (the discarded run was moved to the scratchpad).
  - a test tolerance was too tight (9e-8 float rounding).
  - the queue loop's `pgrep -f` matched its own command line. It worked anyway, but my "fix" then killed the parent shell. The trainer survived and a watcher was added.
- **Performance:** about 4–7k agent-steps/s on MPS. The update dominates, especially the backward pass. CPU was 4× slower.
- **Results:** see AGENT_LOG and `reports/stage2/STAGE2_REPORT.md`. H1 supported, H2 not, H3 mixed.
- **Disk:** about 6 GB of recordings for Stage 2. 29 GB free.
