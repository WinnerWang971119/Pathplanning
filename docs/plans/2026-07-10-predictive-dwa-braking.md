# Predictive DWA v2 — Braking-Inevitability + Cost-to-Go Guidance

**Goal:** Rebuild the predictive DWA family so it faithfully matches Missura & Bennewitz
(ICRA 2019) — a soft emergency-braking inevitable-collision test instead of a blanket
space-time hard reject — and adds a static cost-to-go guidance field, so a predictive DWA
actually beats plain DWA on the arena's dense crossing traffic instead of being
net-harmful (measured baseline: plain `dwa` 0.50, `dwa_predictive_oracle` h10 0.80,
`dwa_predictive` h10 0.70 on the 10-seed quick read — prediction is currently net-harmful
even with a perfect-velocity oracle).

**Approach:** Keep vanilla DWA's present-position lidar floor UNCHANGED (full live cloud —
walls and movers at their current returns — so a tracker miss is always caught). Replace
the freezing hard space-time reject with a soft predictive layer: an imminent backstop, an
emergency-braking inevitable-collision test (reject only when even braking-to-a-stop still
collides at matched time), and an un-clipped monotone predicted-clearance score term that
mechanizes yielding (the collision-free slower candidate wins the argmax). Add a static
cost-to-go field (one Dijkstra from the goal at reset) as the heading term to cure the
local-minima timeouts. Ship as a 2×2 ablation — {paper-only, paper+global} × {oracle,
lidar}, oracle-first. The paper+global lidar variant becomes the canonical `dwa_predictive`;
the `LidarTracker` gains OPT-IN hardening (smoothing + speed clamp) used only by the DWA
lidar keys, so the other canonical planner that shares the tracker (`d_star_lite_predictive`)
is byte-unchanged.

## Scope

- **In scope:**
  - Byte-preserving refactor of `DWAController` to expose ONE seam: `_heading_term`
    (the goal-heading score block), threading `state` through `_score`.
  - A pure `build_cost_to_go_field(grid, goal_cell)` Dijkstra-from-goal field builder.
  - Predictive DWA `_evaluate_candidate` rewrite: imminent backstop + emergency-braking
    inevitable-collision test + un-clipped monotone predicted-clearance term. A
    `_braking_trajectory` helper (decelerate to 0, then hold).
  - A global-guidance mixin: build the cost-to-go field at reset, override `_heading_term`
    to score by normalized field-value decrease; fall back to base heading (no
    `planner_error`) if the start cell is unreachable.
  - Four registry keys: `dwa_predictive` / `dwa_predictive_oracle` REPLACED in place with
    the paper+global behavior; new `dwa_predictive_paper` / `dwa_predictive_paper_oracle`
    for the paper-only ablation. `EXPERIMENTAL_KEYS` grows to 4; canonical set stays 13.
  - OPT-IN LidarTracker hardening: windowed-mean velocity smoothing (≤3 associated frames)
    + clamp to the max regime obstacle speed, off by default so `d_star_lite_predictive`
    is byte-unchanged, on only for the two DWA lidar keys.
  - New/updated TCs in `arena/arena.py --check`.
  - A 10-seed × {0,5,10,20} quick read (all variants + plain dwa), a findings doc + PNG,
    and evaluation of the tightened 2×2 success gates. **Run by Claude** (scratch script,
    not committed; the findings doc + PNG are the committed deliverable).
  - Docs: CLAUDE.md (both the DWA-predictive AND the D* Lite predictive sections — the
    latter to record that the shared tracker's default is unchanged), Mission.md Phase 7,
    README (two new `--algorithm` values).
- **Out of scope:**
  - The full 50-seed canonical / horizon / speed sweeps — **the user launches these**;
    this spec sets them up and runs only the 10-seed quick read.
  - A velocity-obstacle (VO) steering layer, SIPP, or any new non-DWA planner family
    (considered by the consult, deliberately deferred).
  - Global-path *replanning* / any `--replan-k` on the DWA-predict family (field is
    plan-once at reset).
  - `trajectory_conflict` / `predict_blocked_cells` signature changes — reused as-is.
  - Adding the paper-only keys to the horizon-sweep `SWEEP_ALGORITHMS` (driver/plotter) —
    they are an ablation reached only through the runner + the T8 quick-read script;
    `dwa_predictive` / `dwa_predictive_oracle` remain in the sweep as today.

## Decisions

- **Braking = emergency-braking inevitable-collision-state test, NOT a scalar
  stopping-distance formula.** The first-draft `d_c = v·ttc·dt; v_adm = sqrt(2·b·d_c)`
  was found in review to algebraically collapse to a degenerate TTC cutoff (`v > 0.4·k`)
  that never fires beyond 0.2 s — deleted. The correct rule simulates a braking trajectory
  through `trajectory_conflict` and rejects only when the collision is inevitable even if
  the robot brakes now (the ICS test), which structurally cannot freeze.
- **Single-count by softening the FUTURE layer, not by thinning the present floor.** The
  present-position floor keeps the FULL live lidar cloud (walls + movers at current
  returns), so a tracker miss is still caught. The old double-count (present hard-reject +
  every-future hard-reject) is cured by making the future layer soft (braking-admissibility
  + monotone score), NOT by subtracting mover returns. No blindness risk; no
  `_present_floor_points` subtraction.
- **Guidance = cost-to-go field, not a WaypointFollower** — both approach consultants
  independently made this their top pick; a scalar geodesic field cannot wedge under
  constant lateral displacement by ~20 crossers (the pathology this repo already fixed for
  the follower, `project-replanning-commitment-horizon`).
- **Replace canonical in place; paper-only is a new experimental pair** — the main-scatter
  `dwa_predictive` dot becomes the strongest (paper+global) variant.
- **Tightened 2×2 success bar, judged on the ORACLE rows** — prediction must beat its own
  guided baseline; lidar rows are reported but a lidar miss is a perception finding, not a
  spec blocker (oracle-first isolates policy from perception). See AC12.
- **LidarTracker hardening is opt-in (default off)** so the shared tracker leaves
  `d_star_lite_predictive` byte-identical (TC63/TC64 stay binding). `MAX_TRACK_SPEED = 2.0`
  (the `fast` regime cap, NOT 1.5) so the issue-#11 speed sweep is not truncated.
- **`BRAKE_DECEL = MAX_LINEAR_ACCEL` (2.0)** — the braking sim uses the window's own physics.

## Acceptance Criteria

Correctness / determinism (verified by TCs):
- [ ] **AC1** Plain `dwa` trace JSONL byte-identical to pre-change on the same seed
  (`--no-traffic` and traffic-on) — the `_heading_term` extraction is byte-preserving.
- [ ] **AC2** `dwa_predictive_paper` and `dwa_predictive_paper_oracle` at `h0` produce a
  byte-identical trace to plain `dwa` (`--no-traffic` and traffic-on).
- [ ] **AC3** `dwa_predictive` and `dwa_predictive_oracle` (paper+global) at `h0` are
  deterministic (two same-seed runs byte-identical) and are NOT byte-identical to plain
  `dwa` (field guidance active) — the global-only ablation cell.
- [ ] **AC4** All four keys: two same-seed traffic-on runs byte-identical; every trace line
  carries the 8-key schema.
- [ ] **AC5** `--predict-horizon` required for all four (omission → exit 2); `--replan-k`
  rejected for all four (exit 2); `algorithm_label` folds `_h<steps>`.
- [ ] **AC6** The present-position floor is UNCHANGED — it checks the full live lidar cloud;
  no live return is ever subtracted for a tracked mover, so an obstacle with no track is
  still rejected by the floor at its current position (verified on a synthetic scan).
- [ ] **AC7** Predictive layer: an inevitable matched-time collision (the braking+held
  trajectory still collides) is rejected; a brakeable conflict is admitted; a `ttc_step==1`
  imminent conflict is rejected; the predicted-clearance score term is strictly monotone in
  `min_gap` and NOT clipped at 0 (a slower collision-free candidate outscores a faster
  colliding one). The extra braking rollout is RNG-free and fixed-step (determinism).
- [ ] **AC8** `build_cost_to_go_field` returns octile goal-distances (CELL units) matching
  an A*-cost oracle on reachable cells and `inf` on occupied/unreachable cells, and is
  deterministic; the global `_heading_term` is non-saturated and monotone in geodesic
  progress; if the start cell is unreachable the global variant falls back to base heading
  with `planner_error` null.
- [ ] **AC9** Opt-in LidarTracker hardening: DEFAULT construction is byte-identical to
  today (so `d_star_lite_predictive` is unchanged, TC63/TC64 pass); when enabled it is
  deterministic across a cluster-count change AND an association swap, and never reports a
  speed above `MAX_TRACK_SPEED`.
- [ ] **AC10** `run_all` canonical set == 13; `EXPERIMENTAL_KEYS == {d_star_lite_oracle,
  dwa_predictive_oracle, dwa_predictive_paper, dwa_predictive_paper_oracle}`; the import
  assertion holds; all four DWA-predict keys are in `PREDICT_FAMILIES`.
- [ ] **AC11** `act()` never raises mid-episode for any key (the new field lookup, braking
  rollout, and score paths are non-raising / guarded); the DWA family's `planner_error` is
  always null.

Empirical (produced + evaluated by the quick read, T8 — documented, not silently passed):
- [ ] **AC12** The 10-seed × {0,5,10,20} quick read runs for plain `dwa` + all four
  predictive keys; results, the 2×2 table, and a PNG land in a findings doc. **PASS**,
  judged on the ORACLE rows = **gate 1** (`dwa_predictive_paper_oracle` best-horizon
  failure_rate < the measured plain-`dwa` rate on the same 10 seeds) **AND gate 2**
  (`dwa_predictive_oracle` best-horizon failure_rate < the `dwa_predictive_oracle` h0 =
  global-only cell). The lidar rows are reported alongside; a lidar-only miss is recorded
  as a perception finding, not a spec failure. A gate miss is documented with next-step
  levers, never reported as success.

## Contracts & Interfaces

### New / changed signatures

- `DWAController._score(self, state, trajectory, v, clearance) -> float` — **owner T1**;
  gains `state` (threaded from `_evaluate_candidate`, which already has it) so it can pass
  it to `_heading_term`. Byte-preserving: base ignores `state`.
- `DWAController._heading_term(self, state, trajectory) -> float` — **owner T1**; base body
  is the exact goal-heading block lifted verbatim from `_score` (returns the `[0,1]` heading
  alignment, `1.0` inside `GOAL_REACHED_RADIUS`; ignores `state`). Overridden by the
  global-guidance mixin (**consumer T4**).
- `build_cost_to_go_field(grid: OccupancyGrid, goal_cell: tuple[int, int]) -> np.ndarray`
  — **owner T2**, new module `planners/_costfield.py`. Dijkstra from `goal_cell` over
  `grid.cells`, SAME cost model as `astar_search` (8-connected, `np.hypot(dr,dc)` octile
  step cost in CELL units, no corner-cutting: a diagonal is blocked if either orthogonal
  neighbor is occupied). Returns a `(rows, cols)` float64 array of goal-distances in CELL
  units; occupied and unreachable cells are `inf`. Deterministic `(dist, (row, col))` heap
  so ties break on cell index (exactly like `astar_search`). Pure — imports only
  `manual_astar.OccupancyGrid`, numpy, heapq; no irsim. **Consumer T4.**
- `PredictiveDWAController._braking_trajectory(self, state, v, w) -> np.ndarray` — **owner
  T3**; forward-simulate same `w`, `v_k = max(0, v - BRAKE_DECEL*k*dt)`; once `v` hits 0,
  HOLD position for the remaining steps. Returns `(self._rollout_steps, 2)`. RNG-free,
  fixed step count.
- `LidarTracker.__init__(self, grid, bearings, range_max=inf, *, smoothing_frames=0,
  max_track_speed=inf)` — **owner T6**; the two new keyword-only args default to OFF, so
  every existing construction (incl. `d_star_lite_predictive`) is byte-identical. When
  `smoothing_frames>0`, reported `vx/vy` = windowed mean over the last ≤`smoothing_frames`
  associated instantaneous velocities (history rides the existing positional association
  chain); when `max_track_speed<inf`, the velocity magnitude is clamped to it. `update()`
  return type unchanged (`list[Track]`). **Consumers:** the two DWA lidar keys.

### Reused as-is (no signature change)

- `trajectory_conflict(robot_positions, tracks, robot_radius, horizon_steps, dt, margin)
  -> TrajectoryConflict(collides, ttc_step, min_gap)` — the predictive layer calls it once
  on the constant-`(v,w)` rollout (imminent + soft term) and once on the braking trajectory
  (inevitability). `margin = COLLISION_MARGIN` (0.05). No change.
- `OracleTracker`, `Tracker`, `Track`, truth seam (`wants_truth`/`observe_truth`),
  `world_to_grid`, `build_occupancy_grid`, `load_world`, the base `_rollout` /
  `_trajectory_clearance` / present-position floor.

### Registry / family sets (`planners/_grid.py`, **owner T5**)

- `ALGORITHMS` gains `dwa_predictive_paper`, `dwa_predictive_paper_oracle`.
- `PREDICT_FAMILIES` gains the same two (six DWA/D*-predict keys total).
- `EXPERIMENTAL_KEYS = {d_star_lite_oracle, dwa_predictive_oracle, dwa_predictive_paper,
  dwa_predictive_paper_oracle}` (2 → 4; `dwa_predictive` stays canonical).

### New module-level constants (`planners/dwa_predictive.py`, owners T3/T4)

- `BRAKE_DECEL = MAX_LINEAR_ACCEL` (2.0).
- `PREDICTED_GAP_WEIGHT = 0.3` — weight on `clip(min_gap, -CLEARANCE_CAP, CLEARANCE_CAP)/CLEARANCE_CAP`
  (symmetric clip, so negative gaps penalize; NOT the old `clip(·,0,cap)`).
- `MAX_PROGRESS_CELLS = MAX_LINEAR_SPEED * ROLLOUT_STEPS * CONTROL_DT / GRID_RESOLUTION`
  (= 12.0) — the field-progress normalizer, in CELL units to match the field.

### New constants (`planners/_predict.py`, owner T6)

- `MAX_TRACK_SPEED = 2.0` (m/s) — the `fast` regime cap.
- `VELOCITY_SMOOTHING_FRAMES = 3`.

### File ownership

| File | Owner | Consumers |
|------|-------|-----------|
| `planners/dwa.py` | T1 | T3, T4 |
| `planners/_costfield.py` (new) | T2 | T4 |
| `planners/dwa_predictive.py` | T3 → T4 → T5 (chain) | T7, T8 |
| `planners/_grid.py` | T5 | T7 |
| `planners/_predict.py` | T6 | T7, T8 |
| `arena/arena.py` | T7 | — |
| findings doc `...braking.findings.md` (+ `.png`), scratch quick-read script | T8 | — |
| `CLAUDE.md`, `Mission.md`, `README.md` | T9 | — |

## Algorithm detail — the predictive layer (Notes for T3, from the judge's SYNTHESIS)

Per sampled candidate `(v, w)` with constant-`(v,w)` rollout `traj`:
1. **Present-position floor (UNCHANGED):** reject if `_trajectory_clearance(traj[:ROLLOUT_STEPS],
   full_live_cloud)` is `None`. The cloud is the full lidar projection — movers at current
   returns are NOT subtracted. Catches walls and any tracker miss.
2. **Base score:** `_score(state, traj, v, clearance)` using `_heading_term` (Euclidean for
   paper-only, field for global) + clearance + speed.
3. **No tracks** (h0 or empty): return base (paper-only h0 ⇒ plain dwa).
4. **Predicted conflict:** `c = trajectory_conflict(traj, tracks, r, H, dt, COLLISION_MARGIN)`.
   - **Imminent backstop:** `if c.ttc_step == 1: reject`.
   - **Braking inevitability:** `b = _braking_trajectory(state, v, w)`;
     `cb = trajectory_conflict(b, tracks, r, H, dt, COLLISION_MARGIN)`;
     `if cb.collides: reject` (even braking-to-a-stop-and-hold still collides at matched
     time → inevitable).
   - **Soft term (always):** `base + PREDICTED_GAP_WEIGHT * clip(c.min_gap, -CLEARANCE_CAP,
     CLEARANCE_CAP)/CLEARANCE_CAP` — strictly monotone, un-clipped at 0, so a
     collision-free slower/off-heading candidate outscores a grazing faster one. This is
     what mechanizes yielding.
   Determinism: both rollouts share fixed `H`, `dt`, constant obstacle advance, no RNG;
   `_braking_trajectory` uses `max(0, ·)` on `v`. Argmax keeps the existing strict-`>`
   fixed-iteration tie-break.

Global `_heading_term(state, traj)` (T4 override):
`start = world_to_grid(state[:2])`, `end = world_to_grid(traj[-1])`;
`if field[start]==inf: fall back to base heading` (should not happen post-reset guard);
`if field[end]==inf: return 0.0` (rollout ends in a wall — disfavor);
`progress = field[start] - field[end]` (cells);
`return 0.5 + 0.5*clip(progress / MAX_PROGRESS_CELLS, -1.0, 1.0)` (interior, monotone,
retreat<no-progress<progress); keep the `GOAL_REACHED_RADIUS` → `1.0` guard.

## Error Handling

- **Start cell unreachable at reset** (goal walled off from start): the global variant
  disables guidance for the episode (base Euclidean heading), `planner_error` null — DWA
  never fails to plan (AC8, AC11).
- **Tracker / prediction / field lookup raises inside `act()`:** the tracker update keeps
  its existing guard; the new per-candidate paths are non-raising by construction
  (`world_to_grid` clips to bounds; braking rollout is pure math), and the candidate loop
  is additionally wrapped so any unexpected raise degrades that tick to base-DWA scoring —
  never propagates (AC11).
- **Empty window / all rejected:** the existing in-place-rotation fallback fires (far rarer
  now — the soft layer keeps admissible candidates).
- **Reused instance across episodes:** `reset()` clears field, tracker (`_tracker=None`),
  tracks, snapshot; the tracker's smoothing history dies with the rebuilt tracker.

## Testing Strategy

**Levels:** Unit (pure helpers, in-process) + Integration (subprocess runner). Added to
`python arena/arena.py arena/arena_v1.yaml --check`. **Run:**
`& .venv\Scripts\python.exe arena/arena.py arena/arena_v1.yaml --check`.

| ID | Test Case | Type | Expected |
|----|-----------|------|----------|
| TC65* | plain `dwa` unchanged; paper-only h0 == plain dwa (`--no-traffic` + traffic). **Also REPLACE the existing TC65 assertions that `dwa_predictive_h0`/`_oracle_h0` == plain dwa — those become `!=` (moved to TCa).** | Integration | byte-identical (AC1, AC2) |
| TCa | paper+global h0 deterministic AND != plain dwa | Integration | equal pair; differs from plain dwa (AC3) |
| TCb | all four keys traffic-on e2e + determinism + 8-key schema | Integration | terminal, byte-identical pair, 8 keys/line (AC4) |
| TCc | `--predict-horizon` required / `--replan-k` rejected for the 4 keys; `_h<steps>` label | Integration | exit 2 on violation; no `<seed>.json` (AC5) |
| TCd | `build_cost_to_go_field` == A* cost (cells) on reachable; inf on sealed; deterministic | Unit | equal; inf; byte-identical (AC8) |
| TCe | braking-inevitability + soft term: inevitable (braking+held still collides) rejected; brakeable admitted; `ttc==1` rejected; monotone un-clipped soft term makes a slower collision-free candidate outscore a faster colliding one; behavioral head-on crosser ⇒ chosen v decreases tick-over-tick | Unit + Integration | per §Algorithm (AC7) |
| TCf | present floor keeps un-tracked mover returns (a mover with no track still rejected by the floor); no live return subtracted | Unit | rejection preserved (AC6) |
| TCg | field `_heading_term` non-saturated + monotone: two candidates at different geodesic progress → strictly ordered interior scores; retreat < no-progress < progress | Unit | ordered, interior (AC8) |
| TCh | opt-in tracker: DEFAULT construction byte-identical (assert `d_star_lite_predictive` TC63/TC64 unaffected); ENABLED deterministic across cluster-count change AND association swap; speed ≤ `MAX_TRACK_SPEED` | Unit | as stated (AC9) |
| TCi | start-unreachable fallback on `arena/arena_no_path.yaml` (sealed start → `field[start]=inf`) → global variant falls back, `planner_error` null, terminal = timeout (not crash) | Integration | null error, timeout (AC8, AC11) |
| TCj | mid-episode raise guard: make the tracker/prediction raise after the first successful act → `act()` returns a finite `(2,1)` action | Unit (in-process) | no raise (AC11) |
| TCk | `run_all` canonical == 13; `EXPERIMENTAL_KEYS` == the 4; 4 keys in `PREDICT_FAMILIES`; assertion holds | Unit | as stated (AC10) |

**Test data:** synthetic grids/tracks in-process for the unit TCs; seeded subprocess runner
for integration; `arena/arena_no_path.yaml` for TCi.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Base DWA `_heading_term` seam (byte-preserving) | — | med | `planners/dwa.py` | Thread `state` into `_score`; extract the goal-heading block of `_score` into `_heading_term(state, trajectory)->float` (base returns the exact current value, ignores `state`). `_score` calls `HEADING_WEIGHT*self._heading_term(state, traj)`. Keep plain `dwa` byte-identical. No `_present_floor_points` (the floor is unchanged). Satisfies AC1. |
| T2 | Cost-to-go field builder | — | med | `planners/_costfield.py` (new) | Pure `build_cost_to_go_field(grid, goal_cell)->np.ndarray`: Dijkstra from goal, octile `np.hypot` step cost in CELL units, no corner-cutting, `(dist,(r,c))` heap for deterministic ties; occupied/unreachable = inf. Imports only `manual_astar.OccupancyGrid`, numpy, heapq. Satisfies AC8 (field half). |
| T3 | Predictive base: braking-inevitability + soft term | T1 | high | `planners/dwa_predictive.py` | Rewrite `PredictiveDWAController._evaluate_candidate` per §Algorithm: present floor unchanged; imminent `ttc==1` reject; `_braking_trajectory` (decel to 0 then HOLD) + `trajectory_conflict` inevitability reject; un-clipped symmetric monotone `min_gap` soft term. Add `BRAKE_DECEL`, `PREDICTED_GAP_WEIGHT`. Wrap the candidate loop so an unexpected raise degrades to base scoring. Delete the degenerate `d_c`/`sqrt` formula and any `_present_floor_points` idea. Satisfies AC6, AC7, AC11. |
| T4 | Global-guidance mixin + field heading | T1, T2, T3 | high | `planners/dwa_predictive.py` | Add `use_global_guidance` (class attr). When set: `reset()` builds the field (goal via `world_to_grid`); if `field[start]==inf` disable guidance for the episode (base heading, null error). Override `_heading_term` to `0.5+0.5*clip(progress/MAX_PROGRESS_CELLS,-1,1)` with the inf-end→0 and `GOAL_REACHED_RADIUS`→1 guards. Add `MAX_PROGRESS_CELLS`. Ensure paper-only h0==plain dwa, global h0=field-only. Satisfies AC3, AC8, AC11. |
| T5 | Register 4 keys + family sets | T3, T4 | med | `planners/dwa_predictive.py`, `planners/_grid.py` | Concrete classes: `DWAPredictiveController`(dwa_predictive, lidar+hardened, global), `DWAPredictiveOracleController`(dwa_predictive_oracle, oracle, global) — replace behavior in place; `DWAPredictivePaperController`(dwa_predictive_paper, lidar+hardened, no global), `DWAPredictivePaperOracleController`(dwa_predictive_paper_oracle, oracle, no global). `register()` the two new keys; add both to `PREDICT_FAMILIES` and `EXPERIMENTAL_KEYS`. Lidar keys build the tracker with `smoothing_frames=VELOCITY_SMOOTHING_FRAMES, max_track_speed=MAX_TRACK_SPEED`. Satisfies AC5, AC10. |
| T6 | Opt-in LidarTracker hardening | — | high | `planners/_predict.py` | Add keyword-only `smoothing_frames=0`, `max_track_speed=inf` to `LidarTracker.__init__` (default OFF ⇒ byte-identical for `d_star_lite_predictive`). When on: carry a per-prior-cluster velocity history (deque ≤`smoothing_frames`) along the existing positional association chain; reported `vx/vy` = windowed mean incl. current, magnitude-clamped to `max_track_speed`. Preserve full determinism. Add `MAX_TRACK_SPEED=2.0`, `VELOCITY_SMOOTHING_FRAMES=3`. Satisfies AC9. |
| T7 | Tests TC65*+TCa–TCk | T5, T6 | med | `arena/arena.py` | Implement the Testing Strategy table in `--check`, including the TC65 assertion inversion for the canonical keys and the TCh guard that `d_star_lite_predictive` (TC63/TC64) is unaffected. In-process fixtures for unit TCs; seeded subprocess for integration. Satisfies AC1–AC11. |
| T8 | Quick read + findings | T5, T6 | med | findings doc + `.png`, scratch script | Run 10-seed × {0,5,10,20} for plain dwa + 4 keys (traffic on, arena_v1); render failure-rate-vs-horizon PNG; write the 2×2 table; evaluate gate 1 + gate 2 on the ORACLE rows against the measured plain-dwa rate (AC12); report the lidar rows. **Claude runs this.** Document a miss with next-step levers. |
| T9 | Docs | T5, T6, T7, T8 | low | `CLAUDE.md`, `Mission.md`, `README.md` | Rewrite the "space-time predictive DWA" CLAUDE.md section (braking-inevitability model, cost-to-go field, unchanged floor, 4 keys, EXPERIMENTAL_KEYS=4); note in the D* Lite predictive section that the shared `LidarTracker` default is unchanged; update Mission.md Phase 7; add the two new `--algorithm` values to README (`keep-readme-current-with-commands`). |

## Notes for Implementer

- **Byte-identity is the tripwire.** T1's `_heading_term` extraction and `state` threading
  must not perturb plain `dwa` (TC65 guards it). Run TC65 before and after T1.
- **The floor stays FULL** (judge ruling): never subtract a tracked mover's live return.
  "Single count" comes from the future layer being soft, not from thinning the present one.
- **The braking trajectory must include the stationary HELD tail** — decelerate to 0 then
  hold position for the rest of `H`; a robot stopped in a crosser's lane is still an ICS,
  so `trajectory_conflict` must check the full braking+held path, not just the ramp.
- **Soft term is symmetric-clipped, not floored at 0** — `clip(min_gap, -cap, cap)/cap`.
  Flooring at 0 makes faster colliders win ties (the regression the review caught).
- **Determinism traps:** RNG-free fixed-`H` braking rollout; `max(0,·)` on `v`; pin the
  field heap tie-break to `(dist, cell)`; tracker smoothing history dies with the rebuilt
  tracker; keep the argmax strict-`>` fixed order.
- **Oracle-first sequencing:** AC12 gates on the oracle rows; the lidar rows (needing T6)
  are reported alongside. Oracle pass + lidar miss ⇒ perception finding, not a spec block.
- **`MAX_TRACK_SPEED=2.0`**, not 1.5 — 1.5 would truncate genuine `fast`-regime crossers in
  the issue-#11 speed sweep (`dwa_predictive` is in that focus set).
- **`wallclock_per_step` does NOT measure `act()`** — the extra braking rollout per
  candidate does not show there; judge cost by total episode wall time.
- **Run everything with the venv python** (`& .venv\Scripts\python.exe -m ...`).
- **Rollback:** additive except the in-place `dwa_predictive[_oracle]` rewrite and the T1
  refactor; `git revert` restores the old two-layer behavior. Paper-only keys,
  `_costfield.py`, and the opt-in tracker args are pure additions.
