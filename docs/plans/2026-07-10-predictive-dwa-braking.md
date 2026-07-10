# Predictive DWA v2 — Admissible-Velocity Braking + Cost-to-Go Guidance

**Goal:** Rebuild the predictive DWA family so it faithfully matches Missura & Bennewitz
(ICRA 2019) — one soft admissible-velocity braking model instead of a double-counted
hard-reject stack — and adds a static cost-to-go guidance field, so a predictive DWA
actually beats plain DWA on the arena's dense crossing traffic instead of being
net-harmful.

**Approach:** Replace the current two-layer (present-position hard floor + space-time
hard reject) model with a single-count soft model: walls (and un-tracked returns) keep
the present-position lidar floor; tracked movers are removed from that floor and checked
once by an admissible-velocity braking rule (slow to stay stoppable; hard-reject only an
imminent unstoppable conflict). A static cost-to-go field (one Dijkstra from the goal at
reset) replaces DWA's Euclidean heading term to cure the local-minima timeouts. Ship as a
2×2 ablation: {paper-only, paper+global} × {oracle, lidar}, oracle-first. The paper+global
lidar variant becomes the canonical `dwa_predictive`; the LidarTracker is hardened
(smoothing + speed clamp) so the canonical key does not inherit the known-harmful raw
estimator.

## Scope

- **In scope:**
  - Byte-preserving refactor of `DWAController` to expose two seams: `_heading_term`
    (the goal-heading score block) and `_present_floor_points` (identity in base).
  - A pure `build_cost_to_go_field(grid, goal_cell)` Dijkstra-from-goal field builder.
  - Rewrite of the predictive DWA `_evaluate_candidate`: admissible-velocity braking
    over the reused `trajectory_conflict`, single-count floor filter with a blindness
    guard, risk-budgeted so ~20 crossers cannot collapse speed to zero.
  - A global-guidance mixin: build the cost-to-go field at reset, override `_heading_term`
    to score by field-value decrease; fall back to base heading (no `planner_error`) if
    the field build fails.
  - Four registry keys: `dwa_predictive` / `dwa_predictive_oracle` REPLACED in place with
    the paper+global behavior; new `dwa_predictive_paper` / `dwa_predictive_paper_oracle`
    for the paper-only ablation. `EXPERIMENTAL_KEYS` grows to 4; canonical set stays 13.
  - LidarTracker hardening: windowed-mean velocity smoothing (≤3 associated frames) +
    clamp to the env max obstacle speed, deterministic, state cleared on reset.
  - New/updated TCs in `arena/arena.py --check` covering all of the above.
  - A 10-seed × {0,5,10,20} quick read (all variants + plain dwa), a findings doc + PNG,
    and evaluation of the tightened 2×2 success gates. **Run by Claude.**
  - Docs: CLAUDE.md's DWA-predictive section, Mission.md's Phase 7 note, README if a
    command/flag changes.
- **Out of scope:**
  - The full 50-seed canonical / horizon / speed sweeps — **the user launches these**;
    this spec only sets them up and runs the 10-seed quick read.
  - A velocity-obstacle (VO) steering layer, SIPP, or any new non-DWA planner family
    (considered by the consult, deliberately deferred as separate future work).
  - Global-path *replanning* / any `--replan-k` on the DWA-predict family (the field is
    plan-once at reset).
  - `trajectory_conflict` / `predict_blocked_cells` signature changes — reused as-is.

## Decisions

- **Match-the-paper mechanism = admissible-velocity braking** (chosen over a TTC-threshold
  or a VO gate) — it is the literal Missura & Bennewitz model and structurally cannot
  freeze (every heading keeps an admissible speed, including ~0).
- **Guidance = cost-to-go field, not a WaypointFollower** — both approach consultants
  (Fable + Codex) independently made this their top pick; a scalar geodesic field cannot
  wedge under constant lateral displacement by ~20 crossers, the exact pathology this repo
  already paid to fix for the follower (`project-replanning-commitment-horizon`).
- **Single-count with a blindness guard** — a mover is dropped from the present-position
  floor only if it is within a tracked obstacle's current disk; an un-tracked (missed /
  over-segmented) return stays in the floor as a zero-velocity point. Trades a little
  conservatism for never going blind — required because the raw LidarTracker over-segments.
- **Risk-budgeted braking** — the hard admissibility check keys off the *earliest* (binding)
  matched-time conflict from `trajectory_conflict`; distant crossers only shape the soft
  score via `min_gap`. So many simultaneous crossers do not each veto the window.
- **Replace canonical in place; paper-only is a new experimental pair** — the main-scatter
  `dwa_predictive` dot becomes the strongest (paper+global) variant; the paper-only keys
  are experimental ablation, off the canonical set.
- **Tightened 2×2 success bar** — prediction must beat its own guided baseline, not merely
  clear 0.50 (guidance alone would false-positive a plain-0.50 gate). See Acceptance
  Criteria AC11.
- **Harden the LidarTracker now** — shipping the raw 1-frame estimator under a *canonical*
  key would repeat the measured `d_star_lite_predictive` net-harm; smoothing + clamp is
  cheap and deterministic. Oracle is unaffected (perfect velocities).
- **h0 semantics** — paper-only h0 is byte-identical to plain `dwa`; paper+global h0 is
  field-guided-without-prediction, a deliberate 2×2 ablation cell (NOT equal to plain dwa).
- **`BRAKE_DECEL = MAX_LINEAR_ACCEL` (2.0)** — the admissibility bound uses the dynamic
  window's own physics rather than a free-floating constant.

## Acceptance Criteria

Correctness / determinism (verified by TCs):
- [ ] **AC1** Plain `dwa` trace JSONL is byte-identical to pre-change on the same seed
  (`--no-traffic` and traffic-on) — the base refactor is byte-preserving.
- [ ] **AC2** `dwa_predictive_paper` and `dwa_predictive_paper_oracle` at `h0` produce a
  byte-identical trace to plain `dwa` (both `--no-traffic` and traffic-on).
- [ ] **AC3** `dwa_predictive` and `dwa_predictive_oracle` (paper+global) at `h0` are
  deterministic (two same-seed runs byte-identical) and are NOT byte-identical to plain
  `dwa` (field guidance is active) — the global-only ablation cell.
- [ ] **AC4** All four keys: two same-seed traffic-on runs are byte-identical; every trace
  line carries the 8-key schema (incl. `dynamic_obstacles_sha256`).
- [ ] **AC5** `--predict-horizon` is required for all four keys (omission → exit 2);
  `--replan-k` is rejected for all four (exit 2); `algorithm_label` folds `_h<steps>`.
- [ ] **AC6** Single-count + blindness guard: a tracked mover's returns are excluded from
  the present-position floor; an obstacle with no track keeps its returns in the floor
  (verified on a synthetic split). No candidate is rejected by both the floor and the
  braking layer for the same mover.
- [ ] **AC7** `build_cost_to_go_field` returns octile goal-distances matching an A*-cost
  oracle on reachable cells, `inf` on unreachable cells, and is deterministic; a field
  whose start cell is unreachable makes the global variant fall back to base heading with
  `planner_error` null.
- [ ] **AC8** Hardened LidarTracker is deterministic across a multi-frame cluster-count
  change (byte-identical `Track` tuples on two fresh runs), never reports a speed above
  `MAX_TRACK_SPEED`, and clears smoothing state on `reset()`.
- [ ] **AC9** `run_all` canonical set == 13; `EXPERIMENTAL_KEYS == {d_star_lite_oracle,
  dwa_predictive_oracle, dwa_predictive_paper, dwa_predictive_paper_oracle}`; the import
  assertion `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS` holds; all four
  keys are in `PREDICT_FAMILIES`.
- [ ] **AC10** `act()` never raises mid-episode for any key; the DWA family's
  `planner_error` is always null (field-build fallback preserves this).

Empirical (produced + evaluated by the quick read, T8 — documented, not silently passed):
- [ ] **AC11** The 10-seed × {0,5,10,20} quick read runs for plain `dwa` + all four
  predictive keys; results, the 2×2 table, and a PNG land in a findings doc. **PASS** =
  **gate 1** (`dwa_predictive_paper` best-horizon failure_rate < 0.50) **AND gate 2**
  (`dwa_predictive` best-horizon failure_rate < the `dwa_predictive` h0 = global-only
  cell). A miss is recorded as a finding with the next-step levers (weights, horizon,
  estimator), not reported as success.

## Contracts & Interfaces

Single source of truth for every cross-task seam.

### New / changed signatures

- `DWAController._heading_term(self, trajectory: np.ndarray) -> float` — **owner T1**;
  overridden by the global-guidance mixin (**consumer T4**). Base body is the exact
  goal-heading block lifted verbatim from `_score` (returns the `[0,1]` heading alignment,
  `1.0` inside `GOAL_REACHED_RADIUS`). `_score` calls `HEADING_WEIGHT * self._heading_term(traj)`.
- `DWAController._present_floor_points(self, state: np.ndarray, obstacle_points: np.ndarray) -> np.ndarray`
  — **owner T1**; identity in base (returns `obstacle_points` unchanged). Called once in
  `act()` immediately after `obstacle_points` is computed, before the candidate loop.
  Overridden by the predictive base (**consumer T3**) to drop tracked-mover returns.
- `build_cost_to_go_field(grid: OccupancyGrid, goal_cell: tuple[int, int]) -> np.ndarray`
  — **owner T2**, new module `planners/_costfield.py`. Dijkstra from `goal_cell` over
  `grid.cells`, SAME cost model as `astar_search` (8-connected, `np.hypot` octile step
  cost, no corner-cutting: a diagonal is blocked if either orthogonal neighbor is
  occupied). Returns a `(rows, cols)` float64 array of goal-distances; occupied and
  unreachable cells are `inf`. Deterministic: `(dist, (row, col))` heap entries so ties
  break on cell index, exactly like `astar_search`. **Consumer T4.** Pure — imports only
  `manual_astar.OccupancyGrid` + numpy + heapq, no irsim.
- `LidarTracker` (`planners/_predict.py`) — **owner T6**: constructor gains no required
  arg; internally carries a per-prior-cluster short velocity history along the existing
  positional association chain. `update()` return type unchanged (`list[Track]`), but
  `Track.vx/vy` are now the windowed-mean over the last ≤`VELOCITY_SMOOTHING_FRAMES`
  associated instantaneous velocities, magnitude-clamped to `MAX_TRACK_SPEED`. Consumers:
  the two lidar predictive keys (via `_make_tracker`).

### Reused as-is (no signature change)

- `trajectory_conflict(robot_positions, tracks, robot_radius, horizon_steps, dt, margin)
  -> TrajectoryConflict(collides, ttc_step, min_gap)` — the braking model reads `ttc_step`
  (earliest binding conflict) and `min_gap` (soft gradient). No change.
- `OracleTracker`, the `Tracker` protocol, `Track`, the truth seam
  (`wants_truth` / `observe_truth`), `world_to_grid`, `build_occupancy_grid`, `load_world`.

### Registry / family sets (`planners/_grid.py`, **owner T5**)

- `ALGORITHMS` gains `dwa_predictive_paper`, `dwa_predictive_paper_oracle`.
- `PREDICT_FAMILIES = {d_star_lite_oracle, d_star_lite_predictive, dwa_predictive,
  dwa_predictive_oracle, dwa_predictive_paper, dwa_predictive_paper_oracle}`.
- `EXPERIMENTAL_KEYS = {d_star_lite_oracle, dwa_predictive_oracle, dwa_predictive_paper,
  dwa_predictive_paper_oracle}` (grows from 2 to 4; `dwa_predictive` stays canonical).

### New module-level constants (`planners/dwa_predictive.py`, owners T3/T4)

- `BRAKE_DECEL = MAX_LINEAR_ACCEL` (2.0) — admissibility deceleration.
- `PREDICTED_GAP_WEIGHT` — soft-term weight on normalized `min_gap` (replaces
  `PREDICTED_CLEARANCE_WEIGHT`; default ≤ `CLEARANCE_WEIGHT` = 0.3, NOT 0.4).
- `FLOOR_DROP_MARGIN` — extra band added to `track.radius` when deciding which floor
  returns belong to a tracked mover.
- `GUIDANCE_WEIGHT` reuse of `HEADING_WEIGHT` (0.8) via `_heading_term` (no new weight);
  `MAX_PROGRESS_PER_ROLLOUT = MAX_LINEAR_SPEED * ROLLOUT_STEPS * CONTROL_DT` normalizer.

### New constants (`planners/_predict.py`, owner T6)

- `MAX_TRACK_SPEED = 1.5` (m/s) — env obstacle speed cap (0.3–1.5× robot top speed 1.0).
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
| `docs/plans/2026-07-10-predictive-dwa-braking.findings.md` (+ .png), quick-read script | T8 | — |
| `CLAUDE.md`, `Mission.md`, `README.md` | T9 | — |

## Algorithm detail — the soft model (Notes for T3)

Per candidate `(v, w)` with rollout `trajectory` (S poses at `k·dt`):
1. **Floor (walls + un-tracked):** `clearance = _trajectory_clearance(trajectory[:ROLLOUT_STEPS], floor_points)`
   where `floor_points` already has tracked-mover returns removed (in `_present_floor_points`).
   If `None` → reject (unchanged vanilla rule).
2. **Base score:** `base = _score(...)` using `_heading_term` (Euclidean for paper-only,
   field for global) + clearance + speed.
3. **No tracks** (h0, or empty): return `base` (paper-only h0 ⇒ plain dwa).
4. **Predicted conflict:** `c = trajectory_conflict(trajectory, tracks, r, H, dt, margin)`.
   - If `c.collides` (earliest binding step `k_c = c.ttc_step`): `d_c = max(0, v · k_c · dt)`;
     `v_adm = sqrt(2 · BRAKE_DECEL · d_c)`. If `v > v_adm` → **reject** (cannot stop before
     the conflict = imminent). Else keep (the robot will brake over the next ticks).
   - Soft term (always, capped): `base + PREDICTED_GAP_WEIGHT · clip(c.min_gap, 0, CLEARANCE_CAP)/CLEARANCE_CAP`.
   Determinism: `max(0, d_c)` guards the sqrt; the argmax keeps the existing strict-`>`,
   fixed-iteration-order tie-break.

`_present_floor_points` (T3 override): drop every `obstacle_point` within
`track.radius + FLOOR_DROP_MARGIN` of any current `self._tracks` center (blindness guard:
un-tracked returns survive). Works for oracle (truth centers) and lidar (cluster centers).

Global `_heading_term` (T4 override): `progress = field[start_cell] - field[end_cell]`
(via `world_to_grid`); if either is `inf` (unreachable/blocked) treat as `0` progress;
return `clip(progress / MAX_PROGRESS_PER_ROLLOUT, 0, 1)`. Keep the `GOAL_REACHED_RADIUS`
suppression → `1.0` near goal.

## Error Handling

- **Field build fails at reset** (goal cell unreachable / walled): the global variant logs
  nothing to `planner_error`, drops `use_global_guidance` for the episode, and uses base
  Euclidean heading. Preserves DWA's "never fails to plan" property (AC7, AC10).
- **Tracker / prediction raises inside `act()`:** caught (`except Exception`), that tick
  degrades to plain-DWA scoring (no braking layer) — never propagates (AC10). Matches the
  current predictive `act()` guard.
- **Empty dynamic window / all candidates rejected:** the existing in-place-rotation
  fallback fires. The soft braking model makes a total-reject far rarer (every heading
  keeps an admissible low speed), but the fallback stays as the backstop.
- **Reused instance across episodes:** `reset()` clears the field, tracker, tracks,
  snapshot, and (T6) the smoothing history, so episode 2 never differences against
  episode 1.

## Testing Strategy

**Levels:** Unit (pure helpers, in-process) + Integration (subprocess runner drives,
mirrors existing TC pattern). Added to `python arena/arena.py arena/arena_v1.yaml --check`.

| ID | Test Case | Type | Expected |
|----|-----------|------|----------|
| TC65* | plain `dwa` unchanged; paper-only h0 == plain dwa (`--no-traffic` + traffic) | Integration | byte-identical trace (AC1, AC2) |
| TCa | paper+global h0 deterministic AND != plain dwa | Integration | two same-seed runs equal; differs from plain dwa (AC3) |
| TCb | all four keys traffic-on e2e + determinism, 8-key schema | Integration | terminal state, byte-identical pair, 8 keys/line (AC4) |
| TCc | `--predict-horizon` required / `--replan-k` rejected for the 4 keys; `_h<steps>` label | Integration | exit 2 on violation; no `<seed>.json` written (AC5) |
| TCd | `build_cost_to_go_field` == A* cost on reachable cells; inf on sealed cells; deterministic | Unit | equal to oracle; inf; byte-identical (AC7) |
| TCe | admissible-braking pure logic: imminent conflict rejected, brakeable kept, soft gap monotone | Unit | reject/keep per formula; deterministic (AC6 core) |
| TCf | floor blindness guard: tracked mover dropped from floor; un-tracked return kept | Unit | membership per `FLOOR_DROP_MARGIN` (AC6) |
| TCg | hardened LidarTracker: multi-frame cluster-count change deterministic; speed ≤ cap; reset clears | Unit | byte-identical Tracks; clamp; state cleared (AC8) |
| TCh | field-build failure → global variant falls back, `planner_error` null | Integration | reaches terminal / null error (AC7, AC10) |
| TCi | `run_all` canonical == 13; `EXPERIMENTAL_KEYS` == the 4; assertion holds; 4 keys in `PREDICT_FAMILIES` | Unit (in-process) | as stated (AC9) |

`TC65*` = extend the existing TC65 to also assert the two paper-only keys. **Test data:**
synthetic grids/tracks built in-process (no irsim) for TCd–TCg,TCi; the seeded subprocess
runner for the integration TCs. **Run command:**
`& .venv\Scripts\python.exe arena/arena.py arena/arena_v1.yaml --check`.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Base DWA seams (byte-preserving) | — | med | `planners/dwa.py` | Extract the goal-heading block of `_score` into `_heading_term(trajectory)->float` (base returns the exact current value; `_score` calls `HEADING_WEIGHT*self._heading_term(traj)`). Add `_present_floor_points(state, obstacle_points)->np.ndarray` returning input unchanged, called once in `act()` right after `obstacle_points` is built. Must keep plain `dwa` byte-identical. Satisfies AC1. |
| T2 | Cost-to-go field builder | — | med | `planners/_costfield.py` (new) | Pure `build_cost_to_go_field(grid, goal_cell)->np.ndarray`: Dijkstra from goal, octile step cost via `np.hypot`, no corner-cutting (diagonal blocked if either orthogonal neighbor occupied), `(dist,(r,c))` heap for deterministic ties; occupied/unreachable = inf. Imports only `manual_astar.OccupancyGrid`, numpy, heapq. Satisfies AC7 (field half). |
| T3 | Predictive base: soft braking + single-count floor | T1 | high | `planners/dwa_predictive.py` | Rewrite `PredictiveDWAController._evaluate_candidate` per "Algorithm detail": admissible-velocity braking over `trajectory_conflict` (`d_c=max(0,v*ttc*dt)`, reject if `v>sqrt(2*BRAKE_DECEL*d_c)`, else soft `min_gap` bonus). Override `_present_floor_points` to drop tracked-mover returns (`FLOOR_DROP_MARGIN`, blindness guard). Add `BRAKE_DECEL`, `PREDICTED_GAP_WEIGHT`, `FLOOR_DROP_MARGIN`. Keep tracker plumbing + `act()` h0 fast-path. Satisfies AC6. |
| T4 | Global-guidance mixin + field heading | T1, T2, T3 | high | `planners/dwa_predictive.py` | Add `use_global_guidance` (class attr). When set: `reset()` builds the cost-to-go field (goal cell via `world_to_grid`); override `_heading_term` to score field-decrease normalized by `MAX_PROGRESS_PER_ROLLOUT`, keeping the `GOAL_REACHED_RADIUS` guard. Field-build failure → fall back to base heading, `planner_error` null. Ensure paper-only h0==plain dwa and global h0=field-only (AC2, AC3, AC7, AC10). |
| T5 | Register 4 keys + family sets | T3, T4 | med | `planners/dwa_predictive.py`, `planners/_grid.py` | Concrete classes: `DWAPredictiveController`(dwa_predictive, lidar, global), `DWAPredictiveOracleController`(dwa_predictive_oracle, oracle, global) — replace behavior in place; `DWAPredictivePaperController`(dwa_predictive_paper, lidar, no global), `DWAPredictivePaperOracleController`(dwa_predictive_paper_oracle, oracle, no global). `register()` the two new keys; add both to `PREDICT_FAMILIES` and `EXPERIMENTAL_KEYS`. Lidar keys use the hardened tracker. Satisfies AC5, AC9. |
| T6 | Harden LidarTracker | — | high | `planners/_predict.py` | Carry a per-prior-cluster velocity history (deque ≤`VELOCITY_SMOOTHING_FRAMES`) along the existing positional association chain; reported `vx/vy` = windowed mean incl. current instantaneous, magnitude-clamped to `MAX_TRACK_SPEED`. Clear history in the tracker's per-episode reset path. Preserve full determinism (fixed association order, no set-iteration into output). Satisfies AC8. |
| T7 | Tests TC65*+TCa–TCi | T5, T6 | med | `arena/arena.py` | Implement the Testing Strategy table in `--check`. Extend TC65 for the paper-only keys; add TCa–TCi. In-process fixtures for the pure ones (TCd–TCg,TCi), seeded subprocess for integration. Satisfies AC1–AC10. |
| T8 | Quick read + findings | T5, T6 | med | `docs/plans/2026-07-10-predictive-dwa-braking.findings.md`, `.png`, quick-read script | Run 10-seed × {0,5,10,20} for plain dwa + 4 keys (traffic on, arena_v1); render a failure-rate-vs-horizon PNG; write the 2×2 table and evaluate gate 1 + gate 2 (AC11). **Claude runs this.** Document a miss with next-step levers. |
| T9 | Docs | T5, T6, T7, T8 | low | `CLAUDE.md`, `Mission.md`, `README.md` | Rewrite the "space-time predictive DWA" CLAUDE.md section for the new model (admissible braking, cost-to-go field, blindness guard, 4 keys, EXPERIMENTAL_KEYS=4, hardened tracker). Update Mission.md Phase 7 note. Update README only if a command/flag changed (none expected). |

## Notes for Implementer

- **Byte-identity is the tripwire.** T1's `_heading_term` extraction and
  `_present_floor_points` identity insertion must not perturb plain `dwa` (TC65 guards it) —
  compute in the same order, same floats. Run TC65 before and after T1.
- **Blindness guard is not optional** (Fable veto): removing movers from the floor *without*
  the un-tracked fallback turns the raw estimator's over-segmentation into invisible
  obstacles → crashes. Drop a floor return only when it is inside an actual track disk.
- **The braking uses the earliest conflict only** (risk-budgeting, Codex): do not AND a
  separate admissibility constraint per crosser — `trajectory_conflict` already returns the
  binding `ttc_step`; that single constraint plus the soft `min_gap` is the whole rule.
- **Determinism traps:** clamp `d_c = max(0.0, ...)` before `sqrt`; pin the field heap
  tie-break to `(dist, cell)`; clear the LidarTracker smoothing history on reset (reuse
  contract); keep the candidate argmax strict-`>` in fixed order.
- **Oracle-first sequencing:** the AC11 gate is judged first on the oracle rows; the lidar
  rows (needing T6) are reported alongside. If the oracle passes but lidar does not, that
  is a perception finding, not a policy failure — record it, do not block the spec.
- **`wallclock_per_step` does not measure `act()`** (`gotcha-wallclock-per-step-excludes-act`)
  — do not use it to judge the braking/field cost; total episode wall time is the signal.
- **Run everything with the venv python** (`& .venv\Scripts\python.exe -m ...`); PATH python
  lacks irsim (`gotcha-use-venv-python-not-path-python`).
- **Rollback:** the whole change is additive except the in-place `dwa_predictive[_oracle]`
  rewrite; git revert restores the old two-layer behavior. The paper-only keys and
  `_costfield.py` are pure additions.
