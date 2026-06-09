# Phase 6 — Reactive & Sampling Planners Plan

**Goal:** Implement the remaining Mission.md Phase 6 planner families — **DWA** and **APF**
(reactive, velocity output) and **RRT** and **RRT\*** (sampling, path output, `_once` +
`_replan` variants) — and fix the tracked follower-commitment bug so every `_replan`
family becomes a valid comparison point for the scatter plot.

**Approach:** Plug each new planner into the existing `Controller` interface + `ALGORITHMS`
registry seam (no runner changes — both runners are already planner-agnostic). Reactive
planners output `(v, ω)` directly from the live lidar + the YAML goal. Sampling planners are
hand-rolled in `manual_astar` style with a deterministic per-plan RNG (for byte-identical
traces) and reuse the shared grid occupancy / line-of-sight / `WaypointFollower` substrate so
the comparison isolates the search, not the follower. The `_replan` families adopt the
commitment-horizon policy already proven in `DStarLiteController`, ported into the shared
`PathFollowingController` base, so `a_star_replan`, `dijkstra_replan`, `rrt_replan`, and
`rrt_star_replan` all traverse correctly. A GitHub issue tracks both a stronger follower and
consolidating the now-duplicated commitment logic as future work.

## Scope

- **In scope:**
  - `planners/dwa.py` — `DWAController` (registry key `dwa`): reactive Dynamic Window
    Approach; samples feasible `(v, ω)`, forward-simulates each over a short rollout horizon,
    scores by heading-to-goal + clearance + velocity, picks the best. Velocity output.
  - `planners/apf.py` — `APFController` (registry key `apf`): Khatib 1986 Artificial
    Potential Fields; attractive force to goal + repulsive force from live lidar returns.
    Velocity output.
  - `planners/rrt.py` — hand-rolled RRT core + `RRTOnceController` (`rrt_once`) and
    `RRTReplanController` (`rrt_replan`).
  - `planners/rrt_star.py` — RRT\* (choose-parent + rewire) reusing the RRT core +
    `RRTStarOnceController` (`rrt_star_once`) and `RRTStarReplanController`
    (`rrt_star_replan`).
  - Port `DStarLiteController`'s commitment horizon into `PathFollowingController.act()`
    (`planners/_grid.py`); add `rrt_replan`, `rrt_star_replan` to `REPLAN_FAMILIES`.
  - New TCs (TC38–TC45) + a TC32 rewrite, registered in `arena/arena.py`'s `--check` suite.
  - Docs: CLAUDE.md planner-family section, Mission.md Phase 6 status, this plan doc.
  - A GitHub issue documenting (a) the better-follower future work and (b) consolidating the
    two commitment-horizon copies into one shared helper.
- **Out of scope:**
  - Any change to `runners/run_episode.py` / `runners/run_experiment.py` — both are already
    generic over the registry (verified: they forward `--replan-k` and record it; the only
    plumbing is the `REPLAN_FAMILIES` membership edit in `_grid.py`). No new metrics field is
    added to the runner; the planned-path-cost observation (AC7-obs) is computed in-process by
    the test, not threaded through the runner.
  - Modifying `DStarLiteController` — it already commits correctly and stays byte-unchanged.
    Note this means the commitment-horizon logic will exist in TWO places (its own `act()`
    and the new `PathFollowingController`); that duplication is acknowledged and tracked in
    the GitHub issue, not resolved here.
  - Phase 5 `plot.py` / the scatter plot (does not exist yet) and Phase 6b K-sweep.
  - Adding goal-biased escape heuristics to DWA/APF — they stay pure Mission algorithms.
  - New world fixtures — reuse `arena/arena_v1.yaml` and `arena/arena_no_path.yaml`.
  - Implementing the alternative followers (cost-hysteresis / index-transplant /
    pure-pursuit) — captured as a GitHub issue only.

## Decisions

- **All four families this pass** (DWA, APF, RRT, RRT\*) — completes the Phase 6 list.
- **Hand-roll RRT/RRT\***, not a library — the harness's byte-identical-trace guarantee
  needs a controlled RNG; mirrors how D\* Lite was hand-rolled.
- **Deterministic per-plan RNG.** `_once`: a single plan at `reset()` uses
  `np.random.default_rng(RRT_SEED)`. `_replan`: each `_plan` call uses
  `np.random.default_rng(RRT_SEED + self._k)` so successive replans explore *different*
  samples (a replan that re-derives the same path would defeat replanning) while staying
  byte-deterministic (the `_k` sequence is deterministic). No change to the
  `Controller.reset()` signature; cross-seed variation comes from traffic.
- **Both `_once` and `_replan`** for the sampling families — Mission-faithful; feeds the
  `_once`-vs-`_replan` experiment and the future K-sweep.
- **Replan fix = commitment horizon, ported into the shared base** — the grid + RRT
  replanners adopt the same policy `DStarLiteController` already uses, so the scatter plot
  measures searches, not followers. D\* Lite keeps its own copy (left untouched); a GitHub
  issue records both the consolidation cleanup and the stronger-follower alternatives
  (B cost-hysteresis, C index-transplant, D pure-pursuit).
- **Reactive acceptance = run-to-completion + valid trace schema** — DWA/APF are expected to
  stall/crash in arena_v1's corridors; that is the experimental signal, so their TCs do not
  require reaching the goal.
- **Test depth = match existing density** (~8 new TCs + 1 rewrite).
- **Branch** off the current `phase6-grid-planners` HEAD (holds the unmerged grid + D\* Lite
  work), not `main`.

## Acceptance Criteria

- [x] **AC1 — DWA.** `dwa` is registered, reactive (forward-simulates a window of `(v, ω)`,
  scores via goal-heading + lidar clearance + velocity), rejects `--replan-k`, and drives a
  full traffic-on episode end-to-end emitting a well-formed 8-key trace (goal-reaching NOT
  required). `reset()` never raises `planner_error`.
- [x] **AC2 — APF.** `apf` is registered, computes Khatib attractive (to goal) + repulsive
  (from live lidar) forces into a clamped `(v, ω)`, rejects `--replan-k`, runs an episode
  end-to-end with a valid trace. `reset()` never raises `planner_error`.
- [x] **AC3 — Sampling registration.** `rrt_once`, `rrt_replan`, `rrt_star_once`,
  `rrt_star_replan` are registered; `rrt_replan`/`rrt_star_replan` are in `REPLAN_FAMILIES`
  (require `--replan-k`); `rrt_once`/`rrt_star_once` reject `--replan-k`. Each instance's
  `.name` equals its registry key.
- [x] **AC4 — Sampling determinism.** Two same-seed `rrt_once`, `rrt_star_once`, AND
  `rrt_replan` runs each produce byte-identical `trace.jsonl` (deterministic per-plan RNG).
- [x] **AC5 — Sampling solves the static world.** `rrt_once` and `rrt_star_once` reach the
  goal on `arena_v1.yaml --no-traffic` with a measured sim-time **strictly below the 120 s
  cap, recorded here**: `rrt_once` ≈ 73.0 s, `rrt_star_once` ≈ 70.7 s (target ≤ ~110 s to
  absorb cross-machine/numpy float drift). The driven path relies on
  `rrt_points_to_waypoints` line-of-sight shortcutting to stay short. The `RRT_SEED` /
  `RRT_MAX_ITERS` used to hit these are recorded in the constants block.
- [x] **AC6 — Sampling no-path failure.** `rrt_once` and `rrt_star_once` on
  `arena_no_path.yaml` raise a no-path error (whether by exhausting the iteration cap or by a
  start/goal-blocked guard) → the runner records a non-null `planner_error` and writes no
  `trace.jsonl`.
- [x] **AC7 — RRT\* solves the static world (blocking).** `rrt_star_once` reaches the goal on
  `arena_v1.yaml --no-traffic` (the only suite-failing quality gate for RRT\*).
- [x] **AC7-obs — RRT\* quality (non-blocking).** The test records (logged, non-gating) the
  *planned* path cost of `rrt_star_once` vs `rrt_once` at the fixed seed/budget, computed
  in-process by calling the planners directly, and observes whether rewiring reduced the
  planned cost. This is NOT asserted with `≤` and does NOT fail the suite.
- [x] **AC8 — Commitment-horizon fix (binding).** `a_star_replan` AND `dijkstra_replan`
  reach the goal on `arena_v1.yaml --no-traffic` (they previously timed out / collided).
- [x] **AC9 — Cadence preserved & fold-free check.** `compute_path` still fires only on every
  K-th `act()`. The committed-segment check is evaluated on each replan (K-th) act only;
  `_immediate_segment_blocked` reads `self._last_fold` (written exclusively inside
  `compute_path`) and calls `segment_is_clear_grid` only — it never calls
  `lidar_to_occupancy`, so it adds no folds. TC31 passes unchanged.
- [x] **AC10 — Swap semantics.** On a replan (K-th) act, the follower is swapped to the fresh
  plan ONLY when the current follower is finished OR its immediate committed segment
  (robot → current target waypoint) is blocked in the latest fold; otherwise the follower is
  kept (committed). A failed replan keeps the follower. Worst-case obstacle-reaction latency
  is K ticks by design (see the non-goal note in Notes).
- [x] **AC11 — D\* Lite untouched.** `DStarLiteController.act()` is byte-unchanged; TC35–TC37
  still pass. (The ported logic is a behavioral copy, not a shared call site — see Scope.)
- [x] **AC12 — Labeled outputs.** `rrt_replan`/`rrt_star_replan` write to
  `results/<world_stem>/<family>_k<K>/` and produce 8-key traces with traffic on;
  `run_experiment` records `replan_k` in `_manifest.json` (verify; no code change).
- [x] **AC13 — Suite green.** `python arena/arena.py arena/arena_v1.yaml --check` passes all
  prior TCs plus the new ones; the `--check` PASS-count + runtime comments are updated.
- [x] **AC14 — Issue filed.** A GitHub issue documents the better-follower future work
  (B/C/D) and the commitment-horizon consolidation, referencing the chosen approach and why.
- [x] **AC15 — Docs.** CLAUDE.md (planner family: new families + commitment-horizon resolved
  + new TCs), Mission.md (Phase 6 status), and this plan doc are updated. AGENTS.md is
  untracked and stays out of scope.

## Data Model / Interfaces

All new controllers implement the existing `planners._types.Controller` protocol
(`name: str`, `reset(world_yaml, initial_snapshot, lidar0, state0) -> None`,
`act(state, lidar) -> (2,1) float ndarray`). No interface changes.

```text
Registry additions (planners/_grid.py REPLAN_FAMILIES gains the last two):
  dwa              -> DWAController           (reactive, rejects --replan-k)
  apf              -> APFController           (reactive, rejects --replan-k)
  rrt_once         -> RRTOnceController       (standalone, static-grid plan, rejects -k)
  rrt_replan       -> RRTReplanController     (PathFollowingController subclass, REQUIRES -k)
  rrt_star_once    -> RRTStarOnceController   (standalone, static-grid plan, rejects -k)
  rrt_star_replan  -> RRTStarReplanController (PathFollowingController subclass, REQUIRES -k)

REPLAN_FAMILIES = {a_star_replan, dijkstra_replan, rrt_replan, rrt_star_replan}
```

### `PathFollowingController` commitment-horizon refactor (`planners/_grid.py`)

```python
# New/changed state
self._last_fold: np.ndarray | None = None    # occupancy the latest plan was built on
self._planned_path: Path | None = None       # freshest plan from the last K-th act

def compute_path(self, state, lidar) -> Path:
    folded = lidar_to_occupancy(self._static_cells, self._grid, state, lidar,
                                self._geom, self._inflation)
    self._last_fold = folded                  # store for the fold-free segment check
    folded_grid = OccupancyGrid(cells=folded, resolution=self._grid.resolution,
                                offset=self._grid.offset)
    return self._plan(folded_grid, folded, state)   # overridable search hook

def _plan(self, folded_grid, folded, state) -> Path:   # default: A*/Dijkstra
    cur_cell = world_to_grid(state[:2], folded_grid)
    cells_path = astar_search(folded_grid, cur_cell, self._goal_cell,
                              type(self).heuristic_fn)
    return grid_path_to_waypoints(cells_path, self._grid, folded,
                                  state[:2], self._goal_xy, WAYPOINT_STRIDE)

def act(self, state, lidar) -> np.ndarray:
    if self._follower is None: raise RuntimeError("act() before reset()")
    self._k += 1
    if self._replan_k is not None and self._k % self._replan_k == 0:
        try:
            new_path = self.compute_path(state, lidar)   # refresh knowledge every K
        except (ValueError, RuntimeError):
            new_path = None
        if new_path:
            self._planned_path = new_path
            # commitment horizon: adopt the fresh plan ONLY if the current
            # commitment is exhausted or invalidated.
            if self._follower.is_finished or self._immediate_segment_blocked(state[:2]):
                self._follower = WaypointFollower(list(new_path),
                                                  WAYPOINT_REACHED_DISTANCE)
    return compute_action_from_state(state, self._follower)

def _immediate_segment_blocked(self, position) -> bool:
    # Reuses the stored last fold (no lidar_to_occupancy call -> adds no folds, AC9).
    if self._last_fold is None: return False
    target = self._follower.current_waypoint(position)
    return not segment_is_clear_grid(self._last_fold, self._grid, position, target)
```

**Convergence rationale (why this fixes AC8, and why TC45 is the gate).** This is a behavioral
copy of `DStarLiteController._immediate_segment_blocked` (it calls
`follower.current_waypoint(position)` + `segment_is_clear_grid(..., position, target)` against
the stored fold). On a clear `--no-traffic` run the committed segment stays clear and the
follower never finishes mid-traverse, so **no swap ever fires** and `a_star_replan` follows
its t=0 plan identically to `a_star_once` (already verified to reach the goal at ~73.6 s);
the jitter the old code suffered only manifests when a swap fires, which now happens only when
the immediate segment is actually blocked. The "looks one waypoint ahead" concern is bounded:
`current_waypoint()` advances its index as the robot reaches each waypoint, so a blockage two
waypoints ahead becomes the checked immediate segment by the time the robot reaches the
intervening waypoint — lag bounded by one waypoint-segment. Because the underlying path source
is a fresh A\* re-search (not D\* Lite's persistent tree), this is argued *sufficient* for the
binding AC, NOT provably optimal: **TC45 is the binding empirical gate — if `a_star_replan` OR
`dijkstra_replan` fails to reach the goal on `arena_v1 --no-traffic`, T1's design is revisited
before proceeding.**

`RRTReplanController` / `RRTStarReplanController` subclass `PathFollowingController` and
override only `_plan`. Exact override body (avoids the `folded` vs `folded_grid` foot-gun):

```python
def _plan(self, folded_grid, folded, state) -> Path:
    rng = np.random.default_rng(RRT_SEED + self._k)          # per-plan, deterministic
    start_xy = np.asarray(state[:2], dtype=float)
    points = rrt_plan(folded, folded_grid, start_xy, self._goal_xy, rng)  # raises on no-path
    return rrt_points_to_waypoints(points, self._grid, folded,
                                   start_xy, self._goal_xy, WAYPOINT_STRIDE)
```

### RRT core (`planners/rrt.py`)

```python
RRT_SEED = 0          # STARTING VALUE — T4 must replace with a seed empirically confirmed to
                      # solve arena_v1 --no-traffic with sim-time < 120 s; record it + the
                      # measured sim-times in AC5.
RRT_MAX_ITERS = 5000  # STARTING VALUE — raise during T4 if the tuned seed cannot solve
                      # arena_v1's switchback within budget. Exceeding the cap -> raise (no path).
RRT_STEP = 1.0        # steer step (m) — tunable
RRT_GOAL_BIAS = 0.05  # P(sample == goal) — tunable
RRT_GOAL_TOLERANCE = 0.5   # connect-to-goal radius (m)

def rrt_plan(grid_cells, grid, start_xy, goal_xy, rng) -> list[np.ndarray]:
    """Sample free space (goal-biased), steer toward each sample by RRT_STEP, collision-check
    each edge with segment_is_clear_grid(grid_cells, grid, p_near, p_new); on reaching within
    RRT_GOAL_TOLERANCE of the goal, walk parents back to a continuous start..goal point list.
    Raise ValueError on cap exhaustion OR when start/goal maps to a blocked/out-of-bounds cell
    (so AC6's no-path is a clean raise, not a crash)."""

def rrt_points_to_waypoints(points, grid, grid_cells, start_xy, goal_xy, stride) -> Path:
    """Shortcut the continuous RRT point list into a sparse line-of-sight-safe waypoint tuple.
    Replicate grid_path_to_waypoints' structure: seed `waypoints = [points[0]]`, build the
    stride-downsampled candidate-index list, then call _append_clear_waypoints per candidate
    span (it takes (output, points, start_index, end_index, grid, grid_cells)); pin
    waypoints[0]=start_xy and the last=goal_xy. _append_clear_waypoints is NOT a one-shot
    drop-in — it must be driven over candidate spans exactly as grid_path_to_waypoints does."""

def rrt_planned_cost(points) -> float:
    """Sum of consecutive segment lengths of a planned point path — used by AC7-obs only."""
```

`RRTOnceController` mirrors `AStarOnceController`: `reset()` runs `rrt_plan` on the **static**
occupancy grid (no lidar fold) with `np.random.default_rng(RRT_SEED)`, builds a
`WaypointFollower`, propagates `ValueError` on no-path; `act()` ignores lidar.

### RRT\* core (`planners/rrt_star.py`)

Imports the sampling/steer/nearest/collision helpers from `planners/rrt.py`, adds
choose-parent (within a neighborhood radius) and rewire after each insertion for asymptotic
optimality. `RRTStarOnceController` / `RRTStarReplanController` mirror the RRT controllers
(same per-plan RNG rule).

### Reactive controllers (`planners/dwa.py`, `planners/apf.py`)

`reset()` loads the goal from the YAML via `manual_astar.load_world(...).goal` and the lidar
beam geometry via `planners._grid.load_lidar_geometry(world_yaml)` (deliberate reuse — that
helper already mirrors irsim's beam layout; reactive controllers need it to map beams to
bearings); it stores them and never raises `planner_error`. `act()` reads the live lidar each
tick, projects finite returns to world points, and returns a clamped `(2,1)` action
(`MAX_LINEAR_SPEED` / `MAX_ANGULAR_SPEED` from `manual_astar`). DWA forward-simulates each
sampled `(v, ω)` over a short rollout horizon (not a single-step lookahead). Tunables are
module-level `UPPER_SNAKE_CASE` constants.

## Error Handling

- **Sampling no-path:** `rrt_plan` raises `ValueError` on cap exhaustion or a blocked
  start/goal → `reset()` propagates → runner records `planner_error`, deletes the partial
  trace (existing runner path). `_replan` mid-episode failures are swallowed in `act()` (last
  valid follower kept; AC10).
- **Reactive planners:** no global plan, so `reset()` does not raise; episodes end via
  crash/timeout/goal, recorded normally.
- **Bad `--replan-k`:** handled entirely by the existing `build_controller` validation
  (required for `REPLAN_FAMILIES`, forbidden otherwise) → exit 2; the only change is the
  `REPLAN_FAMILIES` membership.
- **Commitment-horizon replan failure:** `compute_path` raising inside `act()` is caught;
  `self._planned_path` and the live follower are left untouched; `act()` never raises.

## Testing Strategy

**Levels:** Unit (pure controller/registry/search) + Integration (subprocess via the runner),
matching the existing `arena/arena.py --check` harness conventions.

| ID  | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC32 (rewrite) | Commitment swap semantics for `a_star_replan` | Unit | Three branches at `replan_k=1`, robot at (2,2): (a) `compute_path` patched to raise → act() keeps the SAME follower object; (b) successful replan with an all-NaN frame (immediate segment clear) → follower KEPT across the replan (identity preserved, the committed branch); (c) successful replan with a finite lidar return placed on the bearing of the current target waypoint (segment blocked) → follower SWAPPED. Give exact poses/frames per branch. |
| TC38 | `dwa` traffic-on drive via runner (subprocess) | Integration | Exit 0; runs to a terminal state; every trace line is the 8-key schema (goal NOT required) |
| TC39 | `apf` traffic-on drive via runner | Integration | Exit 0; runs to completion; 8-key trace |
| TC40 | `rrt_once --no-traffic` on arena_v1 | Integration | `time_to_goal` not null AND ≤ the AC5-recorded margin (not merely < 120 s); two same-seed runs byte-identical |
| TC41 | `rrt_star_once --no-traffic` on arena_v1 | Integration | Reaches goal at the AC5-recorded margin (blocking). Separately, in-process, log `rrt_planned_cost(rrt_star)` vs `rrt_planned_cost(rrt)` as a NON-blocking observation (AC7-obs) — no `≤` assertion |
| TC42 | `rrt_once` & `rrt_star_once` on arena_no_path | Integration | Non-null `planner_error`; no `trace.jsonl` written |
| TC43 | Registration + `--replan-k` validation for all 6 new keys | Unit | `dwa/apf/rrt_once/rrt_star_once` reject `-k`; `rrt_replan/rrt_star_replan` require it; `name==key`; labels fold `_k<K>`; all in `ALGORITHMS` |
| TC44 | `rrt_replan` & `rrt_star_replan` traffic-on via runner | Integration | Write to `rrt_replan_k5/` & `rrt_star_replan_k5/`; 8-key traces; exit 0 |
| TC45 | Commitment-horizon fix proof (binding gate) | Integration | `a_star_replan` and `dijkstra_replan` reach the goal on arena_v1 `--no-traffic`; additionally assert the follower object identity is preserved across ≥1 replan tick on the clear run (proves commitment actually happened, not just that the goal was reached) |

**Test data:** reuse `arena/arena_v1.yaml` (e2e) and `arena/arena_no_path.yaml` (sealed
start). Integration TCs subprocess-invoke `python -m runners.run_episode` into a
`TemporaryDirectory`, mirroring TC14/TC30/TC35.
**Run command:** `python arena/arena.py arena/arena_v1.yaml --check`
**Runtime note:** the `--check` suite is already ~30 min (dominated by full-episode subprocess
TCs). TC38/TC39/TC44 (traffic-on episodes) + TC40/TC41 (RRT solves) add several more
full-episode subprocess runs; T8 must update the CLAUDE.md runtime estimate accordingly.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Commitment-horizon port + `_plan` hook + REPLAN_FAMILIES | — | high | `planners/_grid.py` | Refactor `PathFollowingController`: add `_last_fold`/`_planned_path`; store the fold in `compute_path`; extract the search into an overridable `_plan(folded_grid, folded, state)`; rewrite `act()` so a successful K-th replan swaps the follower only when `is_finished or _immediate_segment_blocked` (fold-free, reuses `_last_fold`). Add `rrt_replan`, `rrt_star_replan` to `REPLAN_FAMILIES`. Implement the convergence rationale exactly as written. Satisfies AC8–AC10, AC3 (membership). Do NOT touch `DStarLiteController`. |
| T2 | DWA controller | — | med | `planners/dwa.py` | Hand-roll `DWAController` (key `dwa`): `reset()` stores goal (`load_world().goal`) + lidar geom (`load_lidar_geometry`), never raises; `act()` samples a dynamic window of `(v,ω)`, forward-simulates a short rollout per candidate, scores goal-heading + lidar clearance + speed, returns the best clamped `(2,1)`. `register("dwa", DWAController)`. Module-level tunable constants. Satisfies AC1. |
| T3 | APF controller | — | med | `planners/apf.py` | Hand-roll `APFController` (key `apf`): Khatib attractive (to YAML goal) + repulsive (live lidar returns within an influence radius) → net force → clamped `(2,1)`. `reset()` never raises. `register("apf", APFController)`. Satisfies AC2. |
| T4 | RRT core + once/replan controllers | T1 | high | `planners/rrt.py` | Hand-roll `rrt_plan` (per-plan `default_rng`, goal bias, step steer, `segment_is_clear_grid` edge checks, cap → raise, start/goal-blocked → raise), `rrt_points_to_waypoints` (replicating the candidate-span bisection), `rrt_planned_cost`. `RRTOnceController` (static-grid plan, mirrors `AStarOnceController`) + `RRTReplanController(PathFollowingController)` overriding `_plan` per the spec body. Register both. **Empirically tune `RRT_SEED` (and raise `RRT_MAX_ITERS`/adjust `RRT_STEP` if needed) against `arena_v1 --no-traffic` until `rrt_once` and `rrt_star_once` reach the goal with sim-time < 120 s; record the chosen seed + measured sim-times in the constants block and AC5.** Satisfies AC3–AC6, AC7-obs (RRT parts). |
| T5 | RRT\* core + once/replan controllers | T1, T4 | high | `planners/rrt_star.py` | Reuse `rrt.py` helpers; add choose-parent + rewire. `RRTStarOnceController` + `RRTStarReplanController(PathFollowingController)` overriding `_plan`. Register both. Satisfies AC3, AC4, AC5, AC7 (goal-reaching) + AC7-obs (planned-cost observation). |
| T6 | Register modules in package init | T2, T3, T4, T5 | low | `planners/__init__.py` | Import the four new modules for their registration side-effect (`# noqa: F401`), mirroring the existing controller imports; do NOT add the controller classes to `__all__` (the established convention exports only the registry helpers). Satisfies AC3 (import-time population). |
| T7 | TCs (TC38–TC45) + TC32 rewrite + suite registration | T1, T6 | high | `arena/arena.py` | Add TC38–TC45 per the Testing Strategy table; rewrite TC32 for the three-branch commitment semantics; register all in `_run_checks`; verify TC30/TC31/TC35–TC37 still pass; update the `--check` PASS-count comment. Satisfies AC4–AC10, AC13. |
| T8 | Docs | T7 | low | `CLAUDE.md`, `Mission.md` | CLAUDE.md "The planner family (Phase 6)": document DWA/APF/RRT/RRT\*, flip the commitment-horizon "Known limitation" to resolved (note the two-call-site duplication tracked in the issue), list TC38–TC45 + the new `--check` count/runtime. Mission.md Phase 6: mark reactive + sampling families landed. Follow the user's global git/prose rules (no AI attribution, avoid tell-words/em-dashes) for any text that may reach commits. Satisfies AC15. |
| T9 | File GitHub issue (after fix verified) | T7 | low | (no repo files; `gh issue create`) | Create an issue covering (a) a stronger follower — alternatives B (cost-hysteresis), C (index-transplant), D (pure-pursuit), why the commitment horizon was chosen, and that any replacement must be applied uniformly incl. D\* Lite to keep the comparison fair; and (b) consolidating the two commitment-horizon copies (`PathFollowingController` + `DStarLiteController`) into one shared helper. Write the issue body in the user's git/prose style (no AI attribution, no tell-words/em-dashes). Satisfies AC14. |

Parallel waves: **W1** = T1, T2, T3. **W2** = T4 (after T1). **W3** = T5 (after T4).
**W4** = T6. **W5** = T7. **W6** = T8, T9 (both after T7).

## Notes for Implementer

- **Non-goal: per-tick obstacle reaction.** Unlike `DStarLiteController` (which folds and
  segment-checks every tick), `PathFollowingController` is a periodic replanner: the fold and
  the commitment check both run only on every K-th act, so obstacle-reaction latency is
  bounded by K ticks. The commitment horizon here only suppresses redundant follower rebuilds
  between replans; it does not add per-tick reactivity. That property belongs to
  `d_star_lite`; the `_replan` families trade it for the fixed K cadence.
- **Fold-free segment check (AC9).** `_immediate_segment_blocked` reuses the stored
  `self._last_fold` and calls only `segment_is_clear_grid` — it must NOT call
  `lidar_to_occupancy`. (TC31 counts folds via the module-level `lidar_to_occupancy` and
  asserts exactly one per replan; using the stored fold keeps that count intact and avoids a
  redundant second fold per replan tick.)
- **TC32 WILL break without the rewrite.** Its current tail (≈ lines 1797–1802) asserts a
  successful replan *always* builds a new follower. Under the commitment horizon a successful
  replan with a clear segment KEEPS the follower. Rewrite to the three concrete branches in
  the Testing table; for the "blocked" branch, place a finite lidar return on the bearing of
  the current target waypoint so the freshly-stored `_last_fold` marks the immediate segment
  blocked.
- **`compute_path` must set `self._last_fold` on every call**, including the initial plan in
  `reset()`, so the commitment check is valid from the first replan act.
- **RRT continuous-vs-cell mismatch:** `grid_path_to_waypoints` expects a *cell* path; RRT
  produces continuous points. Use `rrt_points_to_waypoints`, which drives `_append_clear_waypoints`
  over candidate spans (seed `[points[0]]`, build the stride candidate-index list, call per
  span) — do NOT feed RRT points to `grid_path_to_waypoints`, and do NOT call
  `_append_clear_waypoints` once on the whole list (it would skip the stride downsampling).
- **`_once` plans on the STATIC grid** (no lidar fold), exactly like `AStarOnceController`, so
  AC5 is computed on the same substrate A\* uses; `_replan` plans on the fold via the
  inherited `_plan` seam (override body given above).
- **Per-plan RNG:** `_once` uses `default_rng(RRT_SEED)`; `_replan` uses
  `default_rng(RRT_SEED + self._k)` so replans vary yet stay byte-deterministic (AC4). A
  fixed-every-call seed would make replans re-derive the same path and never adapt.
- **Reactive planners reject `--replan-k`** automatically (not in `REPLAN_FAMILIES`); do not
  add per-controller validation.
- **No runner changes.** `run_episode`/`run_experiment` are generic; if you find yourself
  editing them, stop — the only registry-side change is the `REPLAN_FAMILIES` edit in T1. The
  AC7-obs planned-cost comparison is computed inside the test by calling the planners
  directly, NOT by adding a runner metric.
- **AGENTS.md** is untracked (`?? AGENTS.md`); leave it untracked and out of scope.
- **Rollback:** every task is an additive new file except T1 (base), T6/T7/T8 (small edits);
  reverting T1 + T6 + T7 restores the prior shipped behavior. The GitHub issue (T9) is the
  only outward-facing action — its text lives in this spec for review before filing.
