# Predictive (motion-aware) D* Lite Plan

**Goal:** Stop D\* Lite from crashing into crossing traffic by making it anticipate
obstacle *motion* — estimate each obstacle's velocity and stamp its predicted
future footprint into the occupancy grid so the planner routes *behind* moving
obstacles instead of into where they are about to be. Build it oracle-first
(perfect velocities) as a go/no-go gate, then a lidar-only estimator.

**Approach:** A new experimental D\* Lite controller family that injects a
**predictive stamp** into the per-tick lidar fold *before* the diff, so the
prediction enters purely as changed occupancy cells through D\* Lite's existing
`update_cells` seam — the incremental search core is untouched. The prediction
geometry, the velocity source, and the safety gating are factored into a pure,
in-process-testable substrate (`planners/_predict.py`) so the whole thing gets
fast `--check` TCs (like TC46/TC47) instead of slow irsim subprocess tests. Two
velocity sources behind one `Tracker` seam: an **oracle** (true live velocities,
a deliberate cheat to measure the achievable ceiling) and a **lidar estimator**
(frame-differencing, Mission-faithful). A `--predict-horizon` knob sweeps the
lookahead T (T=0 ≡ plain D\* Lite, the built-in baseline).

This plan was shaped by a brainstorm (`docs/brainstorms/2026-06-27-moving-obstacle-aware-planning.md`)
and a two-model approach consult (Fable + Codex), both of which independently
endorsed the stamping seam and converged on the capsule-vs-cone and
predicted-conflict-gate decisions recorded below.

---

## Scope

- **In scope:**
  - A shared predictive substrate `planners/_predict.py`: a `Track` record, a
    `Tracker` protocol, `OracleTracker`, `LidarTracker`, and the pure
    `predict_blocked_cells(...)` geometry/gate/peel logic (capsule + widening cone).
  - A `PredictiveDStarLiteController` base subclassing `DStarLiteController`,
    plus two registry keys: `d_star_lite_oracle` (oracle-fed, capsule) and
    `d_star_lite_predictive` (lidar-fed, widening cone). Both **experimental**
    (excluded from the canonical-11 set), both **reject** `--replan-k`.
  - The live-truth seam: an additive `EpisodeInfo.dynamic_obstacles` field
    (Arena), a `Controller.wants_truth` flag + `observe_truth(snapshot)` setter
    (protocol), and the runner calling `observe_truth` before `act()` for opt-in
    controllers only.
  - A `--predict-horizon <int steps>` CLI knob on `run_episode` / `run_experiment`
    (required for the predictive family, rejected elsewhere), folded into the
    results label as `_h<steps>`.
  - A horizon-sweep driver (`runners/run_horizon_sweep.py`) and a headless
    read-only plotter (`runners/plot_horizon_sweep.py`) writing
    `failure_rate_vs_horizon.png`, `median_time_vs_horizon.png`, and
    `horizon_sweep_summary.csv`, plus a findings markdown.
  - New `--check` TCs covering the pure predictor (geometry, determinism, gate,
    peel), the baseline-parity (`_h0` ≡ plain `d_star_lite`), the truth seam, the
    `--predict-horizon` validation, and a traffic-on e2e per variant.
  - A **render-only** prediction overlay: in `--render` mode, draw the predicted
    capsule/cone footprint as a translucent overlay plus each tracked obstacle's
    velocity arrow on irsim's matplotlib axes. Strictly read-only w.r.t.
    state/lidar/action/trace; gated entirely behind `render=True`.
  - Documentation updates (CLAUDE.md, Mission.md Phase 7 note).
- **Out of scope:**
  - Promoting either variant to a **canonical** (12th) planner on the headline
    Mission scatter — deferred, trivial follow-up once a variant proves out.
  - Generalizing the stamp to the A\*/Dijkstra/RRT `_replan` families — D\* Lite
    is the flagship; the `_predict.py` substrate is written reusable for a later
    port, but no other controller is touched.
  - A plausibility/speed clamp on the lidar estimator (user deselected it for v1;
    closing — see Notes — relies on the gate + fail-open instead).
  - A space-time (x, y, t) D\* Lite, a VO/RVO velocity-space layer, or an MPC/APF
    action-space override (all considered in the consult and rejected for this
    plan; see Decisions).
  - Tuning the obstacle-speed regime — orthogonal to this plan (issue #11).

---

## Decisions

- **Stamp enters via the existing fold→diff→`update_cells` seam, search core untouched.**
  Both consult models endorsed this as the correct layering; it preserves D\* Lite's
  incremental invariants and determinism (TC46-style: batched `update_cells` + one
  deferred settle == from-scratch A\* on the same folded grid).
- **Geometry: capsule for the oracle, widening cone for the lidar estimator.**
  Obstacles move in exact straight lines at constant velocity (`dynamic._advance()`
  is `x += vx*dt`, never draws `motion_rng`), so with *perfect* velocity the future
  footprint is a straight tube — a widening cone would over-block headings the
  obstacle provably never takes and fight the fail-open guard. The widening
  *represents uncertainty*, which is zero for the oracle and nonzero (estimator
  error) for the lidar variant. Both models converged here; user confirmed.
- **Threat gate: predicted-conflict (geometry), not instantaneous closing-course.**
  An obstacle diverging from the robot's segment *now* can still intersect it
  0.6 s later (shallow-angle fast crosser); an instantaneous-heading gate drops
  exactly that class. Gate instead on whether an obstacle's predicted capsule over
  [0, T] geometrically intersects a corridor around the robot's current planned
  path. Both models flagged the heading gate; user confirmed.
- **Truth seam: opt-in `observe_truth(snapshot)` setter, not an `act()` kwarg.**
  Keeps `act(state, lidar)` uniform across all 11 existing controllers (the
  `Controller` protocol stays honest); only the oracle sets `wants_truth=True`.
  Same opt-in concept the user chose, in a cleaner mechanical form (Fable).
- **Velocity source behind a `Tracker` protocol; prediction is a pure function.**
  `OracleTracker`/`LidarTracker` both return a deterministic, id/cell-sorted
  `list[Track]`; `predict_blocked_cells(...)` is pure (plain floats in, sorted
  cells out, no irsim) so it is unit-tested in-process (Fable #5).
- **Fail-open peel is threat-ordered and bounded; a per-tick area budget caps the
  stamp.** Stamp gated cells; if the deferred settle finds no path to goal, drop
  predicted groups *farthest-future first* (least-imminent) until a path re-exists,
  so the most imminent protection is kept and the robot is never trapped. The
  predicted-conflict gate + robot exclusion zone bound how much gets stamped, and a
  per-tick **stamped-cell area cap** (allocate to soonest-TTC tracks first, skip the
  rest — Thinker #13) hard-bounds both the peel frequency and the per-tick cost.
  Caveat (Fable): the "peel fires rarely" expectation is NOT proven a priori — under
  the matched/fast speed regimes the union of ~20 capsules at T=2.0 s could
  disconnect the arena and make the peel fire often; **T10 measures the actual peel
  frequency**, and if it is high the area cap / a shorter best-horizon is the lever.
- **Oracle is the go/no-go gate with a human checkpoint.** The estimator tasks are
  blocked on an explicit "report oracle sweep to the user" milestone; the user
  greenlights (or kills) the estimator after seeing the ceiling. User chose to
  build everything but keep this checkpoint.
- **Both new keys are experimental.** `run_all`'s canonical-set import assertion is
  relaxed to `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS`, so the
  new keys run through the runner/sweep tooling without landing on the canonical-11
  scatter.
- **Horizon as integer steps, T = steps × dt (dt = 0.1 s).** Sweep set
  `{0, 5, 10, 20}` steps ≡ `{0, 0.5, 1.0, 2.0}` s. `_h0` stamps nothing and is the
  built-in baseline/ablation; a TC pins `d_star_lite_oracle_h0`'s trace to equal
  plain `d_star_lite`.
- **Render overlay is render-only and read-only.** The controller exposes its last
  predicted cells + tracks as read-only debug attributes; the Arena draws them on
  the irsim matplotlib axes ONLY when `render=True`. The draw path never touches
  the step/trace pipeline, so headless determinism (and the `_h0` trace parity) is
  unaffected. User opted in (cells + velocity arrows).

---

## Acceptance Criteria

- [ ] **AC1 (go/no-go — the user's bar):** On `arena/arena_v1.yaml` over the
  canonical 50 seeds, `d_star_lite_oracle` at its best swept **non-zero** horizon
  (the min over `{h5, h10, h20}` — `h0` is EXCLUDED because it is plain
  `d_star_lite` by construction, AC2, so including it makes the comparison
  vacuously true) achieves a crash+timeout failure rate **at least `MARGIN` lower**
  than plain `d_star_lite`, where `MARGIN` ≥ 2 seeds (≥ 0.04 over 50) so a
  one-seed flutter is not counted as a win. If NO non-zero horizon clears that
  margin, the result is a **refutation**: reported to the user with the sweep data,
  and the estimator tasks (T11–T14) do not proceed without a user decision.
- [ ] **AC2:** `d_star_lite_oracle_h0` produces a **byte-identical** `trace.jsonl`
  to plain `d_star_lite` on the same seed (`--no-traffic` and traffic-on), proving
  zero-horizon stamping is a true no-op baseline.
- [ ] **AC3:** Both variants run a full traffic-on episode through the runner to a
  terminal state, emitting the **8-key** trace schema per line (incl.
  `dynamic_obstacles_sha256`); two same-seed runs are byte-identical.
- [ ] **AC4:** `predict_blocked_cells(...)` is a pure function whose output (a
  sorted, deduped `list[(row, col)]`) is byte-identical across runs on fixed
  inputs, and is correct per geometry: capsule = constant-radius disk train along
  v; cone = radius growing linearly with lookahead step.
- [ ] **AC5:** The robot exclusion zone guarantees no predicted cell within radius
  `R` of the robot's current cell is ever stamped; the fail-open peel guarantees a
  stamp can never leave the grid unsolvable (the robot keeps a path every tick).
- [ ] **AC6:** `act()` never raises mid-episode for either variant; only a t=0
  `reset()` failure surfaces as `planner_error`.
- [ ] **AC7:** `--predict-horizon` is **required** for `d_star_lite_oracle` /
  `d_star_lite_predictive` and **rejected** (exit 2) for every other algorithm;
  `--replan-k` is rejected (exit 2) for the predictive family. Results land in
  `<world_stem>/d_star_lite_oracle_h<steps>/`.
- [ ] **AC8:** `run_all` still asserts canonical-set integrity but tolerates the
  two experimental keys; importing `planners` + `run_all` exits 0.
- [ ] **AC9:** `run_horizon_sweep` runs the oracle across `{0,5,10,20}` steps on a
  world and `plot_horizon_sweep` renders the two PNGs + CSV; `--selfcheck` passes
  on synthetic fixtures (no irsim).
- [ ] **AC10:** The existing 55-case `--check` suite stays green; new TCs are
  added and pass; `EpisodeInfo.dynamic_obstacles` is `()` when traffic is off or
  pre-reset.
- [ ] **AC11:** In `--render`, the predicted capsule/cone footprint and per-track
  velocity arrows are drawn as a translucent overlay that refreshes each tick (no
  accumulation); with `render=False` the draw path is never entered and the trace
  is byte-identical to a run built without the overlay code (determinism guard).

---

## Data Model

```python
# planners/_predict.py  (planner-side, decoupled from arena.dynamic)
@dataclass(frozen=True)
class Track:
    id: int          # stable identity (oracle: obstacle id; lidar: synthesized cluster id)
    x: float
    y: float
    vx: float        # m/s, world frame
    vy: float
    radius: float

# arena/arena.py — additive EpisodeInfo field
#   dynamic_obstacles: tuple[DynamicObstacleState, ...]
#     The live post-_advance() snapshot this tick (same tick as dynamic_obstacles_sha256).
#     () when traffic is off or pre-reset. Existing fields unchanged.
```

## Contracts & Interfaces

Single source of truth for every cross-task seam.

### Shared types

- `Track` (frozen dataclass above) — owner: T3 (`planners/_predict.py`); consumers:
  T4, T5, T11, T12.
- `EpisodeInfo.dynamic_obstacles: tuple[DynamicObstacleState, ...]` — owner: T1
  (`arena/arena.py`); consumers: T2 (runner reads it), and the oracle via the seam.

### Signatures

- `class Tracker(Protocol): def update(self, *, snapshot, state, lidar, dt) -> list[Track]`
  — owner: T3. `OracleTracker.update` reads `snapshot` (ignores lidar);
  `LidarTracker.update` reads `state, lidar` (ignores snapshot). Returns tracks
  sorted by `id`. Consumers: T5 (oracle), T12 (lidar).
- `predict_blocked_cells(tracks: list[Track], planned_path: list[np.ndarray], robot_xy: np.ndarray, grid: OccupancyGrid, inflation: float, horizon_steps: int, dt: float, *, geometry: str, exclusion_radius: float, corridor_half_width: float) -> list[tuple[ThreatKey, list[tuple[int,int]]]]`
  — owner: T4 (`planners/_predict.py`). `geometry ∈ {"capsule","cone"}`. Returns
  per-track `(threat_key, cells)` groups in **threat order** (ascending
  time-to-conflict, then `id`), each `cells` sorted row-major, with the robot
  exclusion zone already removed and only gate-passing tracks present. Consumers:
  T5 (controller unions + bounded peel). **Stamp radius:** the disk at each future
  center `(x + vx·k·dt, y + vy·k·dt)` (k = 1..horizon_steps) has radius
  `track.radius + inflation`, where `inflation = robot_radius + SAFETY_MARGIN` —
  i.e. the SAME body-aware band the static grid uses, so a stamped cell is
  collision-equivalent (the lidar is center-to-surface). The function OWNS the
  world→grid conversion of each future center via `world_to_grid` (imported from
  `manual_astar`, like the rest of `_predict.py`).
- `Controller.wants_truth: bool = False` and
  `Controller.observe_truth(self, snapshot: tuple) -> None` (default no-op) —
  owner: T2 (`planners/_types.py`); the oracle overrides both. Consumer: T2
  (runner call site), T5 (oracle controller).
- `build_controller(name, replan_k, predict_horizon=None)` and
  `algorithm_label(name, replan_k, predict_horizon=None)` — the new third
  parameter MUST be defaulted to `None` so the ~40 existing two-arg call sites
  (in `arena/arena.py` TCs, `runners/plot.py`, `run_all.py`, `run_experiment.py`,
  `run_speed_sweep.py`, `plot_speed_sweep.py`) keep working untouched; `algorithm_label`
  folds `_h<steps>` only for the `PREDICT_FAMILIES`. Owner: T5 (`planners/_grid.py`);
  consumers: runner, run_all, run_experiment, plotters. **Do not edit the existing
  call sites** (a non-defaulted positional would break the 55-case suite at import —
  AC8/AC10).
- `PredictiveDStarLiteController.last_predicted_cells: list[tuple[int,int]]` and
  `.last_tracks: list[Track]` (read-only debug attrs, **initialized to `[]` in
  `__init__`, never `None`**, refreshed each `act()`) — owner: T5; consumer: T16
  (render overlay, which must tolerate the empty pre-first-act case). `Arena.draw_prediction(cells, tracks) -> None`
  (no-op unless `render=True`) — owner: T16 (`arena/arena.py`); consumer: T16
  (runner render path).

### Naming (exact strings that must match across tasks)

- Registry keys: `"d_star_lite_oracle"`, `"d_star_lite_predictive"`.
- `PREDICT_FAMILIES = frozenset({"d_star_lite_oracle", "d_star_lite_predictive"})`
  and `EXPERIMENTAL_KEYS = PREDICT_FAMILIES` in `planners/_grid.py`.
- CLI flag `--predict-horizon` (int steps); label suffix `_h<steps>`.
- Horizon sweep step set `(0, 5, 10, 20)`; `_manifest.json` provenance key
  `predict_horizon`.
- `PREDICT_DT = 0.1` (module constant in `planners/_predict.py`; matches irsim
  `step_time` / DWA `CONTROL_DT`).

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `arena/arena.py` (EpisodeInfo + populate) | T1 | T9 → T16 → T13 (TC + render, chained: T1→T9→T16→T13) |
| `planners/d_star_lite.py` (extract stamp hook, byte-identical) | T5 | T12 (subclass) |
| `planners/_types.py` | T2 | T5 |
| `runners/run_episode.py` | T2 (seam) → T5 (`--predict-horizon`) | T7 |
| `planners/_predict.py` | T3 → T4 → T11 (chained) | T5, T12 |
| `planners/d_star_lite_predictive.py` | T5 → T12 (chained) | — |
| `planners/_grid.py` | T5 | T6, T7 |
| `planners/__init__.py` | T5 → T12 (chained) | — |
| `runners/run_all.py` | T6 | — |
| `runners/run_experiment.py` | T7 | T8 |
| `runners/run_horizon_sweep.py` (new) | T8 → T13 (lidar key) | T10, T14 |
| `runners/plot_horizon_sweep.py` (new) | T8 | T10, T14 |

## Error Handling

- **Prediction makes the grid unsolvable:** the threat-ordered bounded peel drops
  farthest-future predicted groups until a path re-exists; if even zero prediction
  is unsolvable (a real static dead-end), the existing D\* Lite swallow keeps the
  last valid follower — `act()` never raises.
- **Bad/garbage lidar velocity estimate (no clamp in v1):** contained by the
  predicted-conflict gate (a wrong-direction tube usually fails the corridor
  intersection) and the fail-open peel (an over-large stamp that seals the map is
  peeled). A persistent blow-up is a Notes-flagged fast-follow (add the clamp).
- **`observe_truth` not called (traffic off / non-oracle):** the oracle tracker
  sees an empty snapshot → zero tracks → zero stamp → behaves as plain D\* Lite.
- **Malformed `--predict-horizon` (negative, or given to a non-predict family, or
  a predict family missing it):** rejected at `build_controller` / parse time with
  exit 2, before any Arena is built (mirrors `--replan-k`).

## Testing Strategy

**Levels:** in-process unit (pure predictor), subprocess integration (runner e2e),
determinism (byte-identical traces).

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC53 | `predict_blocked_cells` capsule geometry on a synthetic track | Unit | Constant-radius disk train along v; cells sorted/deduped; byte-identical across two calls |
| TC54 | `predict_blocked_cells` cone widening + exclusion zone | Unit | Radius grows with step; no cell within R of robot; gate drops a non-intersecting track |
| TC55 | Predicted-conflict gate catches a divergent-now-collide-later crosser | Unit | A shallow-angle fast track whose capsule crosses the planned-path corridor within T is stamped; a receding track is not |
| TC56 | Threat-ordered bounded peel | Unit | A map-sealing stamp is peeled farthest-future-first until a path exists; most-imminent group retained |
| TC56b | Peel-to-zero still unsolvable (negative path) | Unit | When even zero prediction leaves no path (a real dead-end), the controller keeps its last valid follower and `act()` does NOT raise (AC5/AC6) |
| TC57 | `d_star_lite_oracle_h0` ≡ plain `d_star_lite` trace | Integration | Byte-identical `trace.jsonl` (`--no-traffic` and traffic-on), same seed |
| TC58 | `d_star_lite_oracle` traffic-on e2e + determinism | Integration | Runs to terminal state, 8-key trace per line, two same-seed runs byte-identical |
| TC59 | `--predict-horizon` validation | Integration | Required for predict family, rejected for others (exit 2); `--replan-k` rejected for predict family (exit 2); label dir `_h<steps>` |
| TC60 | `EpisodeInfo.dynamic_obstacles` + `observe_truth` seam + tick alignment | Unit | `()` when traffic off/pre-reset; the snapshot the oracle observes is the SAME tick as the `state` its `act()` receives (assert the observed snapshot equals the `info.dynamic_obstacles` from the same `step()` that produced `state`); non-oracle controllers never have `observe_truth` called |
| TC61 | `run_all` canonical assertion tolerates experimental keys | Unit | `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS`; import of `planners` + `run_all` exits 0 |
| TC62 | `plot_horizon_sweep --selfcheck` | Unit | Synthetic-fixture cases pass with no irsim; PNGs + CSV render |
| TC63 (T13) | `d_star_lite_predictive` (lidar) traffic-on e2e + determinism | Integration | Runs to terminal state, 8-key trace, two same-seed runs byte-identical |
| TC64 (T13) | `LidarTracker` determinism across a MULTI-frame fixture with a cluster-count change | Unit | Over ≥4 synthetic frames where the number of clusters changes between frames (an obstacle enters/leaves, mirroring TC52's refill), the velocity estimates and the cluster/association ordering are byte-identical across two runs — exercises the association-stability hazard, not just a single 2-frame diff |

**Test data:** synthetic `Track` lists and 2-frame lidar fixtures built in-process
(no irsim), mirroring TC46/TC47. E2e TCs use `arena/arena_v1.yaml` and the seeded
runner subprocess.
**Run command:** `python arena/arena.py arena/arena_v1.yaml --check` (grows from 55
to ~64 cases) and `python -m runners.plot_horizon_sweep --selfcheck`.
**Render overlay (T16):** verified manually via
`python -m runners.run_episode --algorithm d_star_lite_oracle --predict-horizon 10 --seed 42 --world arena/arena_v1.yaml --render`
(watch the capsule + arrows track the obstacles). Its determinism safety is covered
by the existing headless TC58 (two same-seed headless runs byte-identical) — the
overlay path is never entered when `render=False`, so it cannot perturb the trace.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Arena live-snapshot field | — | med | `arena/arena.py` | Append `dynamic_obstacles: tuple[DynamicObstacleState,...]` to `EpisodeInfo` as the LAST field (after `dynamic_obstacles_sha256`, so existing positional order is preserved) and add it to the **`EXPECTED_EPISODE_INFO_FIELDS`** constant (the real symbol name — NOT `_INFO_FIELDS`) that TC2 asserts against, updating TC2's expected list. Populate from the post-`_advance()` `_last_snapshot` in `reset()` and `step()` (same tick as `dynamic_obstacles_sha256`); `()` when traffic off/pre-reset. Keep it OUT of `_trace_line` (a tuple in the trace would break byte-identity). Satisfies AC10, feeds the truth seam. |
| T2 | Truth seam (protocol + runner) | T1 | med | `planners/_types.py`, `runners/run_episode.py` | Add `wants_truth: bool = False` + `observe_truth(self, snapshot) -> None` (no-op) to the `Controller` protocol; in the runner loop call `controller.observe_truth(snapshot)` BEFORE `act(state, lidar)` only when `controller.wants_truth`, where `snapshot`, `state`, and `lidar` ALL come from the SAME source — `reset()` at t=0 (`state0`, `lidar0`, `info0.dynamic_obstacles`) and the SAME `arena.step(...)` return tuple thereafter (`state`, `lidar`, `info.dynamic_obstacles`). Do NOT reuse an `info` carried from a prior `step()` and do NOT advance the snapshot by a dt — each EpisodeInfo's snapshot is already at the same sim tick as the state returned by that same call (judge-confirmed alignment). Leave `act(state,lidar)` for all others. |
| T3 | `Track` + `Tracker` + `OracleTracker` | — | low | `planners/_predict.py` (new) | Frozen `Track`; `Tracker` Protocol; `OracleTracker.update(snapshot,...)` → id-sorted `list[Track]` from `DynamicObstacleState` (convert, decouple from arena). `PREDICT_DT=0.1`. Pure, no irsim. |
| T4 | Pure `predict_blocked_cells` (capsule+cone+gate+peel-order) | T3 | high | `planners/_predict.py` | Implement the geometry (capsule = const-radius disk train; cone = growing radius), the predicted-conflict gate (capsule-vs-planned-path-corridor intersection over [0,T]), the robot exclusion zone, and threat-ordered grouping. Cell-marking mirrors `_mark_disk` deterministically and returns sorted cells. Pure. Satisfies AC4/AC5. |
| T5 | `PredictiveDStarLiteController` + oracle key + CLI/registry plumbing | T2, T4 | high | `planners/d_star_lite.py`, `planners/d_star_lite_predictive.py` (new), `planners/_grid.py`, `planners/__init__.py`, `runners/run_episode.py` | **First refactor `DStarLiteController.act()` to expose an overridable hook** (e.g. `_extra_blocked_cells(state, lidar, folded_new_cells) -> list[(row,col)]`, default returns `[]`) called between the `lidar_to_occupancy` fold and the `diff_mask = self._cells != new_cells` line, OR-ing the returned cells into `new_cells` before the diff. Base returning `[]` keeps `d_star_lite` BYTE-IDENTICAL to today (guard with TC57). Then `PredictiveDStarLiteController` subclasses it and overrides ONLY that hook to call `predict_blocked_cells` (with the bounded fail-open peel); it does NOT reimplement `act()`. `wants_truth=True`, store snapshot in `observe_truth`, set `last_predicted_cells`/`last_tracks`. Add `PREDICT_FAMILIES`/`EXPERIMENTAL_KEYS`, extend `build_controller`/`algorithm_label` with a DEFAULTED `predict_horizon: int | None = None` (existing 2-arg callers untouched) + `_h<steps>` label, add `--predict-horizon` to the runner (required for predict family, reject `--replan-k`). Register `d_star_lite_oracle` (capsule). Satisfies AC2/AC3/AC6/AC7. |
| T6 | Relax `run_all` canonical assertion | T5 | low | `runners/run_all.py` | Change the import-time assertion to `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS`; do NOT add the experimental keys to `_CANONICAL_ORDER`/`canonical_planner_set()`. Satisfies AC8. |
| T7 | `run_experiment` horizon forwarding + manifest | T5 | med | `runners/run_experiment.py` | Add `--predict-horizon`, forward to each child episode, record `predict_horizon` in `_manifest.json`; reuse the `--replan-k` forwarding pattern. |
| T8 | Horizon sweep driver + plotter | T7 | med | `runners/run_horizon_sweep.py` (new), `runners/plot_horizon_sweep.py` (new) | Driver shells `run_experiment` per horizon step in `{0,5,10,20}` for `d_star_lite_oracle`, writing label dirs `d_star_lite_oracle_h<steps>/`. Read-only headless plotter reads those dirs via `plot.load_world_results`, renders `failure_rate_vs_horizon.png` + `median_time_vs_horizon.png` + `horizon_sweep_summary.csv` (x = T seconds = steps×0.1), with `--selfcheck`. Satisfies AC9. |
| T9 | `--check` TCs (predictor + oracle + seam) | T1, T5, T8 | med | `arena/arena.py` | Add TC53–TC62 per the Testing Strategy. Blocked by T1 too (both edit `arena/arena.py`; the file order is T1 → T9 → T16 → T13). Satisfies AC2/AC4/AC5/AC7/AC8/AC10. |
| T10 | **ORACLE CHECKPOINT — run sweep + report to user** | T8, T9 | low | (run + `docs/plans/2026-06-27-predictive-d-star-lite.findings.md`) | Run the oracle horizon sweep on `arena_v1` (50 seeds), generate plots + findings doc, summarize to the user: oracle crash rate vs baseline, margin, best horizon. **Human go/no-go gate** — T11–T14 require user greenlight. Evaluates AC1. |
| T11 | `LidarTracker` (frame-differencing estimator) | T10 | high | `planners/_predict.py` | Per-tick: lidar→world points implemented LOCALLY in `_predict.py` (mirror DWA's `_lidar_to_world_points` exactly — same `np.linspace` bearings — but do NOT import DWA's private method; duplicating ~10 deterministic lines avoids a `dwa.py` refactor that is out of scope), subtract static returns via the static grid, deterministic grid-bucket clustering, sorted greedy nearest-neighbor association vs the stored prior frame, `v=(centroid_now-centroid_prev)/PREDICT_DT`. No clamp (v1). Stable ordering, no RNG/set-iteration. Satisfies AC determinism. |
| T12 | `d_star_lite_predictive` (lidar) controller + key | T11 | high | `planners/d_star_lite_predictive.py`, `planners/__init__.py` | Wire `LidarTracker` + `geometry="cone"` (widening, half-angle from association noise) into the predictive base; `wants_truth=False` (lidar-only). Register `d_star_lite_predictive`. |
| T13 | Estimator `--check` TCs + sweep inclusion | T12, T16 | med | `arena/arena.py`, `runners/run_horizon_sweep.py` | Add TC63–TC64; let the sweep driver also run `d_star_lite_predictive`. Blocked by T16 too because both edit `arena/arena.py` (serialize the shared file). |
| T14 | Lidar sweep + findings update + report | T13 | low | (run + findings doc) | Run the lidar horizon sweep, update the findings doc + plots, report to the user. |
| T15 | Docs | T5 | low | `CLAUDE.md`, `Mission.md` | Document the predictive family, the `--predict-horizon` knob, the experimental-key carve-out, and the horizon sweep; add a Mission Phase 7 note. |
| T16 | Render prediction overlay | T5, T9 | low | `arena/arena.py`, `runners/run_episode.py` | Add `Arena.draw_prediction(cells, tracks)` that, ONLY when `render=True`, paints the predicted footprint (translucent cell overlay) + per-track velocity arrows on irsim's matplotlib axes, clearing the previous tick's artists so they don't accumulate. The runner, only in render mode, reads `controller.last_predicted_cells` / `.last_tracks` after `act()` and calls it. Strictly read-only w.r.t. the step/trace pipeline (AC11). Shared-file serialization: `arena/arena.py` order is T1 → T9 → T16 → T13; `runners/run_episode.py` order is T2 → T5 → T16. |

## Notes for Implementer

- **Determinism is the landmine, and it lives in T11 (the estimator).** The
  frame-differencing tracker introduces mutable per-tick controller state that must
  be byte-reproducible: cluster by sorted cell key, associate in sorted-centroid
  order (first-match-wins), reduce centroids with stable `np.mean` over a
  sorted-by-cell point list — never set iteration, never RNG. Static-return
  subtraction must reuse the exact `np.linspace` bearing recovery
  (`gotcha-lidar-angle-increment-mismatch`), or wall returns ghost as moving
  obstacles. Prototype T11 against synthetic 2-frame fixtures and prove
  byte-identical output BEFORE wiring it into the controller (T12).
- **`_h0` is the contract baseline.** At horizon 0, `predict_blocked_cells` returns
  no cells, so the fold equals the plain fold and the trajectory must match plain
  `d_star_lite` byte-for-byte (TC57). Keep the oracle's `observe_truth`/tracker
  side-effect-free w.r.t. the trajectory at h=0.
- **Stamp BEFORE the diff** (`d_star_lite.py` `act()` ~ the `diff_mask = self._cells != new_cells` line): union predicted cells into the freshly folded array, then diff, then mutate `self._cells` in place + `update_cells` — never rebind `self._cells` (grid-ownership invariant). Reuse `_mark_disk`'s row-major discipline so the trace stays byte-stable.
- **Lidar is center-to-surface** (`gotcha-lidar-center-relative`): the exclusion
  zone, gate corridor, and any clearance must add the robot radius where a body
  collision is implied.
- **The capsule is a lossy 2D projection of the true space-time reservation**
  (Fable): it blocks every cell the obstacle occupies over [0, T] even though the
  robot passes each cell at one specific time, so it over-blocks somewhat. This is
  accepted for the ceiling study (a true (x,y,t) reservation was scoped out); if
  the over-block causes timeouts that muddy AC1, the horizon sweep surfaces it as a
  shorter best-horizon, and T10's report must call it out.
- **The estimator risk does not vanish by choosing stamping** (Fable): velocity
  estimation (T11) is the same hard, determinism-fragile problem the rejected VO
  approach would need. Stamping wins on *layering* (search core untouched), not by
  dodging the perception problem — which is exactly why the oracle gate (T10)
  precedes the estimator.
- **The crash-floor risk is real** (the consult's lead concern): ~1/3 of obstacles
  outrun the robot, so AC1 may be physically capped. T10 exists precisely to learn
  this cheaply (perfect velocities) before the expensive estimator. If the oracle
  cannot beat baseline, STOP at T10 and report — do not build T11–T14 on a proven-
  unwinnable target.
- **Rollback:** every change is additive (new files + additive fields/flags + a
  relaxed assertion). Reverting the two registry imports in `planners/__init__.py`
  and the `EpisodeInfo` field restores the prior behavior; existing planners are
  untouched.
- Duplication note: `PredictiveDStarLiteController` shares the commitment-horizon
  logic already duplicated between `PathFollowingController` and
  `DStarLiteController` (tracked in an existing GitHub issue) — do not widen that
  duplication; subclass `DStarLiteController` and override only the fold/stamp step.
