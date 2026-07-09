# Predictive DWA (space-time collision avoidance) Plan

**Goal:** Give the arena a planner that does *real* prediction — reasoning in
`(x, y, t)` about where crossing traffic will be, not painting predicted
footprints onto a 2-D grid (the "stamping" the D\* Lite predictive family does).
Extend the existing Dynamic Window Approach so that, inside its forward-simulation
rollout, it also advances each tracked obstacle at constant velocity and checks
the robot against the obstacle **at the matched time step**, then scores and
prunes candidates on that space-time conflict.

**Approach:** A new predictive DWA controller family (`planners/dwa_predictive.py`)
that subclasses `DWAController` and adds a **two-layer** collision model:

1. **Present-position safety floor** — vanilla DWA's clearance check against the
   full live lidar cloud, unchanged, hard-rejecting any candidate whose rollout
   grazes a currently-visible obstacle (robust to a tracker miss).
2. **Space-time predictive layer** — for each tracked obstacle and each rollout
   step `k`, compare the robot's predicted pose `traj[k]` to the obstacle's
   constant-velocity predicted position `(x + vx·k·dt, y + vy·k·dt)`; hard-reject
   candidates that collide within the horizon and add a *predicted-clearance*
   term to the score so the robot yields early and smoothly.

This directly realizes the two cited DWA-prediction papers (Missura & Bennewitz,
ICRA 2019, *Predictive Collision Avoidance for the Dynamic Window Approach*; and
MDPI *Actuators* 2025 14(5):207, which folds a linear obstacle-prediction model
into the DWA rollout cost). It reuses the substrate the D\* Lite predictive family
already built: the `Track` / `Tracker` / `OracleTracker` / `LidarTracker` records
in `planners/_predict.py`, the truth seam (`Controller.wants_truth` /
`observe_truth`, `EpisodeInfo.dynamic_obstacles`), and the whole
`--predict-horizon` / `PREDICT_FAMILIES` / `_h<steps>` CLI+registry machinery. The
genuinely new code is a small pure `trajectory_conflict(...)` helper plus the
controller; the runner needs **zero** changes (it is already generic over
`PREDICT_FAMILIES` via `build_controller` + `algorithm_label` + the `wants_truth`
gate + the render overlay).

Two velocity sources behind the existing `Tracker` seam, mirroring D\* Lite:
`dwa_predictive` (lidar frame-differencing estimator, **Mission-faithful,
promoted to canonical**) and `dwa_predictive_oracle` (perfect live velocities, a
deliberate cheat that measures the achievable ceiling, **experimental**).

---

## Scope

- **In scope:**
  - A pure, in-process-testable `trajectory_conflict(...)` helper in
    `planners/_predict.py` (space-time robot-vs-track conflict + earliest
    time-to-collision + minimum predicted gap).
  - A tiny byte-preserving refactor of `DWAController`: an overridable
    `_evaluate_candidate(...)` hook around the per-candidate clearance+score, and
    a `self._rollout_steps` field so the rollout can be lengthened for the
    space-time check without changing the base score.
  - `planners/dwa_predictive.py`: `PredictiveDWAController` base + two keys —
    `dwa_predictive` (lidar, `wants_truth=False`, `LidarTracker`) and
    `dwa_predictive_oracle` (oracle, `wants_truth=True`, `OracleTracker`).
  - Registry wiring: both keys in `PREDICT_FAMILIES`; `dwa_predictive_oracle` in
    `EXPERIMENTAL_KEYS`; `dwa_predictive` promoted to canonical.
  - **Canonical promotion of `dwa_predictive`**: add it to `run_all._CANONICAL_ORDER`,
    `plot.py CANONICAL`, and the `_seed_filter` canonical mirror + predict-fold set;
    grow the `filter_seeds` "required" roster to include the second experimental
    oracle label (the `all13` set becomes an `all15` set: 13 canonical + 2 oracles).
  - Sweep tooling: add both keys to `run_horizon_sweep` / `plot_horizon_sweep`
    `SWEEP_ALGORITHMS`; add `dwa_predictive` to `run_speed_sweep`'s focus set
    (issue #11: does true prediction floor the crash rate as obstacle speed rises?).
  - New `--check` TCs: baseline parity (`dwa_predictive_oracle_h0` ≡ plain `dwa`),
    `--predict-horizon` validation, traffic-on e2e + determinism per variant, and a
    pure-helper unit test.
  - A quick-read measurement (h0 vs h10, first 10 canonical seeds, traffic on,
    arena_v1) + a findings markdown; the full 50-seed × {0,5,10,20} sweep stays the
    user's to launch.
  - Docs: CLAUDE.md section, README planner list, Mission Phase note.
  - Render overlay reuse: the controller exposes `last_predicted_cells` (empty for
    DWA — it stamps nothing) and `last_tracks`, so the existing
    `Arena.draw_prediction` paints per-track velocity arrows in `--render` mode.
- **Out of scope:**
  - The Missura exact collision-cone / linear-program core (a full replacement of
    DWA's sampling grid) — considered and rejected as an overkill rewrite that
    would jeopardize determinism; the two-layer sampling approach is faithful to
    the MDPI paper and keeps the byte-identical trace guarantee.
  - Pure space-time avoidance that *drops* the present-position floor for movers
    (closest to Missura) — rejected: a tracker miss becomes an invisible obstacle
    and a crash. The floor is kept.
  - A velocity clamp / behaviour-prediction model on the lidar estimator (v1 uses
    the existing `LidarTracker` unchanged); estimator refinement is future work.
  - Tuning the DWA weights beyond adding one predicted-clearance weight.
  - The full 50-seed canonical sweep + verdict (user launches it; quick read gates).

---

## Decisions

- **Two-layer collision model (present floor + space-time layer), not pure
  space-time.** The floor is standard reactive avoidance (not prediction); the
  space-time layer is the real `(x,y,t)` prediction. Keeping the floor makes the
  controller robust to a tracker miss (an untracked mover still blocks via its
  live returns) at negligible cost to the "slip-behind" maneuver (the floor only
  forbids driving into an obstacle's *current* body, which is always correct).
- **Oracle checks walls the same way as the lidar variant** (live lidar cloud),
  so the ONLY difference between the two keys is the velocity source — the oracle
  is a clean ceiling for exactly velocity estimation (user-confirmed).
- **Horizon extends the rollout, base score unchanged.** `--predict-horizon H`
  sets the space-time depth. The robot rollout runs `max(ROLLOUT_STEPS, H)` steps;
  heading/clearance/speed scoring uses only the first `ROLLOUT_STEPS` (a forward
  prefix, byte-identical to today), so the h20 sweep point is meaningful and
  `h0` ≡ plain `dwa`.
- **`h0` is the built-in baseline.** At horizon 0 the predictive `act()` delegates
  to `super().act()` (vanilla DWA), so `dwa_predictive_oracle_h0` /
  `dwa_predictive_h0` produce a byte-identical trace to plain `dwa` (TC65).
- **`dwa_predictive` is canonical, `dwa_predictive_oracle` is experimental** —
  mirrors the current D\* Lite state (`d_star_lite_predictive` canonical,
  `d_star_lite_oracle` experimental). The lidar variant lands on the headline
  scatter alongside its peers regardless of how it scores (user-confirmed); the
  oracle stays a ceiling measurement carved out of the canonical set.
- **Reuse everything from the D\* Lite predictive substrate.** `Track`, the two
  trackers, the truth seam, `--predict-horizon`, `PREDICT_FAMILIES`, the render
  overlay — all already exist; this plan is mostly a new controller + wiring.
- **Determinism preserved.** DWA is already deterministic (fixed sampling grid,
  no RNG); both trackers are deterministic; the new helper is pure. Traces stay
  byte-identical across same-seed runs, the same invariant the harness guarantees.

---

## Acceptance Criteria

- [ ] **AC1 (quick-read gate):** On arena_v1 over the first 10 canonical seeds,
  traffic on, `dwa_predictive_oracle` at h10 achieves a crash+timeout failure rate
  **lower** than plain `dwa` (= h0), reported with the oracle-vs-lidar table. A
  non-improvement is reported honestly (the DWA memory already says DWA halves
  hazard, so the oracle is expected to help; a null result is still a valid,
  reported finding).
- [ ] **AC2:** `dwa_predictive_oracle_h0` and `dwa_predictive_h0` each produce a
  **byte-identical** `trace.jsonl` to plain `dwa` on the same seed
  (`--no-traffic` and traffic-on), proving zero-horizon is a true no-op baseline.
- [ ] **AC3:** Both variants run a full traffic-on episode through the runner to a
  terminal state, emitting the 8-key trace schema per line; two same-seed runs are
  byte-identical.
- [ ] **AC4:** `trajectory_conflict(...)` is a pure function whose output is
  byte-identical across runs on fixed inputs and correct: it reports a collision
  exactly when the robot body overlaps a track body at a matched step within the
  horizon, the earliest such step, and the minimum matched-time gap.
- [ ] **AC5:** `act()` never raises mid-episode for either variant (a tracker /
  prediction failure degrades that tick to plain DWA); only a t=0 `reset()` failure
  surfaces as `planner_error` (DWA `reset()` never raises, so `planner_error` is
  always null for this family).
- [ ] **AC6:** `--predict-horizon` is **required** for `dwa_predictive` /
  `dwa_predictive_oracle` and **rejected** (exit 2) for every other algorithm;
  `--replan-k` is rejected (exit 2) for both; results land in
  `<world_stem>/dwa_predictive[_oracle]_h<steps>/`.
- [ ] **AC7:** `run_all` imports clean with `dwa_predictive` canonical and
  `dwa_predictive_oracle` experimental (`set(_CANONICAL_ORDER) == set(ALGORITHMS) -
  EXPERIMENTAL_KEYS` holds); the main plot charts the 13 canonical planners.
- [ ] **AC8:** The `--check` suite grows from 68 cases; every existing case stays
  green and the new cases pass. `filter_seeds --selfcheck` and
  `plot_horizon_sweep --selfcheck` stay green.

---

## Data Model

```python
# planners/_predict.py — additive pure helper (Track already exists)
@dataclass(frozen=True)
class TrajectoryConflict:
    collides: bool          # robot body overlaps any track body at a matched step within horizon
    ttc_step: int | None    # earliest colliding step k (1-based), else None
    min_gap: float          # min over checked steps/tracks of (center_dist - robot_r - track_r); +inf if no tracks
```

## Contracts & Interfaces

### Signatures

- `trajectory_conflict(robot_positions: np.ndarray, tracks: list[Track], robot_radius: float, horizon_steps: int, dt: float, margin: float) -> TrajectoryConflict`
  — owner: `planners/_predict.py`; consumer: `PredictiveDWAController`. `robot_positions`
  is `(S, 2)` world positions at steps `k = 1..S` (step k is `dt·k` ahead). Checks
  `k = 1..min(horizon_steps, S)`. Obstacle position at step k is
  `(track.x + track.vx·k·dt, track.y + track.vy·k·dt)`. `collides` iff any matched-step
  gap `dist - robot_radius - track.radius <= margin`. Pure/deterministic.
- `DWAController._evaluate_candidate(self, state, trajectory, v, obstacle_points) -> float | None`
  — new overridable hook. Base returns `None` when the present-position clearance
  rejects, else the base weighted score (byte-identical to today's inline body).
  Owner: `planners/dwa.py`; overridden by `PredictiveDWAController`.
- `DWAController._rollout(self, state, v, w) -> np.ndarray` now produces
  `self._rollout_steps` positions (default `ROLLOUT_STEPS`); the predictive base sets
  it to `max(ROLLOUT_STEPS, H)`.
- `PredictiveDWAController.last_predicted_cells: list = []` and `.last_tracks: list = []`
  (read-only debug attrs, init `[]`, refreshed each `act()`) — consumed by the
  render overlay. `last_predicted_cells` is always `[]` for DWA (no grid stamp).
- Construction: `build_controller(name, replan_k, predict_horizon)` already passes
  `(replan_k=, predict_horizon=)` to `PREDICT_FAMILIES` members, so the DWA predictive
  `__init__(self, replan_k=None, predict_horizon=None)` slots in unchanged.

### Naming (exact strings)

- Registry keys: `"dwa_predictive"`, `"dwa_predictive_oracle"`.
- `PREDICT_FAMILIES = {..., "dwa_predictive", "dwa_predictive_oracle"}`;
  `EXPERIMENTAL_KEYS = {"d_star_lite_oracle", "dwa_predictive_oracle"}`.
- Label suffix `_h<steps>`; canonical horizon `PREDICT_HORIZON = 10`.
- `run_all._CANONICAL_ORDER` gains `"dwa_predictive"` (13 canonical).
- `_seed_filter._CANONICAL_ORDER` mirror gains `"dwa_predictive"`;
  `_PREDICT_CANONICAL` gains `"dwa_predictive"`; the required-labels experimental
  tail appends both `d_star_lite_oracle_h<H>` and `dwa_predictive_oracle_h<H>`.
- `SWEEP_ALGORITHMS` (both sweep modules) gains `"dwa_predictive"`,
  `"dwa_predictive_oracle"`.

### File ownership

| File | Change |
|------|--------|
| `planners/_predict.py` | add `TrajectoryConflict` + `trajectory_conflict` (pure) |
| `planners/dwa.py` | extract `_evaluate_candidate` hook + `self._rollout_steps` (byte-identical) |
| `planners/dwa_predictive.py` (new) | base + oracle + lidar controllers; register both |
| `planners/_grid.py` | `PREDICT_FAMILIES` / `EXPERIMENTAL_KEYS` += DWA keys |
| `planners/__init__.py` | import the new module (registers the keys) |
| `runners/run_all.py` | `_CANONICAL_ORDER += dwa_predictive` |
| `runners/plot.py` | `CANONICAL += dwa_predictive` row |
| `runners/_seed_filter.py` | canonical mirror + predict-fold + required-labels tail |
| `runners/filter_seeds.py` | `--planners` choice rename + help |
| `runners/run_horizon_sweep.py`, `runners/plot_horizon_sweep.py` | `SWEEP_ALGORITHMS` += DWA keys |
| `runners/run_speed_sweep.py` | focus set += `dwa_predictive` |
| `arena/arena.py` | new TCs + count/help update |
| `docs/plans/2026-07-10-predictive-dwa.findings.md` (new) | quick-read table |
| `CLAUDE.md`, `README.md`, `Mission.md` | docs |

## Error Handling

- **Tracker / prediction failure mid-episode:** the predictive `act()` wraps the
  space-time layer so any failure degrades the tick to plain DWA (`super().act()`),
  never raising (AC5).
- **All candidates rejected (present floor + space-time):** vanilla DWA's in-place
  rotation fallback (toward the clearer lidar side), unchanged (user-confirmed).
- **`observe_truth` not called (traffic off / non-oracle):** the tracker sees an
  empty snapshot / no dynamic returns → zero tracks → no space-time layer → the
  controller behaves as plain DWA.
- **Malformed `--predict-horizon`:** rejected at `build_controller` / parse time
  with exit 2, before any Arena is built (the existing machinery).

## Testing Strategy

**Levels:** in-process unit (pure helper), subprocess integration (runner e2e),
determinism (byte-identical traces). Mirrors TC53/TC57/TC59/TC63.

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC65 | `dwa_predictive_oracle_h0` & `dwa_predictive_h0` ≡ plain `dwa` trace | Integration | Byte-identical `trace.jsonl` (`--no-traffic` and traffic-on), same seed (AC2) |
| TC66 | `--predict-horizon` validation for the DWA family | Integration | Required for both DWA predict keys (omit → exit 2), rejected for `dwa` (exit 2), `--replan-k` rejected (exit 2); label folds `_h<steps>`; rejected runs write no `<seed>.json` (AC6) |
| TC67 | `dwa_predictive` (lidar) & `dwa_predictive_oracle` traffic-on e2e + determinism | Integration | Each runs to a terminal state, 8-key trace per line, two same-seed runs byte-identical (AC3) |
| TC68 | `trajectory_conflict` pure geometry | Unit | A head-on track within horizon collides with correct `ttc_step`; a receding / out-of-horizon track does not; `min_gap` correct; byte-identical across two calls (AC4) |
| TC69 | `run_all` canonical assertion tolerant + set = 13 | Unit | `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS`; `dwa_predictive` in canonical, `dwa_predictive_oracle` experimental; import exits 0 (AC7) |

**Run command:** `& .venv\Scripts\python.exe arena/arena.py arena/arena_v1.yaml --check`
(grows from 68 to 73 cases), plus `python -m runners.filter_seeds --selfcheck` and
`python -m runners.plot_horizon_sweep --selfcheck`.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Pure `trajectory_conflict` | — | med | `planners/_predict.py` | Add `TrajectoryConflict` + the pure space-time helper. |
| T2 | DWA hook refactor | — | med | `planners/dwa.py` | Extract `_evaluate_candidate`; parameterize `_rollout` by `self._rollout_steps`. Byte-identical for `dwa`. |
| T3 | Predictive DWA controllers | T1, T2 | high | `planners/dwa_predictive.py`, `planners/_grid.py`, `planners/__init__.py` | Base + oracle + lidar; two-layer; build grid+tracker in reset; observe_truth; h0→super; register; `PREDICT_FAMILIES`/`EXPERIMENTAL_KEYS`. |
| T4 | Canonical + tooling wiring | T3 | med | `run_all.py`, `plot.py`, `_seed_filter.py`, `filter_seeds.py`, sweeps | Promote `dwa_predictive`; grow rosters; extend `SWEEP_ALGORITHMS`; speed-sweep focus. |
| T5 | Check-suite TCs | T3 | med | `arena/arena.py` | TC65–TC69; update count + help. |
| T6 | Quick-read + findings | T3, T5 | low | run + `docs/plans/2026-07-10-predictive-dwa.findings.md` | h0 vs h10, 10 seeds; oracle-vs-lidar table. |
| T7 | Docs + commit | T4, T5, T6 | low | `CLAUDE.md`, `README.md`, `Mission.md` | Document the family; commit on the feature branch. |

## Notes for Implementer

- **Determinism landmine is the same one D\* Lite hit:** the `LidarTracker` must
  stay byte-reproducible (it already is — TC64). The space-time helper is pure. DWA
  itself has no RNG. Do not introduce set-iteration or RNG in the controller.
- **h0 must be a true no-op:** at horizon 0 the predictive `act()` returns
  `super().act(state, lidar)` directly — no tracker call, no rollout extension — so
  the trace equals plain `dwa` byte-for-byte (TC65). Keep `observe_truth` side-effect-free
  w.r.t. the trajectory.
- **The base score must stay on the first `ROLLOUT_STEPS`:** even when the rollout is
  extended to H>12, pass `trajectory[:ROLLOUT_STEPS]` to `_score` and the present-floor
  clearance, and `trajectory[:H]` to the space-time check.
- **Lidar is center-to-surface** (`gotcha-lidar-center-relative`): the space-time band
  is `robot_radius + track.radius + margin`; the present floor already adds the body radius.
- **Use the venv python** (`gotcha-use-venv-python-not-path-python`): run everything via
  `& .venv\Scripts\python.exe -m ...`; the `--check` subprocess TCs inherit it via `sys.executable`.
- **Rollback:** every change is additive (new files + additive registry entries + a
  byte-preserving refactor + canonical roster growth). Reverting the import in
  `planners/__init__.py` and the roster edits restores prior behaviour.
