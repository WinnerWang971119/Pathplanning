# Kalman Multi-Object Tracker for the Predictive Planners — Plan

**Goal:** Eliminate `d_star_lite_predictive`'s ghost detours (the robot rerouting around 400–921-cell predicted stamps while the nearest real obstacle is 1.6–3.0 m away) by replacing `LidarTracker`'s frame-differencing velocity estimator with a deterministic Kalman-filter multi-object tracker (MOT).

**Approach:** Rework `LidarTracker` (`planners/_predict.py`) **in place** into a CV-KF MOT while keeping the existing grid-bucket 8-connected clustering as the per-frame *detector*. The new estimator carries persistent tracks across frames and adds four first-class mechanisms the frame-differencer structurally lacked: (1) a hand-rolled decoupled-axis constant-velocity Kalman filter (no `np.linalg`); (2) **prediction-centered** gated, globally-sorted greedy data association (replacing the loose gate-on-last-centroid that admitted ~10 m/s implied velocities); (3) an N-hit/M-miss track lifecycle with **tentative withholding** and **coast-through-merge**; (4) capped-EMA radius with a merge-suspect gate. Every lidar consumer inherits it (`d_star_lite_predictive`, `dwa_predictive`, `dwa_predictive_paper`); `OracleTracker` is untouched. The ghost dies because a teleporting merged/split centroid never enters a track's state (it fails the tight prediction gate and the surviving track coasts), and a 1–2 frame blip dies unconfirmed and unstamped.

This approach was endorsed (no veto) by the Fable approach consult; the Codex consult was unavailable (CLI version error) and skipped.

## Scope

- **In scope:**
  - Rework `LidarTracker` internals into the CV-KF MOT (`planners/_predict.py`), preserving its public seam: `update(*, snapshot, state, lidar, dt) -> list[Track]` returning `Track(id, x, y, vx, vy, radius)` sorted by `id`.
  - Keep the existing detector (`_lidar_to_world_points` → `_drop_static_returns` → `_cluster` → `_build_cluster`) as-is; replace only `_associate_and_build` and the cross-frame state (`_prev_centroids`, `_prev_velocity_histories`).
  - Prediction-gated globally-sorted greedy association with total-order deterministic tie-breaks.
  - N-hit/M-miss lifecycle: tentative tracks are withheld from output; confirmed tracks coast on their prediction for up to M misses, then die; monotonic per-episode birth-counter track ids (reset on construction).
  - Capped-EMA radius with a merge-suspect gate (skip the radius update when a detection's radius jumps > threshold × the filtered radius).
  - Remove the `smoothing_frames` / `max_track_speed` kwargs from `LidarTracker.__init__` and all construction sites; repurpose the speed bound as an **input innovation/association gate constant** inside the tracker.
  - Update the two consumer construction sites (`planners/dwa_predictive.py`, `planners/d_star_lite_predictive.py`) to the new (kwarg-free) construction.
  - Rewrite the fixture-semantics tests (TC64, TCh) for the new confirmation-lag behavior; verify the run-vs-run determinism tests (TC63, TC58, TCb) still pass unchanged in form; add new in-process unit tests for the KF/association/lifecycle/radius and a ghost-elimination assertion.
  - Update `CLAUDE.md` (the `LidarTracker` / predict-family sections) and write a findings doc.
  - Validation runs: the 100-seed acceptance gate for `d_star_lite_predictive` and the 3-DWA-key × 100-seed re-validation.

- **Out of scope:**
  - Detector-level prediction-conditioned cluster splitting (Fable's suggestion 6) — deferred to a v2; the arena's merges are 1–3 frame blips that coast+withhold already cover. Only build it if a post-fix read still shows merge-driven ghosts.
  - Any change to `OracleTracker`, the oracle predict keys (`d_star_lite_oracle`, `dwa_predictive_oracle`, `dwa_predictive_paper_oracle`), or the `predict_blocked_cells` / `trajectory_conflict` stamp/rollout geometry — the tracker output contract is unchanged, so these consumers are untouched by design.
  - The D* Lite search core, the settle/peel machinery, the commitment horizon, and the cone/capsule stamping geometry — all unchanged.
  - Tuning the prediction horizon, area cap, or cone growth — orthogonal levers, not this fix.

## Decisions

- **Replace `LidarTracker` in place (not a new class):** the same estimator rot degrades the canonical `dwa_predictive` (a spurious ~5 m/s velocity fed to `trajectory_conflict` manufactures phantom matched-time conflicts the braking/soft-yield act on), and its band-aids (`smoothing_frames`, `max_track_speed`) are exactly the "clamp the spike after it happened" approach the diagnosis measured as insufficient. A parallel class would strand the canonical DWA key on a known-bad estimator and force maintaining two lidar estimators. — Endorsed by the Fable consult.
- **Hand-rolled decoupled-axis CV Kalman (no `np.linalg`):** axis-symmetric noise makes the 4-state filter decouple into two identical 2-state filters sharing one *data-independent* covariance/gain recursion (`P`, `S`, `K` depend only on track age + miss pattern, never on measurements). Scalar/2×2 math keeps byte-determinism auditable by eye, matching this repo's convention (cf. the `_segment_clear_fast` hand-roll). At steady state it *is* an α-β filter, so it supplies the innovation covariance the association gate needs for free.
- **Ghost elimination is ~70% MOT scaffolding, ~30% filter:** the prediction-tight gate, coast-through-merge, confirmation withholding, and radius cap are first-class requirements, not the filter alone. A KF-only build would still be yanked by a merged centroid and would NOT fix the bug.
- **Remove the hardening kwargs, don't deprecate:** smoothing is subsumed by the KF gain; the speed cap is repurposed as an input association/innovation gate (reject a detection whose jump from the predicted position exceeds a physical bound) rather than an output clamp. Both DWA constructors and `d_star_lite_predictive._make_tracker` converge on one kwarg-free construction.
- **Track ids become a monotonic per-episode birth counter** (reset on tracker construction), replacing the per-frame geometric `rep_cell` hash. Safe: `_Cluster`'s own docstring states nothing relies on cross-frame id stability, and counter ids are strictly better for `ThreatKey` tie-breaks (soonest-TTC, then id).
- **Withholding does not blind either consumer:** a tentative (unconfirmed) or coasted-out obstacle is still fully covered reactively — D* Lite's plain lidar fold marks its current cells and DWA's present-position safety floor is unchanged. Withholding delays only the *predictive cone* by ~0.2–0.3 s (≤0.6 m at the 2.0 m/s cap).
- **DWA re-validation set = 3 keys × 100 seeds** — the two lidar-consuming predictive DWA keys (`dwa_predictive`, `dwa_predictive_paper`) plus plain `dwa` as the untouched control baseline. *(User to confirm the exact third key at approval; plain `dwa` assumed as the control.)*
- **Primary acceptance gate:** the KF `d_star_lite_predictive` at 100 seeds must beat the current frame-differencing lidar variant on failure rate, and ghost detours must be ~0 on the sampled seeds.

## Acceptance Criteria

- [ ] **AC1 (ghost elimination):** On the diag-style ghost metric (a settle whose committed segment is blocked by a predicted stamp while no real obstacle is within 1.5 m and the un-stamped fold is clear), seeds 0 and 1 — which currently show 2/13 and 6/6 ghost reroutes — drop to 0 ghost reroutes with the KF tracker.
- [ ] **AC2 (byte-determinism preserved):** TC63, TC58, and TCb (run-vs-run, two same-seed runs → byte-identical trace JSONL) still pass unchanged in form. Two fresh KF-`LidarTracker` runs over an identical multi-frame fixture return byte-identical `Track` sequences (`dataclasses.astuple`).
- [ ] **AC3 (no `np.linalg`):** the KF core uses only scalar / 2×2 arithmetic (grep shows no `np.linalg` / matrix-inverse call in the tracker path).
- [ ] **AC4 (tentative withholding):** a synthetic 2-frame spurious cluster (appears, then gone) never appears in `update()`'s output (confirmation N ≥ 3 not reached).
- [ ] **AC5 (coast-through-merge):** in a synthetic two-obstacle merge fixture, neither surviving track's reported velocity spikes above a physical bound (e.g. ≤ 2.5 m/s) across the merge/split; the pre-merge velocities are preserved within tolerance.
- [ ] **AC6 (radius cap):** no reported `Track.radius` exceeds the configured cap (`R_MAX`), even across a merge that balloons the raw cluster radius to ~0.86 m.
- [ ] **AC7 (contract preserved):** `update(*, snapshot, state, lidar, dt)` still returns `list[Track]` sorted by `id`; the first `update` on a fresh tracker yields no *confirmed* tracks (cold start), and all existing non-tracker predict machinery is byte-unaffected (TC57/TC58 oracle guarantees hold).
- [ ] **AC8 (kwargs removed):** `smoothing_frames` / `max_track_speed` are gone from `LidarTracker.__init__`; `dwa_predictive.py` and `d_star_lite_predictive.py` construct the tracker without them; `--check` passes.
- [ ] **AC9 (primary gate):** `d_star_lite_predictive` (KF) 100-seed traffic failure rate on `arena_v1` is strictly lower than the current frame-differencing lidar variant's 100-seed failure rate.
- [ ] **AC10 (DWA no-regression):** the 3-key × 100-seed DWA re-validation shows no failure-rate regression for `dwa_predictive` / `dwa_predictive_paper` vs their current (pre-change) numbers; plain `dwa` (control) is unchanged.
- [ ] **AC11 (full check passes):** `python arena/arena.py arena/arena_v1.yaml --check` passes (all TCs, including the rewritten TC64/TCh and the new tracker unit tests).

## Contracts & Interfaces

Single source of truth for every seam. The tracker's public seam is **unchanged**; only its internals and construction kwargs change.

### Public seam (unchanged — owner T1, consumers T2)

- `class LidarTracker` — `Tracker` Protocol implementation.
- `LidarTracker.update(*, snapshot: object, state: np.ndarray, lidar: np.ndarray, dt: float) -> list[Track]` — returns **confirmed** tracks only, sorted by `id` ascending. `snapshot` ignored (lidar-only). First call on a fresh instance returns `[]` (no track reaches confirmation in one frame).
- `Track` frozen dataclass `(id: int, x: float, y: float, vx: float, vy: float, radius: float)` — **unchanged**. `id` is now a monotonic per-episode birth counter.
- The `Tracker` Protocol (`update(*, snapshot, state, lidar, dt) -> list[Track]`) is unchanged, so `OracleTracker` and every downstream (`predict_blocked_cells`, `trajectory_conflict`, the DWA rollout) bind to the same shape.

### New constructor signature (owner T1, consumers T2)

- `LidarTracker.__init__(self, grid, bearings, range_max=inf)` — the `smoothing_frames` and `max_track_speed` keyword-only params are **removed**. `d_star_lite_predictive._make_tracker` and `dwa_predictive._make_tracker` both call this 3-arg form.

### New module constants (owner T1, in `planners/_predict.py`)

Pin exact names so tests and docs reference one definition:

- `KF_PROCESS_NOISE: float` — CV process-noise intensity `q` (accel white-noise).
- `KF_MEASUREMENT_NOISE: float` — centroid measurement variance `r`.
- `KF_INITIAL_VELOCITY_VARIANCE: float` — birth covariance on the velocity states.
- `ASSOCIATION_GATE_DISTANCE: float` — max centroid-to-**predicted-position** distance for a valid association (physics-tight; replaces `MAX_ASSOCIATION_DISTANCE = 1.0`). Derived from a max-plausible per-frame displacement bound.
- `CONFIRM_HITS: int` (= 3) — consecutive gated hits to promote tentative → confirmed.
- `COAST_MISSES: int` (= 3) — consecutive misses a confirmed track coasts before deletion.
- `RADIUS_EMA_BETA: float` (≈ 0.3) — radius EMA weight.
- `RADIUS_MAX: float` (≈ 0.45) — hard radius cap (1.5× the arena's true 0.3 m mover).
- `RADIUS_MERGE_SUSPECT_RATIO: float` (≈ 1.5) — a detection radius jump above this × the filtered radius is treated as a merge; the radius update is skipped.
- `MIN_TRACK_RADIUS` — retained (existing floor).
- `MAX_ASSOCIATION_DISTANCE`, `CLUSTER_ID_MULTIPLIER`, `VELOCITY_SMOOTHING_FRAMES`, `MAX_TRACK_SPEED` — **removed** (or `MAX_TRACK_SPEED` repurposed into the gate-distance derivation; if kept it is renamed to reflect the input-gate role, e.g. `MAX_PLAUSIBLE_SPEED`).

### Internal track state (owner T1, private — not a cross-task seam)

- `_KTrack` — internal per-track record: birth-counter `id`, KF state `[x, y, vx, vy]`, decoupled covariance scalars, `radius` (filtered), `hits`, `misses`, `confirmed: bool`. Not exported; `update` maps confirmed `_KTrack`s → `Track`.

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `planners/_predict.py` | T1 | T2, T3, T4 |
| `planners/dwa_predictive.py` | T2 | — |
| `planners/d_star_lite_predictive.py` | T2 | — |
| `arena/arena.py` | T3 | — |
| `CLAUDE.md`, `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | T4 | — |

## Data Model

The decoupled CV-KF (per track, per axis — x and y share one recursion):

```
State (per axis):   s = [pos, vel]
Transition:         F = [[1, dt], [0, 1]]
Process noise Q(q): [[q*dt^3/3, q*dt^2/2], [q*dt^2/2, q*dt]]
Measurement:        H = [1, 0]  (centroid position);  R = r (scalar)
Predict:  s <- F s ;  P <- F P F^T + Q
Gate:     |z - H s_pred| <= ASSOCIATION_GATE_DISTANCE  (per detection, on predicted pos)
Update:   y = z - H s ;  S = P00 + r ;  K = [P00/S, P10/S]
          s <- s + K y ;  P <- (I - K H) P
```

`P`, `S`, `K` are functions of age + miss pattern only (data-independent) — precompute-friendly, and the whole update is scalar mul/add. Radius is a separate scalar EMA, not part of the KF state.

Lifecycle state machine per track:

```
detection unassociated ─────────────────▶ TENTATIVE (hits=1, withheld)
TENTATIVE + gated hit ×(CONFIRM_HITS-1) ─▶ CONFIRMED (emitted)
TENTATIVE + miss ───────────────────────▶ DELETED
CONFIRMED + gated hit ──────────────────▶ CONFIRMED (misses:=0, KF update)
CONFIRMED + miss (<COAST_MISSES) ───────▶ CONFIRMED (coast: emit predicted pos, misses+1)
CONFIRMED + miss ×COAST_MISSES ─────────▶ DELETED
```

## Error Handling

- **Empty / all-NaN lidar frame:** detector returns no clusters → every confirmed track takes a miss (coasts up to `COAST_MISSES`), tentatives die. Deterministic; no exception.
- **Degenerate KF (zero/near-zero innovation variance):** `S = P00 + r` with `r > 0` is always positive, so no division guard needed; assert `r > 0` at construction.
- **First frame (no priors):** all detections birth tentative tracks; `update` returns `[]`. Matches AC7 cold-start.
- **Association ties (equal float distances under symmetric geometry):** the sort key is `(distance, track_id, detection_rank)` — a total order over the two int fields, so ties never resolve nondeterministically (the load-bearing determinism landmine).
- **Obstacle despawn at the perimeter:** the track coasts up to `COAST_MISSES` (≤0.3 s) emitting a bounded predicted stamp at the world edge, then deletes — usually rejected by the corridor gate anyway. Acceptable, documented.
- **Input validation:** keep the existing `update` shape checks (state `(3,)`, lidar `(bearings,)`, `dt > 0`); add `grid`-not-None and `r > 0` construction checks.

## Testing Strategy

**Levels:** Unit (in-process, no irsim — the tracker is pure logic), Integration (subprocess e2e through the runner), plus offline validation runs.

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC64 (rewrite) | KF tracker determinism across a multi-frame cluster-count change | Unit | Two fresh trackers over the same ≥5-frame fixture return byte-identical `Track` sequences; the +x mover, once **confirmed** (after `CONFIRM_HITS`), reports `vx>0` and `|vy|` small. Confirmation lag replaces the old frame-2 velocity assertion. |
| TCh (rewrite) | Default construction determinism + no hardening kwargs | Unit | `LidarTracker(grid, bearings)` constructs (no kwargs), is deterministic across a cluster-count change, and reported speed never exceeds the physical bound. The old "enabled hardening" sub-cases are removed. |
| TC-K1 (new) | Prediction-gated association tie determinism | Unit | A symmetric two-detection / two-track frame associates identically across two runs; swapping input order does not change id assignment (total-order tie-break). |
| TC-K2 (new) | Tentative withholding | Unit | A cluster present for only 2 frames (< `CONFIRM_HITS`) never appears in any `update()` output. |
| TC-K3 (new) | Coast-through-merge, no spike | Unit | Two obstacles approach, merge for 1–2 frames, split. Neither surviving track's reported `|v|` exceeds ~2.5 m/s across the event; post-split velocities match pre-merge within tolerance. |
| TC-K4 (new) | Radius cap + merge-suspect | Unit | Across a merge that balloons the raw cluster radius to ~0.86 m, no reported `Track.radius` exceeds `RADIUS_MAX`. |
| TC-K5 (new) | Ghost elimination (in-process) | Unit/Integration | Driving `d_star_lite_predictive` on seed 0 (or 1) for ~120 ticks, the ghost-reroute count (settle blocked only by a stamp, no real obstacle within 1.5 m) is 0 (currently 2 / 6). |
| TC63 (verify) | Predictive traffic-on e2e determinism | Integration | Unchanged in form: two same-seed traffic runs → byte-identical trace JSONL, 8-key schema, plans at t=0. |
| TC58 / TCb (verify) | Oracle + DWA predictive e2e determinism | Integration | Unchanged: oracle keys byte-stable (OracleTracker untouched); DWA keys terminal + deterministic (trajectories change, determinism holds). |
| VAL1 (offline) | Primary 100-seed gate | Validation | `d_star_lite_predictive` (KF) 100-seed failure rate < current lidar-variant 100-seed failure rate (AC9). |
| VAL2 (offline) | DWA 3×100 re-validation | Validation | `dwa_predictive` / `dwa_predictive_paper` no failure-rate regression; plain `dwa` unchanged (AC10). |

**Test data:** synthetic lidar fixtures built in-process (reuse TC64's fixture-builder pattern — a grid + bearings + hand-placed range returns walking a cluster count). Ghost metric reuses the diag instrumentation (wrap `_settle_and_extract` + un-stamped fold check). No new irsim dependency for the unit tests.

**Run command:** `python arena/arena.py arena/arena_v1.yaml --check` (unit + integration TCs). Validation runs: `python -m runners.run_experiment --algorithm d_star_lite_predictive --predict-horizon 10 --world arena/arena_v1.yaml --num-seeds 100` (and the current-variant baseline captured before the change), plus the 3 DWA keys at `--num-seeds 100`.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|-----------|------|-------|-------------|
| T0 | Capture pre-change baselines | — | low | (results only) | BEFORE any code change, record the current frame-differencing lidar variant's numbers so AC9/AC10 have a baseline: run `d_star_lite_predictive`, `dwa_predictive`, `dwa_predictive_paper` at `--num-seeds 100` (traffic on, arena_v1) and save the failure rates. Also snapshot the seed-0/seed-1 ghost counts (2 and 6). Satisfies the baseline half of AC9/AC10. May be a long run — can be backgrounded. |
| T1 | Rework `LidarTracker` into the CV-KF MOT | — | xHigh | `planners/_predict.py` | Replace `_associate_and_build` + the `_prev_centroids`/`_prev_velocity_histories` state with: the hand-rolled decoupled CV-KF (Data Model above, no `np.linalg`), the `_KTrack` internal state, prediction-gated globally-sorted greedy association (sort key `(distance, track_id, detection_rank)`), the N-hit/M-miss lifecycle with tentative withholding + coast-through-merge, monotonic birth-counter ids (reset on construction), and capped-EMA radius + merge-suspect gate. Remove `smoothing_frames`/`max_track_speed` from `__init__`; add the new module constants (Contracts). Keep the detector (`_lidar_to_world_points`/`_drop_static_returns`/`_cluster`/`_build_cluster`) and the `update` signature + `Track` output contract unchanged. Deterministic: no RNG, no set-iteration in output/order, sorted iteration, fixed-order reductions. Satisfies AC2–AC7. |
| T2 | Update consumer construction sites | T1 | med | `planners/dwa_predictive.py`, `planners/d_star_lite_predictive.py` | Change both `_make_tracker()` calls to the kwarg-free `LidarTracker(grid, bearings, range_max=...)` form (drop `smoothing_frames`/`max_track_speed` and the now-removed `VELOCITY_SMOOTHING_FRAMES`/`MAX_TRACK_SPEED` imports). Confirm `OracleTracker` construction is untouched. Satisfies AC8. |
| T3 | Rewrite fixture tests + add tracker unit tests | T1, T2 | med | `arena/arena.py` | Rewrite TC64 and TCh for the confirmation-lag semantics (velocity asserted after `CONFIRM_HITS`, no hardening kwargs). Add TC-K1..TC-K5 (Testing Strategy). Verify TC63/TC58/TCb still pass unchanged in form. Wire the new TCs into the `--check` registry/counters and update the TC-count banner. Satisfies AC2, AC4–AC7, AC11. |
| T4 | Docs | T1, T2 | low | `CLAUDE.md`, `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | Update the `LidarTracker` / predict-family sections of `CLAUDE.md` (KF MOT, lifecycle, removed kwargs, new TC list, id semantics). Scaffold the findings doc (design + the VAL1/VAL2 results table to be filled by T5). Keep README/`--check` banner counts current. |
| T5 | Validation runs + findings verdict | T1, T2, T3, T0 | low | `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | Run VAL1 (100-seed `d_star_lite_predictive` KF vs T0 baseline) and VAL2 (3 DWA keys × 100). Record failure rates + the seed-0/1 ghost recheck in the findings doc; state whether AC9/AC10 pass. Long-running — likely user-launched or backgrounded. |

## Notes for Implementer

- **The determinism landmine is the association sort key.** Symmetric arena geometry produces exactly-equal float gate distances; the sort key MUST be `(distance, track_id, detection_rank)` where `detection_rank` is the detection's index in `rep_cell`-sorted order, so ties resolve by a total order over ints. A bare `(distance,)` sort leaks nondeterministic order and breaks TC63/TC64. This is the single highest-risk line in T1.
- **No `np.linalg`.** The decoupled 2-state filter's `S` is a scalar and `K` is a 2-vector; the whole update is scalar mul/add. A matrix inverse would both be overkill and put a float path you can't audit into the determinism-critical core (AC3).
- **`update` still returns confirmed tracks only.** Coasting emits a confirmed track's *predicted* position each miss frame (deterministically), never a frozen last-seen position — else the D* stamp jitters. Tentative tracks are never emitted.
- **Birth counter resets on construction, not on `update`.** Each episode builds a fresh tracker via `_make_tracker`, so the per-episode counter reset is automatic — mirrors the `id_iter` reset-on-make gotcha. Do not add a per-`update` reset.
- **OracleTracker is the control arm — do not touch it.** TC57/TC58's oracle byte-guarantees must hold. The whole change lives on the lidar path.
- **The two hardening kwargs are removed, not deprecated.** Grep the repo for `smoothing_frames` / `max_track_speed` / `VELOCITY_SMOOTHING_FRAMES` / `MAX_TRACK_SPEED` and clean up every reference (construction sites + TCh + CLAUDE.md). If `MAX_TRACK_SPEED`'s value is reused for the gate distance, rename it to reflect the input-gate role.
- **Capture T0 baselines BEFORE editing code** — once the tracker changes, the "current lidar variant" number is unrecoverable without a git stash/checkout, so run T0 first.
- **Fixture-semantics tests will fail until rewritten** — TC64 asserts a frame-2 velocity that the confirmation lag (N=3) no longer produces; TCh asserts the removed hardening kwargs. Rewrite them in the same change (T3), do not weaken them.
- **Rollback plan:** the change is contained to `planners/_predict.py` internals + two one-line construction sites + tests; if VAL1 regresses, `git revert` restores the frame-differencer with no cross-module entanglement (the `Track`/`update` seam never changed).
- **Deferred (v2):** detector-level prediction-conditioned cluster splitting (Fable suggestion 6). Only revisit if the post-fix quick read still shows merge-driven ghosts.
