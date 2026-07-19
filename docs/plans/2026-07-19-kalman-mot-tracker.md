# Kalman Multi-Object Tracker for the Predictive Planners — Plan

**Goal:** Eliminate `d_star_lite_predictive`'s ghost detours (the robot rerouting around 400–921-cell predicted stamps while the nearest real obstacle is 1.6–3.0 m away) by replacing `LidarTracker`'s frame-differencing velocity estimator with a deterministic Kalman-filter multi-object tracker (MOT).

**Approach:** Rework `LidarTracker` (`planners/_predict.py`) **in place** into a CV-KF MOT while keeping the existing grid-bucket 8-connected clustering as the per-frame *detector*. The new estimator carries persistent tracks across frames and adds four first-class mechanisms the frame-differencer structurally lacked: (1) a hand-rolled decoupled-axis constant-velocity Kalman filter (no `np.linalg`); (2) **prediction-centered** gated, globally-sorted greedy data association (replacing the loose gate-on-last-centroid that admitted ~10 m/s implied velocities); (3) an N-hit/M-miss track lifecycle with **tentative withholding** and **coast-through-merge**; (4) capped-EMA radius with a merge-suspect gate. Every lidar consumer inherits it (`d_star_lite_predictive`, `dwa_predictive`, `dwa_predictive_paper`); `OracleTracker` is untouched. The ghost dies because a teleporting merged/split centroid never enters a track's state (it fails the tight prediction gate and the surviving track coasts), and a 1–2 frame blip dies unconfirmed and unstamped.

This approach was endorsed (no veto) by the Fable approach consult; the Codex consult was unavailable (CLI version error) and skipped. The spec was then adversarially reviewed by a Fable critic (2 critical + 11 warning + 7 nit, all accepted and folded in below).

## Scope

- **In scope:**
  - Rework `LidarTracker` internals into the CV-KF MOT (`planners/_predict.py`), preserving its public seam: `update(*, snapshot, state, lidar, dt) -> list[Track]` returning `Track(id, x, y, vx, vy, radius)` sorted by `id`.
  - Keep the existing detector (`_lidar_to_world_points` → `_drop_static_returns` → `_cluster` → `_build_cluster`) as-is; replace only `_associate_and_build` and the cross-frame state (`_prev_centroids`, `_prev_velocity_histories`).
  - Prediction-gated globally-sorted greedy association with total-order deterministic tie-breaks.
  - N-hit/M-miss lifecycle: tentative tracks are withheld from output; confirmed tracks coast on their prediction for up to M misses, then die; monotonic per-episode birth-counter track ids (reset on construction).
  - Capped-EMA radius with a merge-suspect gate (skip the radius update when a detection's radius jumps > threshold × the filtered radius).
  - Remove the `smoothing_frames` / `max_track_speed` kwargs from `LidarTracker.__init__` and all construction sites; the fast-regime speed bound survives as `MAX_PLAUSIBLE_SPEED` feeding the input association-gate distance.
  - Update the one changed consumer construction site (`planners/dwa_predictive.py`); confirm `d_star_lite_predictive.py` needs no change (already kwarg-free).
  - Delete the now-dead frame-differencing prose (module docstring, tunables comment block, `Tracker` Protocol "future LidarTracker … frame-differencing" note).
  - Rewrite the fixture-semantics tests (TC64, TCh) for the new confirmation-lag behavior; audit and adjust every other `--check` TC that constructs, patches, or is timing-coupled to `LidarTracker`; add new in-process unit tests for the KF/association/lifecycle/radius and a ghost-elimination assertion.
  - Update `CLAUDE.md` (the `LidarTracker` / predict-family sections) and write a findings doc.
  - Validation runs: the 100-seed acceptance gate for `d_star_lite_predictive` and the 3-DWA-key × 100-seed re-validation (all at `--jobs 6`, `--predict-horizon 10`).

- **Out of scope:**
  - Detector-level prediction-conditioned cluster splitting (Fable's suggestion 6) — deferred to a v2; the arena's merges are 1–3 frame blips that coast+withhold already cover. Only build it if a post-fix read still shows merge-driven ghosts.
  - Any change to `OracleTracker`, the oracle predict keys (`d_star_lite_oracle`, `dwa_predictive_oracle`, `dwa_predictive_paper_oracle`), or the `predict_blocked_cells` / `trajectory_conflict` stamp/rollout geometry — the tracker output contract is unchanged, so these consumers are untouched by design.
  - The D* Lite search core, the settle/peel machinery, the commitment horizon, and the cone/capsule stamping geometry — all unchanged.
  - Tuning the prediction horizon, area cap, or cone growth — orthogonal levers, not this fix.

## Decisions

- **Replace `LidarTracker` in place (not a new class):** the same estimator rot degrades the canonical `dwa_predictive` (a spurious ~5 m/s velocity fed to `trajectory_conflict` manufactures phantom matched-time conflicts the braking/soft-yield act on), and its band-aids (`smoothing_frames`, `max_track_speed`) are exactly the "clamp the spike after it happened" approach the diagnosis measured as insufficient. A parallel class would strand the canonical DWA key on a known-bad estimator and force maintaining two lidar estimators. — Endorsed by the Fable consult.
- **Hand-rolled decoupled-axis CV Kalman (no `np.linalg`):** axis-symmetric noise makes the 4-state filter decouple into two identical 2-state filters sharing one covariance/gain recursion. `S = P00 + r` is a scalar, so the update is scalar mul/add — byte-determinism auditable by eye, matching this repo's convention (cf. the `_segment_clear_fast` hand-roll). At steady state it *is* an α-β filter, so the "lighter fix" is subsumed.
- **Ghost elimination is ~70% MOT scaffolding, ~30% filter:** the prediction-tight gate, coast-through-merge, confirmation withholding, and radius cap are first-class requirements, not the filter alone. A KF-only build would still be yanked by a merged centroid and would NOT fix the bug.
- **Fixed-Euclidean physics association gate (NOT a Mahalanobis / innovation-covariance gate):** a detection associates to a track iff its centroid is within `ASSOCIATION_GATE_DISTANCE` of the track's **predicted** position `(x + vx·dt, y + vy·dt)`. The gate distance is a physical per-frame-displacement bound, not a covariance-scaled ellipse — simpler, fully deterministic, and it does not silently widen during coast re-acquisition. (This resolves the spec's earlier both-ways framing; the KF's innovation covariance is used only in the filter update, never as the gate radius.)
- **The gate is sized for the `fast` speed regime, not `current`:** `ASSOCIATION_GATE_DISTANCE = MAX_PLAUSIBLE_SPEED · PREDICT_DT + GATE_SLACK`. `MAX_PLAUSIBLE_SPEED = 2.0` (the issue-#11 `fast`-regime cap — deliberately 2.0, NOT 1.5, so fast-regime crossers are not truncated one frame at a time), `PREDICT_DT = 0.1`, `GATE_SLACK = 0.15` (centroid/cluster jitter allowance) ⇒ ≈ 0.35 m. This is ~3× tighter than the old `MAX_ASSOCIATION_DISTANCE = 1.0` (which admitted ~10 m/s implied velocities) yet still passes every physically-real association.
- **Remove the hardening kwargs, don't deprecate:** `smoothing_frames` (subsumed by the KF gain) and `max_track_speed` (subsumed by the input gate) are deleted from `__init__`. The old `MAX_TRACK_SPEED = 2.0` constant survives, renamed `MAX_PLAUSIBLE_SPEED`, feeding the gate distance above; `VELOCITY_SMOOTHING_FRAMES`, `MAX_ASSOCIATION_DISTANCE`, and `CLUSTER_ID_MULTIPLIER` are removed. One kwarg-free construction site for every consumer.
- **Track ids become a monotonic per-episode birth counter** (reset on tracker construction — automatic, since each episode builds a fresh tracker via `_make_tracker`), replacing the per-frame geometric `rep_cell` hash. Safe: `_Cluster`'s own docstring states nothing relies on cross-frame id stability, and counter ids are strictly better for `ThreatKey` tie-breaks.
- **Withholding does not blind either consumer:** a tentative or coasted-out obstacle is still fully covered reactively — D* Lite's plain lidar fold marks its current cells and DWA's present-position safety floor is unchanged. Withholding delays only the *predictive cone* by ~0.2–0.3 s (≤0.6 m at the 2.0 m/s cap).
- **DWA re-validation set = 3 keys × 100 seeds** — the two lidar-consuming predictive DWA keys (`dwa_predictive`, `dwa_predictive_paper`) plus plain `dwa` as the untouched control baseline. *(User to confirm the exact third key at approval; plain `dwa` assumed as the control.)*
- **Primary acceptance gate:** the KF `d_star_lite_predictive` at 100 seeds must beat the current frame-differencing lidar variant on failure rate, and ghost detours must be 0 on the sampled seeds.
- **Batch runs use `--jobs 6`** (user directive) and `--predict-horizon 10` (canonical). `--jobs 6` keeps trace/manifest byte-identical and only perturbs `wallclock_per_step`, which these failure-rate gates do not read.

## Acceptance Criteria

- [ ] **AC1 (ghost elimination):** On the ghost metric (a settle whose committed segment is blocked by a predicted stamp while the un-stamped fold is clear AND no real obstacle is within 1.5 m), seeds **0 and 1** — which currently show 2/13 and 6/6 ghost reroutes — both drop to **0** ghost reroutes with the KF tracker.
- [ ] **AC2 (byte-determinism preserved):** TC63, TC58, and TCb (run-vs-run, two same-seed runs → byte-identical trace JSONL) still pass unchanged in form. Two fresh KF-`LidarTracker` runs over an identical multi-frame fixture return byte-identical `Track` sequences (`dataclasses.astuple`).
- [ ] **AC3 (no `np.linalg`):** the KF core uses only scalar / 2×2 arithmetic (review-time grep shows no `np.linalg` / matrix-inverse call in the tracker path). *(Structural invariant; enforced by review + grep, not a `--check` TC.)*
- [ ] **AC4 (tentative withholding):** a synthetic ≤2-frame spurious cluster (appears, then gone, `< CONFIRM_HITS`) never appears in any `update()` output; constructing `LidarTracker(..., smoothing_frames=3)` now raises `TypeError`.
- [ ] **AC5 (coast-through-merge):** in a synthetic two-obstacle merge fixture, **both confirmed tracks remain present in `update()` output on every merge frame with their pre-merge birth-counter ids preserved** (not deleted and re-born), neither reported `|v|` exceeds ~2.5 m/s across the event, and post-split velocities match pre-merge within tolerance.
- [ ] **AC6 (radius cap):** no reported `Track.radius` exceeds `RADIUS_MAX`, even across a merge that balloons the raw cluster radius to ~0.86 m.
- [ ] **AC7 (contract preserved):** `update(*, snapshot, state, lidar, dt)` still returns `list[Track]` sorted by `id`; the first `update` on a fresh tracker yields no *confirmed* tracks (cold start); the oracle byte-guarantees (TC57/TC58) hold unchanged.
- [ ] **AC8 (kwargs removed):** `smoothing_frames` / `max_track_speed` are gone from `LidarTracker.__init__`; `dwa_predictive.py` constructs the tracker without them; `d_star_lite_predictive.py` is confirmed already-conforming; `--check` passes. (Negative-path: the `TypeError` assertion in AC4.)
- [ ] **AC9 (primary gate):** `d_star_lite_predictive` (KF) 100-seed traffic failure rate on `arena_v1` (`--jobs 6 --predict-horizon 10`) is strictly lower than the T0 baseline (current frame-differencing lidar variant, same flags).
- [ ] **AC10 (DWA no-regression):** the 3-key × 100-seed DWA re-validation shows no failure-rate regression for `dwa_predictive` / `dwa_predictive_paper` vs their T0 baseline; plain `dwa` (control) is byte-unchanged.
- [ ] **AC11 (full check passes):** `python arena/arena.py arena/arena_v1.yaml --check` passes — the rewritten TC64/TCh, the new tracker unit tests, AND every audited TC that touches `LidarTracker` (see T3), with the TC-count banner updated.

## Contracts & Interfaces

Single source of truth for every seam. The tracker's public seam is **unchanged**; only its internals and construction kwargs change.

### Public seam (unchanged — owner T1, consumers T2)

- `class LidarTracker` — `Tracker` Protocol implementation.
- `LidarTracker.update(*, snapshot: object, state: np.ndarray, lidar: np.ndarray, dt: float) -> list[Track]` — returns **confirmed** tracks only, sorted by `id` ascending. `snapshot` ignored. First call on a fresh instance returns `[]`.
- `Track` frozen dataclass `(id: int, x: float, y: float, vx: float, vy: float, radius: float)` — **unchanged**. `id` is now a monotonic per-episode birth counter.
- The `Tracker` Protocol shape is unchanged, so `OracleTracker` and every downstream (`predict_blocked_cells`, `trajectory_conflict`, the DWA rollout) bind to the same output.

### New constructor signature (owner T1, consumer T2)

- `LidarTracker.__init__(self, grid, bearings, range_max=inf)` — `smoothing_frames` and `max_track_speed` are **removed**. `d_star_lite_predictive._make_tracker` already calls this 3-arg form (no change); `dwa_predictive._make_tracker` is edited down to it.

### Module constants (owner T1, in `planners/_predict.py`) — starting values, tunable

Pin exact names + starting values so tests and docs reference one definition:

- `KF_PROCESS_NOISE: float = 4.0` — CV process-noise intensity `q` (accel white-noise, m²/s³). Starting value; tune so the filter tracks a ~1 m/s straight-line mover without lag and damps merge spikes. Continuous CV `Q(q, dt)` per the Data Model.
- `KF_MEASUREMENT_NOISE: float = 0.02` — centroid measurement variance `r` (m²), ~ (0.14 m)² for the near-surface centroid jitter.
- `KF_INITIAL_POSITION_VARIANCE: float = 0.05` — birth covariance on the position states (m²).
- `KF_INITIAL_VELOCITY_VARIANCE: float = 4.0` — birth covariance on the velocity states (m²/s²) — wide, so the first hits move the estimate freely.
- `MAX_PLAUSIBLE_SPEED: float = 2.0` — `fast`-regime cap (was `MAX_TRACK_SPEED`); feeds the gate distance. Deliberately 2.0, NOT 1.5 (issue-#11 fast regime).
- `GATE_SLACK: float = 0.15` — jitter allowance added to the physical displacement bound.
- `ASSOCIATION_GATE_DISTANCE: float = MAX_PLAUSIBLE_SPEED * PREDICT_DT + GATE_SLACK` (≈ 0.35 m) — max centroid-to-**predicted-position** distance for a valid association. Replaces `MAX_ASSOCIATION_DISTANCE = 1.0`.
- `CONFIRM_HITS: int = 3` — consecutive gated hits to promote tentative → confirmed.
- `COAST_MISSES: int = 3` — consecutive misses a confirmed track coasts before deletion.
- `RADIUS_EMA_BETA: float = 0.3` — radius EMA weight (new = β·measured + (1−β)·prev).
- `RADIUS_MAX: float = 0.45` — hard radius cap (1.5× the arena's true 0.3 m mover).
- `RADIUS_MERGE_SUSPECT_RATIO: float = 1.5` — a detection radius jump above this × the filtered radius is treated as a merge; the radius update is skipped that frame.
- `MIN_TRACK_RADIUS` — retained (existing floor; seeds the EMA at birth).
- **Removed:** `MAX_ASSOCIATION_DISTANCE`, `CLUSTER_ID_MULTIPLIER`, `VELOCITY_SMOOTHING_FRAMES`, `MAX_TRACK_SPEED`.

### Internal track state (owner T1, private — not a cross-task seam)

- `_KTrack` — internal per-track record: birth-counter `id`, KF state `[x, y, vx, vy]`, decoupled covariance scalars, filtered `radius`, `hits`, `misses`, `confirmed: bool`. Not exported; `update` maps confirmed `_KTrack`s → `Track`.

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `planners/_predict.py` | T1 | T2, T3, T4 |
| `planners/dwa_predictive.py` | T2 | — |
| `arena/arena.py` | T3 | — |
| `CLAUDE.md`, `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | T4 (findings also written by T0/T5) | — |

## Data Model

The decoupled CV-KF (per track, per axis — x and y share one recursion):

```
State (per axis):   s = [pos, vel]
Transition:         F = [[1, dt], [0, 1]]
Process noise Q(q): [[q*dt^3/3, q*dt^2/2], [q*dt^2/2, q*dt]]
Measurement:        H = [1, 0]  (centroid position);  R = r (scalar)
Predict:  s <- F s ;  P <- F P F^T + Q
Gate:     |z - s_pred_pos| <= ASSOCIATION_GATE_DISTANCE   (Euclidean, on the PREDICTED position)
Update:   y = z - s_pos ;  Sm = P00 + r ;  K = [P00/Sm, P10/Sm]
          s <- s + K*y ;  P <- (I - K H) P
```

The whole update is scalar mul/add; `Sm` is a scalar (no matrix inverse, AC3). `P`/`K` evolve with the track's age and its miss pattern (fixed-order float ops — deterministic; NOT a static precomputed schedule). Radius is a separate scalar EMA, not part of the KF state.

Lifecycle state machine per track:

```
detection unassociated ─────────────────▶ TENTATIVE (hits=1, radius:=max(measured, MIN_TRACK_RADIUS), withheld)
TENTATIVE + gated hit ×(CONFIRM_HITS-1) ─▶ CONFIRMED (emitted)
TENTATIVE + miss ───────────────────────▶ DELETED
CONFIRMED + gated hit ──────────────────▶ CONFIRMED (misses:=0, KF+radius update; radius update SKIPPED if merge-suspect)
CONFIRMED + miss (<COAST_MISSES) ───────▶ CONFIRMED (coast: emit PREDICTED pos, keep last radius, misses+1)
CONFIRMED + miss ×COAST_MISSES ─────────▶ DELETED
```

Radius at birth is seeded from the first detection's floored radius; the merge-suspect gate does NOT apply on the birth frame (no prior filtered radius to compare against).

## Error Handling

- **Empty / all-NaN lidar frame:** detector returns no clusters → every confirmed track takes a miss (coasts up to `COAST_MISSES`), tentatives die. Deterministic; no exception.
- **Degenerate KF:** `Sm = P00 + r` with `r > 0` is always positive; assert `r > 0` at construction, no division guard needed.
- **First frame (no priors):** all detections birth tentative tracks; `update` returns `[]` (AC7 cold start).
- **Association ties (equal float distances under symmetric geometry):** the sort key is `(distance, track_id, detection_rank)` — a total order over the two int fields, so ties never resolve nondeterministically. THE determinism landmine (T1).
- **Obstacle despawn at the perimeter:** the track coasts up to `COAST_MISSES` (≤0.3 s) emitting a bounded predicted stamp at the world edge, then deletes — usually rejected by the corridor gate. Documented, acceptable.
- **Input validation:** keep the existing `update` shape checks (state `(3,)`, lidar `(bearings,)`, `dt > 0`); keep `grid`-not-None and add `r > 0` construction checks.

## Testing Strategy

**Levels:** Unit (in-process, no irsim), Integration (subprocess e2e through the runner), plus offline validation runs.

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC64 (rewrite) | KF tracker determinism across a cluster-count change | Unit | Fixture extended so obstacle B is present ≥ `CONFIRM_HITS` frames (so the two-confirmed-track association is actually exercised) and A's coast tail is accounted for. Two fresh trackers over the fixture return byte-identical `Track` sequences; the +x mover, once **confirmed**, reports `vx>0`, `|vy|` small. The old frame-2 velocity assertion is replaced by a confirmed-frame assertion. |
| TCh (rewrite→negative) | Hardening kwargs removed | Unit | `LidarTracker(grid, bearings, smoothing_frames=3)` raises `TypeError`; `LidarTracker(grid, bearings, max_track_speed=2.0)` raises `TypeError`; `LidarTracker(grid, bearings)` constructs. The old "reported speed never exceeds bound" output-clamp sub-case is DROPPED (no output clamp exists). May be folded adjacent to TC64. |
| TC-K1 (new) | Prediction-gated association tie determinism | Unit | A symmetric two-detection / two-track frame associates identically across two runs; permuting input detection order does not change id assignment (total-order `(distance, id, rank)` tie-break). |
| TC-K2 (new) | Tentative withholding | Unit | A cluster present for only 2 frames (< `CONFIRM_HITS`) never appears in any `update()` output. |
| TC-K3 (new) | Coast-through-merge (present-every-frame, ids preserved) | Unit | Two obstacles approach, merge for 1–2 frames, split. Assert: **both confirmed tracks are present in `update()` output on EVERY merge frame with their pre-merge birth-counter ids** (rules out delete-and-rebirth); neither reported `|v|` exceeds ~2.5 m/s; post-split velocities match pre-merge within tolerance. |
| TC-K4 (new) | Radius cap + merge-suspect | Unit | Across a merge that balloons the raw cluster radius to ~0.86 m, no reported `Track.radius` exceeds `RADIUS_MAX`. |
| TC-K5 (new) | Ghost elimination (in-process, both seeds) | Integration | Driving `d_star_lite_predictive` on **seeds 0 and 1** over a window pinned to contain the baseline ghost events (seed 0: through ≥ tick 45; seed 1: through ≥ tick 200 — the T0 snapshot records the exact ghost ticks), the ghost-reroute count is **0** (currently 2 and 6). Uses the ghost-metric helper defined below. |
| TC63 (verify) | Predictive traffic-on e2e determinism | Integration | Unchanged in form: two same-seed traffic runs → byte-identical trace JSONL, 8-key schema, plans at t=0. |
| TC57 / TC58 (verify) | Oracle h0 / oracle determinism | Integration | Unchanged: oracle keys byte-stable (OracleTracker untouched). |
| TCb (verify) | DWA predictive e2e determinism | Integration | Unchanged in form: DWA keys terminal + deterministic (trajectories change, determinism holds; no golden-byte assertion involves the lidar tracker at h>0). |
| TCe (audit) | DWA head-on braking speed-decrease | Integration | The `CONFIRM_HITS−1` frame confirmation lag delays when the braking layer first sees a crosser; T3 re-checks TCe's speed-decrease window still holds (widen the window or lengthen the approach if the lag shifts it). |
| TCj (audit) | Mid-episode tracker-raise degrade | Unit/Integration | TCj monkeypatches tracker internals; T3 re-points it at surviving symbols (the `update` seam), since the frame-diff internals it may patch are deleted. |
| VAL1 (offline) | Primary 100-seed gate | Validation | `d_star_lite_predictive` (KF) 100-seed failure rate < T0 baseline (AC9). `--jobs 6 --predict-horizon 10`. |
| VAL2 (offline) | DWA 3×100 re-validation | Validation | `dwa_predictive` / `dwa_predictive_paper` no failure-rate regression vs T0; plain `dwa` unchanged (AC10). `--jobs 6 --predict-horizon 10`. |

**Ghost-metric helper (defined in T3, in `arena/arena.py` test scope):** given a controller, wrap `_settle_and_extract` to count settles; on each settle, a ghost is `stamp_blocks_segment AND NOT real_blocks_segment AND nearest_real_obstacle > 1.5 m`, where `real_blocks_segment` is checked against the un-stamped fold (captured by wrapping `_extra_blocked_cells`), and `nearest_real_obstacle` uses `info.dynamic_obstacles` (truth, for measurement only). This is the diag instrumentation from the debug session, promoted into the test file (it does not live in the repo yet — T3 writes it as a helper).

**Test data:** synthetic lidar fixtures built in-process (reuse TC64's fixture-builder pattern — a grid + bearings + hand-placed range returns walking a cluster count). No new irsim dependency for the unit tests. TC-K5 drives a real Arena in-process.

**Run command:** `python arena/arena.py arena/arena_v1.yaml --check` (unit + integration TCs). Validation: `python -m runners.run_experiment --algorithm <key> --predict-horizon 10 --world arena/arena_v1.yaml --num-seeds 100 --jobs 6` per key, with the T0 baseline captured under identical flags before any code change.

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|-----------|------|-------|-------------|
| T0 | Capture pre-change baselines | — | low | `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | BEFORE any code change (so the baseline is not overwritten — `results/` is gitignored and shared-label), run the current frame-differencing lidar variant at `--num-seeds 100 --jobs 6 --predict-horizon 10` (traffic on, arena_v1) for `d_star_lite_predictive`, `dwa_predictive`, `dwa_predictive_paper`, and plain `dwa`. WRITE the resulting failure rates into the findings doc (a "T0 baseline" table) — not just to the gitignored `results/`. Also snapshot the seed-0/seed-1 ghost counts and the exact ticks at which they occur (via the ghost helper on the current tracker), and record them in the findings doc so TC-K5's window can be pinned. Baseline half of AC9/AC10. Long run — background it. |
| T1 | Rework `LidarTracker` into the CV-KF MOT | T0 | xHigh | `planners/_predict.py` | Replace `_associate_and_build` + the `_prev_centroids`/`_prev_velocity_histories` state with: the hand-rolled decoupled CV-KF (Data Model, no `np.linalg`), the `_KTrack` internal state, prediction-gated globally-sorted greedy association (sort key `(distance, track_id, detection_rank)`), the N-hit/M-miss lifecycle with tentative withholding + coast-through-merge (emit predicted pos on coast), monotonic birth-counter ids (reset on construction), and capped-EMA radius (seed at birth from floored measured radius) + merge-suspect gate (skip radius update, not birth frame). Remove `smoothing_frames`/`max_track_speed` from `__init__`; add/rename the module constants per Contracts; DELETE the dead frame-differencing prose (module docstring, tunables comment block `~44–91`, the `Tracker` Protocol "future LidarTracker … frame-differencing" note). Keep the detector and the `update`/`Track` contract unchanged. Deterministic: no RNG, no set-iteration in output/order, sorted iteration, fixed-order reductions. Satisfies AC2–AC7. Blocked by T0 so a parallel executor cannot edit this file mid-baseline-run. |
| T2 | Update the DWA construction site | T1 | med | `planners/dwa_predictive.py` | Change `_make_tracker()` to the kwarg-free `LidarTracker(self._grid, self._bearings, range_max=self._geom.range_max)` form and drop the `VELOCITY_SMOOTHING_FRAMES` / `MAX_TRACK_SPEED` imports/usages (`dwa_predictive.py` lines ~83, ~85, ~492–498). Confirm `d_star_lite_predictive._make_tracker` (line ~509) already conforms — NO edit expected there; verify only. `OracleTracker` construction untouched. Satisfies AC8. |
| T3 | Rewrite/audit tests + add tracker unit tests | T1, T2 | med | `arena/arena.py` | Rewrite TC64 (extend the fixture so B is present ≥`CONFIRM_HITS` frames) and TCh (→ `TypeError` negative-path on the removed kwargs; drop the output-clamp sub-case). Add TC-K1..TC-K5 (Testing Strategy) plus the ghost-metric helper. AUDIT every TC that constructs, monkeypatches, or is timing-coupled to `LidarTracker` — at minimum TC57, TC58, TC63, TC65, TCa, TCb, TCe (braking-window shift under confirmation lag), TCj (re-point monkeypatch at surviving symbols) — and state each one's disposition (unchanged / adjusted) in the test comments. Wire new TCs into the `--check` registry/counters and update the TC-count banner. Satisfies AC2, AC4–AC7, AC11. |
| T4 | Docs | T1, T2, T3 | low | `CLAUDE.md`, `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | Update the `LidarTracker` / predict-family sections of `CLAUDE.md` (KF MOT, lifecycle, removed kwargs, new constants + TC list, id semantics). Extend the findings doc (design recap + a VAL1/VAL2 results table T5 fills). Keep README / `--check` banner counts current. Blocked by T3 so the documented TC list matches what landed. |
| T5 | Validation runs + findings verdict | T0, T1, T2, T3 | low | `docs/plans/2026-07-19-kalman-mot-tracker.findings.md` | Run VAL1 (100-seed `d_star_lite_predictive` KF, `--jobs 6 --predict-horizon 10`) vs the T0 baseline, and VAL2 (the 3 DWA keys × 100). Re-run the seed-0/1 ghost check on the KF tracker. Record failure rates + ghost recheck in the findings doc; state whether AC9/AC10 pass. Long-running — user-launched or backgrounded at `--jobs 6`. |

## Notes for Implementer

- **The determinism landmine is the association sort key.** Symmetric arena geometry produces exactly-equal float gate distances; the sort key MUST be `(distance, track_id, detection_rank)` where `detection_rank` is the detection's index in `rep_cell`-sorted order — a total order over ints. A bare `(distance,)` sort leaks nondeterministic order and breaks TC63/TC64. Single highest-risk line in T1.
- **No `np.linalg`.** The decoupled 2-state filter's `Sm` is a scalar and `K` is a 2-vector; the whole update is scalar mul/add (AC3).
- **`update` returns confirmed tracks only, and coasting emits the PREDICTED position** each miss frame (deterministically), never a frozen last-seen position — else the D* stamp jitters. Tentative tracks are never emitted.
- **The gate must be sized to `MAX_PLAUSIBLE_SPEED = 2.0`, not the `current` 1.5 regime.** A gate sized to 1.5 makes every `fast`-regime crosser fail association each frame, coast out, and die — silently un-tracking exactly the traffic issue-#11 spawns. This is a correctness trap, not a tuning nicety.
- **Birth counter resets on construction, not on `update`.** Each episode builds a fresh tracker via `_make_tracker`; the reset is automatic (mirrors the `id_iter` reset-on-make gotcha). Do NOT add a per-`update` reset.
- **OracleTracker is the control arm — do not touch it.** TC57/TC58's oracle byte-guarantees must hold.
- **Capture T0 baselines BEFORE editing code (T0 blocks T1)** and persist them into the findings doc — once the tracker changes, the current-variant number is unrecoverable without a git checkout, and `results/` is gitignored/shared-label so VAL1 would overwrite it.
- **Fixture-semantics tests fail until rewritten** — TC64 asserts a frame-2 velocity the confirmation lag (N=3) no longer produces, TCh asserts removed kwargs. Rewrite in the same change (T3); do not weaken. Watch TCe (the braking-speed-decrease window shifts by `CONFIRM_HITS−1` frames) and TCj (monkeypatches frame-diff internals that are deleted).
- **TC-K5's baseline ghost ticks come from T0**, so pin its window from the T0 snapshot; asserting exactly 0 ghosts is robust to later horizon/cap re-tuning (0 is the target regardless), but flag it in a comment as an empirical seed-specific gate.
- **Rollback plan:** the change is `planners/_predict.py` internals + one construction site + tests; if VAL1 regresses, `git revert` restores the frame-differencer with no cross-module entanglement (the `Track`/`update` seam never changed).
- **Deferred (v2):** detector-level prediction-conditioned cluster splitting. Revisit only if the post-fix quick read still shows merge-driven ghosts.
