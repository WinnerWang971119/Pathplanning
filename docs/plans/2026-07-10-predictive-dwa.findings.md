# Space-time predictive DWA — findings

## What shipped

A **real** predictive planner that reasons in `(x, y, t)` instead of stamping a
grid. `planners/dwa_predictive.py` extends `DWAController` with a two-layer
collision model: vanilla DWA's present-position clearance as a safety floor, plus
a space-time layer that advances each tracked obstacle at constant velocity inside
the rollout and checks the robot against it **at the matched time step** (the pure
`planners._predict.trajectory_conflict`). Candidates that would meet a moving
obstacle in matched time are rejected; a capped predicted-clearance term biases the
robot to yield early. This realizes Missura & Bennewitz (ICRA 2019) and MDPI
*Actuators* 2025 14(5):207.

Two variants behind the existing tracker seam:
- `dwa_predictive` — lidar frame-differencing velocities (`LidarTracker`),
  **canonical** (13th canonical planner, on the main scatter).
- `dwa_predictive_oracle` — perfect live velocities via the truth seam,
  **experimental** (the ceiling; the only difference from the lidar variant is the
  velocity source — walls are checked identically against the live lidar cloud).

`h0` is a true no-op (byte-identical to plain `dwa`, TC65), so it is the exact
baseline. Determinism, the 8-key traffic trace, and `--predict-horizon` validation
are covered by TC65–TC69.

## Quick-read sweep (10 canonical-prefix seeds, traffic on, arena_v1, h0 vs h10)

Mirrors the D* Lite predictive quick read. `h0` is plain `dwa` (byte-identical).
Horizon 10 = 1.0 s of lookahead.

| Variant | Horizon | Failure rate (crash+timeout) | Median time (s) | success / crash / timeout |
|--------|--------:|:----------------------------:|:---------------:|:-------------------------:|
| plain `dwa` (= h0)                  | 0  | **0.50** (5/10) | 85.9 | 5 / 2 / 3 |
| `dwa_predictive_oracle` (perfect v) | 10 | **0.80** (8/10) | 89.2 | 2 / 4 / 4 |
| `dwa_predictive` (lidar v)          | 10 | **0.70** (7/10) | 86.2 | 3 / 4 / 3 |

All 30 episodes ran to a terminal state; `planner_error` is null everywhere (DWA
`reset()` never raises); 0 runner failures. Master seed 20260605, first 10 derived
seeds.

## Interpretation

Space-time prediction layered onto DWA is **net-harmful at h10 on these 10 seeds —
even with a perfect-velocity oracle** (0.50 → 0.80 oracle, 0.50 → 0.70 lidar). This
is the **opposite** of the D* Lite predictive result, where the oracle cut failure
0.60 → 0.20 by stamping. The contrast is the lesson.

**Why.** DWA is *already* a velocity-space reactive planner: every tick it
forward-simulates each `(v, ω)` candidate and rejects present-position collisions,
which is exactly why plain DWA already halves hazard versus the grid planners
(`project-replanning-no-safer-than-single-plan`). Adding a hard space-time
rejection layer on top **double-counts moving obstacles** — they are avoided both at
their current position (the present-position floor) AND at every predicted future
position along the horizon (the space-time layer). Under ~20 crossing obstacles, the
union of predicted `(x, y, t)` reservations over a 1.0 s horizon rejects a large
fraction of the dynamic window each tick, shrinking the feasible set until the robot
is forced into its in-place-rotation fallback — the classic **freezing-robot
over-conservatism** (Trautman & Krause 2010). The harm is graded, not a total freeze
(the oracle still solves 2/10), and it shows up as BOTH more timeouts (the robot
creeps or stalls) and more crashes (it gets clipped while creeping), with median time
edging up (85.9 → 89.2 s = more exposure). Crucially, the **oracle fails too**: with
perfect velocities the estimator is not the bottleneck, so the problem is the policy
(the over-constraint), not perception — and it is not a bug (the space-time check is
deterministic and correct, TC65–TC68).

**Takeaway.** Prediction helps a planner that otherwise *ignores* motion — a global
grid search like D* Lite plans straight into where movers are heading, so stamping
their future footprints is pure gain. Prediction hurts a planner that *already*
reasons in velocity space — DWA's one-step reactive check is far less conservative
than a hard multi-step space-time reservation, so bolting the reservation on top
just freezes it. "Real" `(x, y, t)` prediction is therefore not automatically better
than stamping or than plain reactivity; it has to be matched to a planner that lacks
motion reasoning, or made much softer.

**Levers this quick read does not test** (the clear next steps, mirroring how the
D* Lite estimator refinement was deferred): a shorter horizon (h5 reserves a smaller
`(x, y, t)` tube and may over-reject far less), a **soft** time-to-collision penalty
instead of a hard rejection, **dropping the present-position floor for tracked
movers** so they are not double-counted, a smaller collision margin, or a lower
`PREDICTED_CLEARANCE_WEIGHT`. The full 50-seed × `{0, 5, 10, 20}` horizon sweep would
locate the best non-zero horizon, if any clears the plain-`dwa` baseline.

## What this quick read does and does not establish

- Establishes: the direction and rough magnitude of the space-time DWA effect at
  h10 on these 10 seeds, oracle vs lidar-estimator, against the exact plain-`dwa`
  baseline (`h0`).
- Does not establish: the rigorous 50-seed verdict, the best non-zero horizon among
  `{h5, h10, h20}`, or the obstacle-speed-cap dependence (issue #11). Those are the
  full sweeps below, which are the user's to launch.

## Reproduce

The full 50-seed × `{0, 5, 10, 20}` horizon sweep and the obstacle-speed sweep are
deferred (the user launches them):

```powershell
.venv\Scripts\Activate.ps1

# Horizon sweep (now covers both D* Lite and DWA predictive families)
python -m runners.run_horizon_sweep --world arena/arena_v1.yaml
python -m runners.plot_horizon_sweep --world arena/arena_v1.yaml

# Obstacle-speed-cap sweep — dwa_predictive is in the focus set
python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus
python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus

# Full 13-planner canonical comparison (dwa_predictive lands on the main scatter)
python -m runners.run_all --world arena/arena_v1.yaml
python -m runners.plot    --world arena/arena_v1.yaml
```
