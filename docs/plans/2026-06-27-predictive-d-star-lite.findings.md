# Predictive D* Lite — findings

## T10 oracle checkpoint (go/no-go gate)

**Outcome: PASS — greenlight to build the lidar estimator (T11–T14).**

### Performance rework (prerequisite for any sweep)

The first oracle build ran a from-scratch A* reachability probe inside the
fail-open peel on every stamp-bearing tick (~1.8 s/tick), so a horizon>0 episode
never terminated. The peel was moved to settle-time: the per-tick hook only
stamps the predicted footprint, and the fail-open peel runs inside
`_settle_and_extract`, reusing the D* Lite search's own `compute_shortest_path`
/ `extract_path` (g(start) finite == solvable) as the reachability oracle, and
dropping farthest-future groups only when the stamp actually seals the grid.
Result: oracle `h10` traffic dropped to ~16 ms/tick (near plain `d_star_lite`),
reaching the arena_v1 traffic goal at seed 42 (sim 79 s).

### Quick-read sweep (10 canonical-prefix seeds, traffic on, arena_v1)

Instead of the full 50-seed × {0,5,10,20} sweep, a quick read over the first 10
canonical seeds at horizons {0, 10} was run to gate the estimator decision.
Because TC57 proves `d_star_lite_oracle_h0` is byte-identical to plain
`d_star_lite`, `h0` is the exact baseline.

| Horizon | Failure rate (crash+timeout) | Median time (s) | n |
|--------:|:----------------------------:|:---------------:|:-:|
| h0 (= plain d_star_lite) | 0.60 (6/10) | 80.1 | 10 |
| h10 (oracle, T = 1.0 s)  | 0.20 (2/10) | 81.6 | 10 |

The oracle cut the failure rate by two-thirds (Δ = −0.4, i.e. 4 of 10 seeds),
for ~1.5 s of extra median time (the expected mild over-blocking cost). This is
well past the AC1 margin (≥ 2 seeds / ≥ 0.04). All 20 episodes ran to a terminal
state with zero runner failures.

### Decision

The user reviewed the quick read and greenlit building the lidar-only estimator
(T11–T14) directly. The **full 50-seed × {0,5,10,20} sweep is deferred** (it is
the only thing the quick read does not establish: the rigorous 50-seed verdict
and the best non-zero horizon among h5/h10/h20). It can be run later with:

```
python -m runners.run_horizon_sweep --world arena/arena_v1.yaml
python -m runners.plot_horizon_sweep --world arena/arena_v1.yaml
```

## T14 lidar estimator — refuted at v1 (oracle ceiling holds, perception is the wall)

**Outcome: the realizable lidar estimator makes things worse, not better.** The
oracle proves motion-aware stamping helps with perfect velocities; the
frame-differencing estimator that has to earn those velocities from lidar does
not — at the same horizon where the oracle wins, it fails every seed.

### Quick-read sweep (10 canonical-prefix seeds, traffic on, arena_v1, h0 vs h10)

Same 10-seed quick read as T10, now charting both predictive keys. `h0` is the
plain-`d_star_lite` baseline for both (byte-identical, TC57).

| Variant | Horizon | Failure rate | Median time (s) | success / crash / timeout |
|--------|--------:|:------------:|:---------------:|:-------------------------:|
| plain d_star_lite (= h0)         | 0  | 0.60 | 80.1 | 4 / 6 / 0 |
| d_star_lite_oracle (perfect v)   | 10 | 0.20 | 81.6 | 8 / 2 / 0 |
| d_star_lite_predictive (lidar v) | 10 | 1.00 | —    | 0 / 8 / 2 |

The oracle cuts crashes by two-thirds (6 -> 2). The lidar variant at the same
horizon goes the other way: crashes rise 6 -> 8 AND two seeds newly time out, so
every one of the 10 episodes fails (failure rate 1.00, no successful time).

### Why it fails — over-stamping from a noisy estimator

A 120-step traffic smoke of the lidar controller showed the mechanism directly:
the frame-differencing tracker reported up to **27 clusters from 20 obstacles**
(it over-segments an obstacle's return arc and keeps some incompletely-subtracted
wall returns), and the per-tick stamp ran up to **~5128 of the 6000-cell area
cap**. The estimator's velocity vectors are noisy (the centroid of the visible
near-surface arc jitters frame to frame), so the cone footprints land in the
wrong places and cover a large fraction of the arena. The fail-open peel keeps a
path alive, but the robot is steered into long detours that either drive it into
a *different* obstacle (the extra crashes) or stall it past the 120 s wall (the
new timeouts).

### Interpretation

This is the result the plan's Notes anticipated: "the estimator risk does not
vanish by choosing stamping ... velocity estimation is the same hard,
determinism-fragile problem." Stamping wins on *layering* (the D\* Lite search
core is untouched, determinism holds, TC63/TC64 pass), not by dodging the
perception problem. The oracle gate (T10) existed precisely to separate "is
motion-awareness worth it?" (yes) from "can we estimate motion well enough from
lidar?" (not with the v1 estimator). The gap between the oracle's 0.20 and the
estimator's 1.00 is the cost of imperfect perception, and it is the whole story
here.

### What this quick read does and does not establish

- Establishes: the v1 lidar estimator (no velocity clamp, grid-bucket clustering,
  cone widening) is net-harmful at h10 on these 10 seeds, by a wide margin.
- Does not establish: whether a shorter horizon (h5 stamps a smaller footprint
  and might do less damage), a velocity clamp, tighter clustering, or a smaller
  area cap could pull the estimator back below baseline; nor the rigorous 50-seed
  number. These are estimator-refinement questions beyond the plan's T11–T14
  scope.

### Reproduce

```
python -m runners.run_horizon_sweep --world arena/arena_v1.yaml
python -m runners.plot_horizon_sweep --world arena/arena_v1.yaml
```

(`SWEEP_ALGORITHMS` in both the driver and the plotter now includes
`d_star_lite_predictive`, so the charts carry the oracle ceiling and the lidar
line together.)
