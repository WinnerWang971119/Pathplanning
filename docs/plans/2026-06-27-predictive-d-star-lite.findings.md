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

## T14 lidar estimator — (to be filled in after the lidar variant lands)
