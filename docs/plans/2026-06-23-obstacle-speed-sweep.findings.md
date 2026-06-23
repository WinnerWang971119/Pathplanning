# Obstacle-Speed-Cap Sweep: Findings Note (issue #11)

This note covers how to run the obstacle-speed-cap sweep, how to read its two
charts, and what the shipped smoke run does and does not prove. The full
50-seed result and its verdict are the user's to produce; this work ships the
tooling.

## The hypothesis

From issue #11: the dynamic-obstacle speed band sets a ceiling on how fast
traffic can move relative to the robot. If no obstacle can outrun the robot
(the cap is at or below 1.0× robot top speed), an incremental planner like
D* Lite should be able to dodge nearly every crossing obstacle and reach the
goal on close to every seed, so its crash rate should drop toward zero. Once the
cap exceeds 1.0× (obstacles can move faster than the robot), some collisions
become geometrically unavoidable no matter how good the planner is, so the
crash rate should floor above zero and stay there as the cap climbs.

The four regimes are picked to straddle that 1.0× line:

| Regime | Speed band (× robot top speed) | Max cap | Where it sits |
| --- | --- | --- | --- |
| `slow` | 0.3 – 0.7 | 0.7 | every obstacle slower than the robot |
| `matched` | 0.3 – 1.0 | 1.0 | nothing outruns the robot |
| `current` | 0.3 – 1.5 | 1.5 | the Mission baseline; some outrun it |
| `fast` | 0.5 – 2.0 | 2.0 | faster, denser-feeling traffic |

`current` is the Mission baseline and is byte-identical to the prior harness
behavior, so the sweep adds three new caps around the existing one.

## How to run the full experiment

Traffic is ON by default; the sweep is the whole point with traffic on.

```powershell
.venv\Scripts\Activate.ps1

# 1. Drive the focus set across all 4 regimes over the canonical 50 seeds.
python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus

# 2. Chart the four regime subtrees.
python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus
```

The driver writes one per-regime subtree per planner:
`results/speed_<regime>/<world_stem>/<label>/<seed>.json` for each of
`slow`, `matched`, `current`, `fast`. The plotter reads those four subtrees and
writes `failure_rate_vs_cap.png`, `median_time_vs_cap.png`, and
`speed_sweep_summary.csv` into `results/<world_stem>/speed_sweep_plots/`.

The `focus` set is `a_star_once`, `d_star_lite`, `dwa`, `apf`: the static
baseline, the incremental planner the hypothesis is about, and the two reactive
planners the issue calls out. This is a multi-hour run: it is 50 seeds × 4
regimes × 4 planners (800 episodes), and D* Lite is the slowest planner and
dominates the wall time, approaching the 600 s per-episode wall on some `fast`
seeds. Run it when you can leave it.

Use `--algorithms all` for the full 11-planner picture instead of `focus`. That
is 11 labels vs 4 (about 2.75× the episode count; `focus` already includes D*
Lite, the slowest planner), with the replan families adding per-replan wall time
at the canonical `--replan-k 5`.

Both the driver and the plotter take `--master-seed`, `--num-seeds`, and
`--jobs`; the driver also takes `--resume` and `--traffic` / `--no-traffic`.
For a clean wall-clock comparison run the driver at `--jobs 1` (contention
perturbs `wallclock_per_step`), but the failure-rate and time-to-goal numbers
the two sweep charts read are byte-identical at any `--jobs`.

## How to read the two charts

Both charts put the **max-cap factor** on the x-axis (0.7 / 1.0 / 1.5 / 2.0,
one tick per regime) and draw one line per present algorithm.

- **`failure_rate_vs_cap.png`**: failure rate (crash + timeout + planner_error
  + DNF) on the y-axis. Down is better. The hypothesis reads directly off
  D* Lite's line: watch whether it drops to roughly 0 in the `slow` and
  `matched` regimes (caps 0.7 and 1.0, where nothing outruns the robot) and
  whether it floors above 0 once the cap passes 1.0× into `current` and `fast`.
  A line that stays near zero across all four caps, or one that is flat and
  high, would refute the "cap matters" story.
- **`median_time_vs_cap.png`**: median time-to-goal over successful episodes,
  per regime. This is the cost side: it shows whether dodging faster traffic
  makes the successful runs slower, and lets you separate "safe but slow" from
  "safe and fast" as the cap changes. A regime with too few successes degrades
  to a gap in the line rather than a misleading point.

## What the smoke proved

The orchestrator ran a small end-to-end smoke: a `--num-seeds 1 --no-traffic`
pass through the driver, producing the four per-regime subtrees, then the
plotter, producing the two PNGs and the CSV. It also ran the 7-case plotter
`--selfcheck` (synthetic fixtures, no irsim) and the full 55-case
`python arena/arena.py arena/arena_v1.yaml --check`.

This smoke validates **plumbing only**. With `--no-traffic` there are no dynamic
obstacles, so the speed band is a no-op: all four regimes run identical episodes
and the chart lines are flat. That is expected and is not a result; the real
signal needs the traffic-ON 50-seed run above, which is the user's to launch
(per the agreed scope: this work ships tooling plus a smoke, not the verdict).

The chart **structure** (one line per algorithm across the four caps, with
D* Lite included as its own series, correct axes, and all four regimes as
distinct x positions) is proven by the plotter `--selfcheck`, which builds
synthetic per-regime data with varied failure rates and times so the lines
actually move. So the chart that would answer the hypothesis is rendered
correctly; only the data behind it is pending.

## Caveat: `a_star_once` is a static baseline, not a hypothesis subject

`a_star_once` is in the `focus` set as a reference line, not as a planner whose
crash rate should track the speed cap. It plans once at t=0 and never dodges, so
its outcome on a seed depends on whether traffic happens to cross its fixed path
while the robot is on it. A faster obstacle crosses that path in less time, so a
higher cap does not cleanly raise (or lower) `a_star_once`'s crash rate. The
relationship is not monotone in the cap. Read D* Lite as the primary hypothesis
subject, with DWA and APF as the reactive comparison; treat `a_star_once`'s line
as the "no dodging at all" baseline the others are measured against.
