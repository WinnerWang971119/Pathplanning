# Path Planning Arena

A controlled comparison of path-planning algorithms on a shared, reproducible
arena with crossing traffic, built on top of [`irsim`](https://github.com/hanruihua/ir-sim)
(a 2D differential-drive robot simulator).

The end product is a 2D scatter plot of **time-to-goal vs. crash-rate** —
down-and-left wins. The deeper goal is to understand *why* some planners are
fast, some are safe, and whether the two properties trade off. The full
research design lives in [`Mission.md`](Mission.md).

Every algorithm runs against the same 50 seeded traffic streams in the same
50×50 world, start `(2, 2)` → goal `(48, 48)`, so the cross-algorithm
comparison is apples-to-apples.

---

## Repository layout

```
pathplanning/
├── arena/                    # the seeded test environment (Phase 0 + 2)
│   ├── arena.py              #   Arena: wraps irsim, uniform step() API, --check suite
│   ├── dynamic.py            #   DynamicObstacle + TrafficSpawner (crossing traffic)
│   ├── speed_regimes.py      #   obstacle-speed band table + CLI helpers (stdlib only)
│   ├── arena_v1.yaml         #   canonical 50×50 world (walls + 12 pillars)
│   ├── arena_v2_hard.yaml    #   second 50×50 world (walls relocated)
│   └── arena_no_path.yaml    #   start boxed in → A* must fail (failure-path fixture)
├── planners/                 # pluggable planner adapters (Phase 6 + 7)
│   ├── _types.py             #   Controller protocol (reset + act) + Path type
│   ├── _grid.py              #   shared grid substrate + lidar fold + ALGORITHMS registry
│   ├── _predict.py           #   motion-prediction substrate (Track, trackers, stamp geometry + space-time conflict)
│   ├── a_star.py             #   a_star_once / a_star_replan
│   ├── dijkstra.py           #   dijkstra_once / dijkstra_replan (A* with a zero heuristic)
│   ├── d_star_lite.py        #   d_star_lite (incremental; rejects --replan-k)
│   ├── dwa.py                #   dwa (reactive Dynamic Window Approach)
│   ├── apf.py                #   apf (reactive artificial potential fields)
│   ├── rrt.py                #   rrt_once / rrt_replan
│   ├── rrt_star.py           #   rrt_star_once / rrt_star_replan
│   ├── d_star_lite_predictive.py  # d_star_lite_predictive (canonical, h10) / d_star_lite_oracle (experimental)
│   ├── _costfield.py         #   cost-to-go field (Dijkstra-from-goal) for DWA global guidance
│   └── dwa_predictive.py     #   space-time DWA: dwa_predictive (canonical, h10) + 3 experimental variants
├── runners/                  # experiment harness (Phase 1 + 3 + 5)
│   ├── run_episode.py        #   one planner × one seed × one world → metrics + trace
│   ├── run_experiment.py     #   one planner × the canonical 50 seeds → batch + manifest
│   ├── run_all.py            #   all 13 canonical planners → the plotter's input
│   ├── plot.py               #   read-only plotter → summary.csv + 7 comparison charts
│   ├── run_speed_sweep.py    #   one planner set × the 4 obstacle-speed regimes
│   ├── plot_speed_sweep.py   #   failure-rate / median-time vs speed-cap charts
│   ├── run_horizon_sweep.py  #   predictive keys × prediction horizons {0,5,10,20}
│   ├── plot_horizon_sweep.py #   failure-rate / median-time vs horizon charts
│   └── filter_seeds.py       #   flags degenerate seeds (every planner dies instantly)
├── results/                  # generated metrics/traces (gitignored except .gitkeep)
├── docs/plans/               # per-phase implementation plans + findings
├── manual.py                 # standalone demo: naive go-to-goal
├── manual_obstacle.py        # standalone demo: reactive lidar avoidance
├── manual_astar.py           # standalone demo: A* planner + waypoint follower
├── test.py                   # standalone demo: minimal irsim "hello world"
├── *.yaml                    # demo worlds (robot_world, obstacle, obstacle_harder)
├── tests/                    # A* edge-case world fixtures (inputs, not pytest files)
├── Mission.md                # the research plan (phases 0–7)
└── requirements.txt          # irsim, numpy, pyyaml, matplotlib
```

The single-file demos (`test.py`, `manual*.py`) are self-contained and don't
share code with each other. The `arena/` + `planners/` + `runners/` stack is
the reusable harness that drives the actual comparison study.

---

## Setup

Windows + PowerShell. A `.venv/` is already provisioned at the repo root.

```powershell
# Activate the virtual environment (do this in every new shell)
.venv\Scripts\Activate.ps1

# Install / refresh dependencies if needed
pip install -r requirements.txt
```

Dependencies: `irsim`, `numpy`, `pyyaml`, `matplotlib`. There is no separate
build step.

---

## Quick start

```powershell
.venv\Scripts\Activate.ps1

# 1. Eyeball the canonical world in a render window
python arena/arena.py arena/arena_v1.yaml --render

# 2. Run A* against one seed (traffic on by default)
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml

# 3. Run A* against all 50 canonical seeds
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml

# 4. Run all 13 canonical planners, then chart the comparison
python -m runners.run_all  --world arena/arena_v1.yaml
python -m runners.plot     --world arena/arena_v1.yaml

# Results land under results/arena_v1/<label>/ ; charts under results/arena_v1/plots/
```

---

## The standalone demos

Each opens an irsim render window. Run them directly with `python`.

| Command | What it does |
| --- | --- |
| `python test.py` | Minimal irsim "hello world" on `robot_world.yaml`. |
| `python manual.py` | Pure proportional go-to-goal on `obstacle.yaml`. No obstacle awareness. |
| `python manual_obstacle.py` | Reactive lidar avoidance on `obstacle_harder.yaml`. Repulsive turn from close returns + a left/right clearance bias. |
| `python manual_astar.py` | A* global planner + waypoint follower on `obstacle_harder.yaml`. |
| `python manual_astar.py tests\no_path.yaml` | Run the A* planner against a specific world (positional arg). |

`manual_astar.py` is the substantive demo: it parses the world YAML into a
frozen `WorldModel`, rasterizes an occupancy grid inflated by the robot radius
plus a safety margin, runs 8-connected A* with no corner-cutting, collapses the
dense grid path into a small set of line-of-sight-checked waypoints, then
follows them with a heading-gated speed schedule. All tuning knobs are the
`UPPER_SNAKE_CASE` constants at the top of the file.

---

## The planner families

Seventeen controllers are registered. Thirteen are **canonical** (they land on the
headline scatter, including the lidar-only `d_star_lite_predictive` and
`dwa_predictive` at h10); four are **experimental** (`d_star_lite_oracle`,
`dwa_predictive_oracle`, `dwa_predictive_paper`, `dwa_predictive_paper_oracle` —
perfect-velocity cheats and braking-only ablations, excluded from the canonical
comparison). Pick one with `--algorithm <key>`.

| Key | Family | Notes |
| --- | --- | --- |
| `a_star_once` | grid A* | Plans once on the static grid, follows it forever. |
| `a_star_replan` | grid A* | Re-searches the lidar-folded grid every K acts. Needs `--replan-k`. |
| `dijkstra_once` | grid Dijkstra | A* with a zero heuristic (same machinery). |
| `dijkstra_replan` | grid Dijkstra | Needs `--replan-k`. |
| `d_star_lite` | incremental D* Lite | Hand-rolled Koenig–Likhachev; rejects `--replan-k`. |
| `dwa` | reactive | Dynamic Window Approach (velocity output, no global plan). |
| `apf` | reactive | Khatib artificial potential fields (velocity output, no global plan). |
| `rrt_once` | sampling RRT | Plans once on the static grid. |
| `rrt_replan` | sampling RRT | Re-grows on the lidar fold every K acts. Needs `--replan-k`. |
| `rrt_star_once` | sampling RRT* | Adds choose-parent + rewire. |
| `rrt_star_replan` | sampling RRT* | Needs `--replan-k`. |
| `d_star_lite_oracle` | predictive (experimental) | Stamps each obstacle's predicted future footprint onto the grid using **perfect** velocities (the motion-aware ceiling). Needs `--predict-horizon`. |
| `d_star_lite_predictive` | predictive (canonical, h10) | Same stamp, velocities **estimated from lidar** (frame-differencing). Needs `--predict-horizon`; canonical planner runs at h10. |
| `dwa_predictive` | predictive DWA (canonical, h10) | **Space-time** DWA: forward-simulates each obstacle inside the rollout and checks matched-time collision via an emergency-braking inevitable-collision test, plus a cost-to-go global guidance field. Velocities **estimated from lidar**. Needs `--predict-horizon`; canonical at h10. |
| `dwa_predictive_oracle` | predictive DWA (experimental) | Same braking-inevitability + global guidance model with **perfect** velocities (the ceiling). Needs `--predict-horizon`. |
| `dwa_predictive_paper` | predictive DWA (experimental) | The braking-inevitability layer **only**, no cost-to-go guidance field — the Missura-faithful ablation. Velocities estimated from lidar. Needs `--predict-horizon`. |
| `dwa_predictive_paper_oracle` | predictive DWA (experimental) | Braking-only ablation with **perfect** velocities. Needs `--predict-horizon`. |

The `_replan` families (`a_star_replan`, `dijkstra_replan`, `rrt_replan`,
`rrt_star_replan`) require `--replan-k`; every other key rejects it. The
predictive keys require `--predict-horizon` and reject `--replan-k`. The
reactive (`dwa`, `apf`) and `_once` planners are expected to stall or crash in
the traffic world — that is the experimental signal, not a bug.

The D* Lite predictive family is documented in
`docs/plans/2026-06-27-predictive-d-star-lite.md` and its findings (the oracle
confirms motion-awareness helps; the v1 lidar estimator does not yet) in
`docs/plans/2026-06-27-predictive-d-star-lite.findings.md`. The predictive-DWA
rebuild — the braking-inevitability collision model and the cost-to-go
guidance field — is documented in
`docs/plans/2026-07-10-predictive-dwa-braking.md`, with findings in
`docs/plans/2026-07-10-predictive-dwa-braking.findings.md`: the braking policy
beats plain `dwa` on a quick read, but the guidance field measured
net-harmful, so the canonical `dwa_predictive` key currently trails plain
`dwa` until the field is reworked.

---

## The experiment harness

Each layer is usable from the command line.

### 1. Arena — the seeded environment

`arena/arena.py` wraps irsim and exposes a uniform
`step(action) -> (state, lidar, done, info)` interface. The canonical world is
`arena/arena_v1.yaml` (50×50, two staggered length-30 walls + 12 circle
pillars). Pass `traffic=True` to spawn a 20-obstacle population of crossing
traffic that travels in straight lines and bounces off the arena walls and the
interior static obstacles (so it stays inside, never passes through an obstacle,
and the robot can't wait it out).

```powershell
# Visible smoke loop — drive the world and watch the render window
python arena/arena.py arena/arena_v1.yaml --render

# Headless verification suite (68 checks, TC1–TC64 + TC-CLI/TC-FWD; ~50 min)
python arena/arena.py arena/arena_v1.yaml --check
```

`--check` is the health gate for the whole harness. It covers the Arena API,
the episode runner, the traffic substrate, the batch runner, every planner
family end-to-end, the obstacle-speed sweep, and the predictive (motion-aware)
family. All 68 PASS means the harness is healthy. (With neither flag, it
defaults to `--check`.)

| Flag | Default | Meaning |
| --- | --- | --- |
| `yaml_path` (positional) | required | World YAML, e.g. `arena/arena_v1.yaml`. |
| `--seed N` | 42 | Master seed for the smoke/check run. |
| `--render` | off | Interactive smoke loop in a visible window. |
| `--check` | (default) | Run the headless TC1–TC64 + TC-CLI/TC-FWD verification suite. |

### 2. `run_episode` — one planner, one seed

`runners/run_episode.py` wires a registered planner to the Arena, runs a single
episode, and writes per-episode metrics plus a step-by-step trace.

```powershell
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--algorithm NAME` | required | Registered planner, e.g. `a_star_once`. |
| `--seed N` | required | Episode seed. |
| `--world PATH` | required | World YAML. |
| `--replan-k N` | none | Replan cadence; required for the `_replan` family, forbidden otherwise. |
| `--predict-horizon N` | none | Lookahead in steps (T = N × 0.1 s); required for the predictive family, forbidden otherwise. |
| `--speed-regime {slow,matched,current,fast}` | `current` | Obstacle-speed band (factor of robot top speed). |
| `--speed-min-factor` / `--speed-max-factor` | none | Raw off-menu band; mutually exclusive with `--speed-regime`. |
| `--render` | off | Open the irsim render window. |
| `--results-dir DIR` | `results` | Override the output directory. |
| `--traffic` / `--no-traffic` | traffic on | Toggle Phase 2 crossing traffic. |

A* `_once` planners don't dodge, so most traffic seeds end in collision — that
is the experimental signal the scatter plot consumes. Use `--no-traffic` to
reproduce the deterministic static-world success path.

### 3. `run_experiment` — one planner, the canonical 50 seeds

`runners/run_experiment.py` derives 50 seeds from a single master seed via
`SeedSequence.spawn` and shells out to `run_episode` once per seed (one fresh
irsim subprocess each). This is what guarantees every algorithm faces the same
50 traffic streams.

```powershell
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--algorithm NAME` | required | Registered planner. |
| `--world PATH` | required | World YAML. |
| `--replan-k N` | none | Replan cadence; required for the `_replan` family. Forwarded to each episode and recorded in the manifest. |
| `--predict-horizon N` | none | Prediction horizon (steps); required for the predictive family. Forwarded + recorded in the manifest. |
| `--speed-regime` / `--speed-min-factor` / `--speed-max-factor` | `current` | Obstacle-speed band; recorded in the manifest. |
| `--master-seed N` | 20260605 | Master seed the 50 episode seeds derive from. |
| `--num-seeds N` | 50 | Run a prefix of the canonical stream (prefix-stable). |
| `--jobs N` | 1 | `1` = sequential. `N>1` = up to N concurrent subprocesses. |
| `--results-dir DIR` | `results` | Forwarded to each episode. |
| `--resume` | off | Skip seeds whose `<seed>.json` already exists. |
| `--traffic` / `--no-traffic` | traffic on | Forwarded to each episode. |

Result bytes are identical at any `--jobs` value; only `wallclock_per_step`
(a Mission.md "freebie" metric) is perturbed by contention. Produce headline
wall-clock numbers with `--jobs 1`.

### 4. `run_all` + `plot` — the whole canonical comparison

`runners/run_all.py` runs all 13 canonical planners against the canonical seed
stream in one shot: a parallel bulk pass into `results/<world_stem>/<label>/`,
then a serial wallclock mini-pass into
`results/__wallclock__/<world_stem>/<label>/` for a clean per-step wall-clock.
`runners/plot.py` is the read-only plotter: it reads those JSONs and writes a
`summary.csv` plus the seven comparison charts as PNGs into
`results/<world_stem>/plots/` (gitignored).

```powershell
python -m runners.run_all --world arena/arena_v1.yaml          # produce the data (long run)
python -m runners.plot    --world arena/arena_v1.yaml          # chart it (read-only, no irsim)
python -m runners.plot    --selfcheck                          # plotter's headless fixture suite
```

See "The Phase 5 plotter and batch driver" in `CLAUDE.md` for the seven charts.

### 5. The obstacle-speed-cap sweep

`runners/run_speed_sweep.py` runs a planner set across all four
dynamic-obstacle speed bands so you can see how the obstacle-speed cap drives
failure rate and time-to-goal. `runners/plot_speed_sweep.py` charts the result.

```powershell
# Drive the focus set (a_star_once, d_star_lite, dwa, dwa_predictive, apf) across the 4 regimes
python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus

# Chart it: failure-rate-vs-cap + median-time-vs-cap (x = max-cap factor)
python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus
```

The four regimes are factors of robot top speed: `slow` 0.3–0.7, `matched`
0.3–1.0, `current` 0.3–1.5 (the Mission baseline), `fast` 0.5–2.0. Pass
`--algorithms all` for the full 13-planner picture (~3× the episode count, more
in wall time as the replan families add per-replan cost). The driver writes one
per-regime subtree per planner under
`results/speed_<regime>/<world_stem>/<label>/`; the plotter writes
`failure_rate_vs_cap.png`, `median_time_vs_cap.png`, and a
`speed_sweep_summary.csv` into `results/<world_stem>/speed_sweep_plots/`.
`python -m runners.plot_speed_sweep --selfcheck` runs the plotter's headless
fixture suite (no irsim). This is a multi-hour run; see
`docs/plans/2026-06-23-obstacle-speed-sweep.findings.md` for how to read the
charts and the hypothesis the sweep tests.

The same `--speed-regime {slow,matched,current,fast}` knob is available on a
single `run_episode` / `run_experiment` run (default `current` is byte-identical
to the prior baseline), with raw `--speed-min-factor` / `--speed-max-factor`
overrides for off-menu bands.

### 6. The prediction-horizon sweep (motion-aware planners)

`runners/run_horizon_sweep.py` runs the four predictive keys
(`d_star_lite_oracle`, `d_star_lite_predictive`, `dwa_predictive_oracle`,
`dwa_predictive`) across the prediction horizons `{0, 5, 10, 20}` steps (T =
steps × 0.1 s; `h0` is the plain baseline). `runners/plot_horizon_sweep.py` charts failure rate and
median time vs horizon, with the oracle (perfect-velocity ceiling) and lidar
(estimated) lines together.

```powershell
python -m runners.run_horizon_sweep  --world arena/arena_v1.yaml
python -m runners.plot_horizon_sweep --world arena/arena_v1.yaml
python -m runners.plot_horizon_sweep --selfcheck      # headless fixture suite (no irsim)
```

The driver writes label dirs `results/<world_stem>/<key>_h<steps>/`; the plotter
writes `failure_rate_vs_horizon.png`, `median_time_vs_horizon.png`, and
`horizon_sweep_summary.csv` into `results/<world_stem>/horizon_sweep_plots/`. A
single horizon-bearing run is also available directly via `run_episode` /
`run_experiment --predict-horizon`.

### 7. The degenerate-seed filter

`runners/filter_seeds.py` reads a world's result tree and flags any canonical
seed where every required planner crashed within the first 4 s of sim time —
a spawn no planner could have dodged — so `plot.py` can drop it from the
headline `failure_rate`. It is read-only and post-hoc: it never re-runs an
episode.

```powershell
# Default required set is all13 (the 12 canonical + the experimental oracle at h10).
# run_all already produces d_star_lite_predictive_h10; produce the oracle dir too:
python -m runners.run_experiment --algorithm d_star_lite_oracle --predict-horizon 10 --world arena/arena_v1.yaml

python -m runners.filter_seeds --world arena/arena_v1.yaml
python -m runners.filter_seeds --selfcheck    # headless fixture suite (no irsim)
```

Without the experimental oracle dir, the default `--planners all13` returns
`indeterminate` (exit 3) naming the missing label — a loud no-op, not a
silent one. Pass `--planners canonical` to filter on the 12 canonical
planners only, skipping the `run_experiment` call above. `runners/plot.py`
then reads the resulting `_seed_filter.json` sidecar automatically (pass
`--no-filter` to ignore it).

---

## Results layout

Output is partitioned by world stem so the same seed against two different
worlds never clobbers itself:

```
results/<world_stem>/<label>/
├── <seed>.json          # 7-field metrics, one object per episode
├── <seed>.trace.jsonl   # per-step trace (only written on planning success)
└── _manifest.json       # provenance receipt (run_experiment only)
```

`<world_stem>` is `Path(--world).stem`, so `arena/arena_v1.yaml` →
`results/arena_v1/`. `<label>` is the algorithm key, with the replan cadence or
prediction horizon folded in (`a_star_replan_k5`, `d_star_lite_oracle_h10`).
`results/` is gitignored except for `.gitkeep`.

**Metrics JSON** (`<seed>.json`) — 7 fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `time_to_goal` | `float \| null` | Sim seconds to reach the goal; null on crash, timeout, or planner error. |
| `crashed` | `bool` | irsim collision flag. |
| `timed_out` | `bool` | `sim_time >= 120.0` without reaching the goal. |
| `path_length` | `float` | Σ of per-step XY displacement over the executed trajectory. |
| `mean_speed` | `float` | `path_length / sim_time`. |
| `wallclock_per_step` | `float` | Mean wall-clock per step (a `perf_counter` mean; not byte-deterministic). |
| `planner_error` | `str \| null` | Exception message if the t=0 plan in `reset()` raised, else null. |

**Trace JSONL** (`<seed>.trace.jsonl`) — one JSON object per line, keys sorted:
`step`, `state` `[x, y, θ]`, `action` `[v, ω]`, `lidar_sha256`, `crashed`,
`reached_goal`, `done`. With traffic on, an 8th key
`dynamic_obstacles_sha256` is added per line. Step 0 records the post-reset
state with a sentinel `action=[0.0, 0.0]`.

**Manifest** (`_manifest.json`) — `master_seed`, `num_seeds`,
`derived_seeds`, per-episode `{seed, exit_code, status}` in derivation order,
the cadence/horizon/speed-regime provenance, and a best-effort `git_sha`. No
timestamps, so it is byte-reproducible.

**Degenerate-seed sidecar** (`_seed_filter.json`, world-level — a sibling of
the label dirs, written by `runners/filter_seeds.py`) — records which seeds
had every required planner crash within the first 4 s of sim time. `plot.py`
drops those seeds from the headline `failure_rate` and the numeric
per-algorithm stats (a CONDITIONAL rate, on non-degenerate seeds only);
`summary.csv` always keeps the pre-drop `failure_rate_raw` and `n_dropped`
alongside it, so the excluded seeds stay auditable. See "The degenerate-seed
filter (issue #9)" in `CLAUDE.md` for the full criterion and the freshness
contract that guards it.

---

## Determinism

The harness is built so the same seed always produces the same bytes:

- Same seed → **byte-identical** `<seed>.trace.jsonl` across runs.
- Two same-master-seed `run_experiment` runs → byte-identical per-seed JSON
  and `_manifest.json`.
- A `--jobs N` run keeps the manifest in derivation order (completion order
  never leaks into the output).

The one exception is `wallclock_per_step`, a real-time `perf_counter` mean that
cannot be byte-identical across two live runs.

Traffic substreams are derived from the master seed via
`SeedSequence.spawn(2)` (`traffic_rng` for spawning, `motion_rng` reserved for
future motion noise), drawn in a fixed order per spawn attempt. The sampling
(RRT/RRT*) planners and the lidar motion estimator are likewise fully
deterministic (a fixed-seed generator / sorted-order reductions), so their
traces are byte-stable too.

---

## Adding a planner

Planners live in `planners/<name>.py` and satisfy the `Controller` protocol in
`planners/_types.py`:

```python
class Controller(Protocol):
    name: str  # the FAMILY name, e.g. "a_star_replan"; the results label adds _k<K>

    def reset(self, world_yaml, initial_snapshot, lidar0, state0) -> None: ...
    def act(self, state, lidar) -> np.ndarray: ...  # (2,1) float [[v],[w]]

    # Optional opt-in truth seam (the predictive oracle uses it):
    wants_truth: bool = False
    def observe_truth(self, snapshot) -> None: ...  # called before act() only when wants_truth
```

`reset()` builds the static substrate and the t=0 plan (raise `ValueError` /
`RuntimeError` to surface a no-path as `planner_error`); `act()` returns the next
`(2,1)` action and must not raise on a mid-episode replan failure. The runner
calls `reset()` once, then `act()` until the Arena reports done.

Register the class by self-registering into the `ALGORITHMS` registry: the
controller module calls `register(name, cls)` from `planners/_grid.py` at import
(see `a_star.py`), and importing the `planners` package populates the registry.
The runner builds the instance via `build_controller`. Seventeen keys ship today
— thirteen canonical (`a_star_once`, `a_star_replan`, `dijkstra_once`,
`dijkstra_replan`, `d_star_lite`, `d_star_lite_predictive`, `dwa`, `dwa_predictive`,
`apf`, `rrt_once`, `rrt_replan`, `rrt_star_once`, `rrt_star_replan`) and four
experimental, motion-aware keys (`d_star_lite_oracle`, `dwa_predictive_oracle`,
`dwa_predictive_paper`, `dwa_predictive_paper_oracle`) held out of the canonical
scatter via `EXPERIMENTAL_KEYS`. The `_replan` families take a required
`--replan-k`; the predictive family takes a required `--predict-horizon` (the
canonical `d_star_lite_predictive` and `dwa_predictive` both run at h10).

---

## World YAML schema

All scripts consume the irsim world format. The fields the scripts rely on:

- `world.width`, `world.height`, optional `world.offset` (sizes the occupancy grid)
- `robot.shape.radius` (obstacles are inflated by this + a safety margin)
- `robot.state` = `[x, y, theta]` start pose; `robot.goal` = `[x, y, theta]` goal pose
- `robot.sensors` — a `lidar2d` entry (required by `manual_obstacle.py` and by `Arena`)
- `obstacle[]` with `shape.name` in `{circle, rectangle, polygon, linestring}`

irsim is strict about field shapes — when adding a world, copy an existing one
as the template. World fixtures that live in the repo go in `tests/`; scratch
worlds belong outside the repo or under the gitignored `_tmp_*` prefix.

---

## Project status

Following the phase plan in `Mission.md`:

| Phase | Status | Deliverable |
| --- | --- | --- |
| 0 — Arena | done | `arena/arena.py` + `arena_v1.yaml` |
| 1 — Harness sanity check | done | `runners/run_episode.py` + metrics/trace |
| 2 — Dynamic obstacles | done | `arena/dynamic.py` crossing traffic |
| 3 — Reproducibility | done | `runners/run_experiment.py` + manifest |
| 4 — Metrics | done | per-algorithm aggregation (lands in the plotter's loader) |
| 5 — Scatter plot | done | `runners/plot.py` + `runners/run_all.py` |
| 6 — Algorithms | done | `planners/` — grid (A*/Dijkstra once+replan), incremental D* Lite, reactive (DWA/APF), sampling (RRT/RRT*); only the 6b K-sweep is deferred |
| 7 — The actual question | in progress | the insight the plot produces, plus the experimental motion-aware D* Lite family (oracle confirms motion-awareness helps; the v1 lidar estimator does not yet) |

Phase-by-phase implementation notes and findings live in `docs/plans/`.
Per-phase architecture and conventions are documented in `CLAUDE.md`.
