# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A small sandbox of differential-drive path-planning demos built on top of [`irsim`](https://github.com/hanruihua/ir-sim) (2D robot simulator). Each top-level Python script is a self-contained controller experiment — they don't share modules with each other. World layouts live in YAML files at the repo root, A* edge-case fixtures live in `tests/`.

## Environment & commands

Windows + PowerShell. A `.venv/` is already provisioned at the repo root.

```powershell
# Activate the venv before running anything
.venv\Scripts\Activate.ps1

# Install / refresh dependencies
pip install -r requirements.txt

# Run the demos (each opens an irsim render window)
python test.py                          # minimal "irsim hello world" using robot_world.yaml
python manual.py                        # naive go-to-goal on obstacle.yaml
python manual_obstacle.py               # reactive lidar avoidance on obstacle_harder.yaml
python manual_astar.py                  # default: A* + waypoint follower on obstacle_harder.yaml
python manual_astar.py tests\no_path.yaml   # planner on a specific world (positional arg)
```

There is no test runner, linter, or build step configured. The `tests/` directory currently contains A* world fixtures (`blocked_start.yaml`, `no_path.yaml`, `partial_block.yaml`) used by hand against `manual_astar.py` — they are inputs, not pytest files.

## World YAML schema

All scripts consume the same irsim world format. The fields the scripts actually rely on:

- `world.width`, `world.height`, optional `world.offset` (planner reads these to size its occupancy grid)
- `robot.shape.radius` (planner inflates obstacles by this + a safety margin)
- `robot.state` = `[x, y, theta]` start pose
- `robot.goal` = `[x, y, theta]` goal pose
- `robot.sensors` — only `manual_obstacle.py` requires a `lidar2d` entry (see `obstacle_harder.yaml` for the canonical config)
- `obstacle[]` with `shape.name` in `{circle, rectangle, polygon, linestring}`. Polygons/linestrings can carry a `state` pose that the planner applies as a rotate+translate.

When adding a new world, copy an existing one as the template — irsim is strict about field shapes.

## The three controllers, at a glance

1. **`manual.py`** — pure proportional go-to-goal: heading error → angular velocity, constant linear velocity. No obstacle awareness; only works on `obstacle.yaml` where the start pose is already clear of the central blocker.

2. **`manual_obstacle.py`** — reactive lidar avoider. Reads `robot.get_lidar_scan()`, computes a repulsive turn from close-range returns plus a side-bias term from left-vs-right mean clearance. Single `action()` function dispatches on `closest_forward_distance` thresholds (escape / side-bias / slow / caution / cruise / turning). All tunables are module-level constants at the top of the file.

3. **`manual_astar.py`** — the substantive script. Global planner pipeline:
   - `load_world()` parses the YAML into a `WorldModel` (frozen dataclass) with normalized obstacle specs (circle / rectangle / polygon / linestring → `ObstacleSpec`).
   - `build_occupancy_grid()` rasterizes the world at `GRID_RESOLUTION` (0.1 m), marking any cell within `robot_radius + SAFETY_MARGIN` of any obstacle as blocked. Uses analytic distance per obstacle kind (`point_to_obstacle_distance`).
   - `astar_search()` runs 8-connected A* with octile-distance step cost and Euclidean heuristic; diagonal moves are blocked if either orthogonal neighbor is occupied (no corner-cutting).
   - `path_to_waypoints()` collapses the dense grid path into a sparse waypoint list by sampling at `WAYPOINT_STRIDE`, then recursively bisecting any segment that fails an inflation-aware line-of-sight check (`segment_is_clear`). This is the key non-obvious step — it turns the staircase grid path into a small set of safe waypoints.
   - `WaypointFollower` + `compute_action()` advance the waypoint index when within `WAYPOINT_REACHED_DISTANCE`, then apply a heading-gated speed schedule (full speed only when heading error is small).

   Tuning knobs are the `UPPER_SNAKE_CASE` constants at the top of the file — change those rather than threading parameters through call sites.

## The arena harness (Phase 0)

`arena/` is a reusable seeded 50×50 test environment wrapping irsim, intended as the shared substrate for every planner experiment in Mission.md. Phase 0 contains static obstacles only; dynamic traffic plugs in at Phase 2 behind the `initial_dynamic_snapshot` seam.

**API:**
- `Arena(yaml_path, seed, render=False, timeout_s=120.0)` — construct; validates lidar config at init time.
- `reset() -> (state, lidar, info)` — returns `state` as `np.ndarray` shape `(3,)` (x, y, theta), `lidar` shape `(360,)` float64 (NaN = no return), and an `EpisodeInfo` frozen dataclass.
- `step(action) -> (state, lidar, done, info)` — `action` is `np.ndarray([[v],[w]], dtype=float)` shape `(2,1)`; raises `ValueError` on bad input, `RuntimeError` if called after `done`.
- `arena.close()` — tears down the irsim env. Always call in a `finally` block.
- `arena.initial_dynamic_snapshot` — returns `()` in Phase 0; Phase 2 narrows the type.

**Smoke and verification:**
```powershell
.venv\Scripts\Activate.ps1
python arena/arena.py arena/arena_v1.yaml --check     # 55 PASS = harness healthy (TC1-TC52 + TC-CLI/TC-FWD; estimate ~50 min, dominated by the full-episode traffic-on and --no-traffic solve subprocess TCs)
python arena/arena.py arena/arena_v1.yaml --render    # visible smoke loop (use to eyeball YAML)
```

`arena/arena_v1.yaml` is the canonical world: 50×50, robot start (2,2) → goal (48,48), two staggered length-30 rectangle walls + 12 circle pillars (14 obstacles total).

Phase 2 will plug `DynamicObstacle` / `TrafficSpawner` behind `Arena.initial_dynamic_snapshot` and consume the already-plumbed `traffic_rng` / `motion_rng` from `__init__`. Do not add dynamic obstacle code to Phase 0.

## The episode runner (Phase 1)

`runners/run_episode.py` is the harness entry point that wires a planner to the Arena and records per-episode metrics and a step-by-step trace.

**Run command:**
```powershell
.venv\Scripts\Activate.ps1
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml
```

Optional flags: `--render` (opens the irsim render window) and `--results-dir <dir>` (overrides the default `results/` output directory).

**Results layout:**
- `results/<world_stem>/<algorithm>/<seed>.json` — per-episode metrics (one JSON object).
- `results/<world_stem>/<algorithm>/<seed>.trace.jsonl` — per-step trace (one JSON object per line, keys sorted); written only if planning succeeded (i.e., `planner_error` is null).
- `<world_stem> = Path(args.world).stem` (so `arena/arena_v1.yaml` → `arena_v1/`); prevents same-seed runs against different YAMLs from clobbering each other.
- `results/` is gitignored except for `.gitkeep`.

**Metrics JSON schema** (7 fields — extends Mission.md Phase 1's original 6-field list by adding `planner_error`):
- `time_to_goal: float | null` — sim seconds to reach goal on success; null on crash, timeout, or planner error.
- `crashed: bool` — irsim collision flag.
- `timed_out: bool` — sim_time >= 120.0 without reaching goal.
- `path_length: float` — Σ ‖state[t+1][:2] − state[t][:2]‖ over the executed trajectory.
- `mean_speed: float` — path_length / sim_time.
- `wallclock_per_step: float` — mean of `EpisodeInfo.wallclock_per_step` across all steps; NOT byte-deterministic across real-time runs (perf_counter mean).
- `planner_error: str | null` — exception message if `plan()` raised, else null.

**Trace JSONL schema** (one JSON object per line, keys sorted):
- `step: int`, `state: [x, y, θ]`, `action: [v, ω]`, `lidar_sha256: str` (SHA256 hex of `lidar.tobytes()`), `crashed: bool`, `reached_goal: bool`, `done: bool`.
- Step 0 records the post-reset state with `action=[0.0, 0.0]` as a sentinel; subsequent steps record state AFTER each `arena.step(action)`.

**Determinism guarantee:** same seed → byte-identical `<seed>.trace.jsonl` files across runs. Metrics JSON is equal in every field EXCEPT `wallclock_per_step`, which is a `perf_counter` mean and cannot be byte-identical across two real-time runs.

**TC13–TC16** (added to `python arena/arena.py arena/arena_v1.yaml --check`):
- TC13: scripted wall-crash via teleport — proves irsim's `collision_flag` fires on a rectangle wall.
- TC14: full A* drive through the runner (subprocess) + trace-line schema audit — verifies all 7 trace fields are present and typed correctly.
- TC15: byte-identical trace JSONL across two seeded subprocess runs — verifies the determinism guarantee end-to-end.
- TC16: planner-failure path on `arena/arena_no_path.yaml` — verifies that a sealed-box world causes A* to raise and that `planner_error` is populated and `trace.jsonl` is not written.

**`arena/arena_no_path.yaml` fixture:** An Arena-compatible world where the robot **start** `(2,2)` is walled in by a 1.5 m box of four rectangles (the goal `(48,48)` is open) so A* cannot find a path (used by TC16, and as the fast-failure world for Phase 3's TC26). The legacy `tests/no_path.yaml` cannot substitute here because it lacks the `lidar2d` sensor block that `Arena.__init__` requires.

## The traffic harness (Phase 2)

`arena/dynamic.py` adds Mission.md's crossing-traffic substrate. `Arena(..., traffic=True)` instantiates a `TrafficSpawner` that maintains a ~20-obstacle population of straight-line, edge-spawned, uniformly-on-perimeter-distributed dynamic obstacles. Each obstacle is a circle (r=0.3 m) registered into irsim via `env.create_obstacle({'name':'omni'}, ...) + env.add_object`, so lidar and `robot.collision_flag` see them natively — no custom collision code. Traffic runs pass `log_level="ERROR"` to `irsim.make` to mute the per-tick `Behavior not defined` omni warning irsim emits for every obstacle.

**API:**
- `Arena(yaml, seed, traffic=True, ...)` — opt-in flag; default `False` for Phase 0/1 compatibility.
- `arena.initial_dynamic_snapshot` — returns `tuple[DynamicObstacleState, ...]` (length 20 after `reset()` when `traffic=True`; `()` pre-reset or when `traffic=False`). `DynamicObstacleState` is a frozen dataclass with fields `(id, x, y, vx, vy, radius)`.
- `EpisodeInfo.dynamic_obstacles_sha256: str | None` — per-tick deterministic hash of the obstacle `(x, y, vx, vy, radius)` matrix, rows ordered by id. The irsim object id itself is excluded from the hash so the digest is reproducible across repeated `reset()` on one Arena (`id_iter` resets per `make()`, not per `reset()`). Used by the determinism TCs.
- `EpisodeInfo.dynamic_obstacle_count: int` — population each tick (Phase 0/1: always 0; Phase 2: 20).

**Determinism guarantees:**
- `traffic_rng` (derived from master seed via `SeedSequence.spawn(2)`) draws in a fixed order per spawn attempt: perimeter position → heading → speed; ALL THREE re-drawn on overlap rejection.
- `motion_rng` is plumbed but never drawn from in Phase 2 (forward-compat for Phase 2b motion noise).
- Two `Arena(seed=K, traffic=True)` runs produce byte-identical `dynamic_obstacles_sha256` sequences over identical action streams — whether two fresh instances or repeated `reset()` on one instance (the hash excludes the per-episode object id).

**Runner default:**
- `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` — traffic ON by default. A* `_once` planners do not dodge, so most seeds end in collision; that is the experimental signal Mission.md's scatter plot consumes.
- Pass `--no-traffic` to reproduce Phase 1's deterministic A* success path; the trace JSONL stays 7 keys per line.
- With traffic on, the trace JSONL gains an 8th key `dynamic_obstacles_sha256` per step (step-0 line uses the reset-time hash; subsequent lines use the post-step hash).

**Results layout:**
- `results/<world_stem>/<algorithm>/<seed>.{json,trace.jsonl}` — runner output. World-stem partitioning means same-seed runs on `arena_v1.yaml` and `arena_v2_hard.yaml` do not overwrite each other.

**TC17–TC24** (added to `python arena/arena.py arena/arena_v1.yaml --check`):
- TC17: init population of 20, every spawn on a perimeter edge with inward heading.
- TC18: refill maintains population at 20 across a full-traversal window (verifies the despawn/respawn cycle).
- TC19: robot-vs-dynamic-obstacle collision fires `info.crashed` via `_inject_for_test`.
- TC20: two same-seed runs produce identical `dynamic_obstacles_sha256` sequences (per-tick).
- TC21: `initial_dynamic_snapshot` is a tuple of frozen `DynamicObstacleState` of length 20; mutation raises `FrozenInstanceError`.
- TC22: world-stem partitioning — same seed against two different YAMLs produces two distinct result files; neither clobbers the other.
- TC23: subprocess import-cycle guard — `import planners; import arena.arena` and the reverse both exit 0.
- TC24: traffic-ON runner end-to-end — every trace line carries the 8th `dynamic_obstacles_sha256` key, and two same-seed `--traffic` runs produce byte-identical trace JSONL (trace-level determinism through the runner). Covers the shipped default path, which the other runner TCs force `--no-traffic` to avoid.

`arena/arena_v2_hard.yaml` is a second 50×50 world (same robot start/goal/lidar as arena_v1, but walls relocated) used by TC22 to cross-check the partitioning. It otherwise has no special semantics in Phase 2.

## The batch experiment runner (Phase 3)

`runners/run_experiment.py` runs ONE algorithm against the canonical 50 seeds so every algorithm in Mission.md faces the same 50 traffic streams (what makes the cross-algorithm scatter plot meaningful). It derives the seeds from a single master seed and shells out to the already-deterministic single-episode runner once per seed (one fresh-irsim subprocess each), so per-episode determinism and the `SeedSequence.spawn(2)` traffic/motion substreams carry over unchanged.

**Run command:**
```powershell
.venv\Scripts\Activate.ps1
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml
# default: master-seed 20260605, 50 seeds, traffic ON, jobs 1 (sequential)
# writes results/arena_v1/a_star_once/<seed>.{json,trace.jsonl} x50 + _manifest.json
```

**Seed derivation:** `derive_episode_seeds(master, n)` = `SeedSequence(master).spawn(n)`, each child's first two uint32 words packed into a 64-bit int used as that episode's `--seed`. Prefix-stable (`spawn(3) == spawn(50)[:3]`), so `--num-seeds` selects a prefix of the canonical stream; uniqueness-asserted (64-bit width avoids the silent same-filename collision a 32-bit seed would risk).

**Flags:**
- `--master-seed N` (default 20260605), `--num-seeds N` (default 50).
- `--jobs N` — sequential at 1 (default); N>1 runs up to N child subprocesses concurrently via a `ThreadPoolExecutor` over `subprocess.run` (threads, NOT multiprocessing — the Windows spawn/pickle path never enters). Each seed is isolated, so trace JSONL and the manifest are byte-identical at any `--jobs`; the metrics JSON matches too except `wallclock_per_step` (a Mission.md "freebie"), a `perf_counter` mean that contention perturbs, so produce headline wallclock numbers with `--jobs 1`.
- `--resume` skips seeds whose `<seed>.json` already exists (default: overwrite).
- `--traffic` / `--no-traffic` forwarded to each episode (default ON).

**Failure policy:** a child exit of 0 includes in-sim crashes, timeouts, and planner failures (those are recorded inside the metrics JSON, not the exit code). Only a non-zero child exit (a runner/config fault — e.g. a malformed world) is a "runner failure": the batch continues past it, lists it in the end summary, and itself exits non-zero if any seed failed.

**Outputs:** per-seed `results/<world_stem>/<algorithm>/<seed>.{json,trace.jsonl}` (identical to the single-episode runner) plus a deterministic provenance receipt `_manifest.json` in the same dir (`master_seed`, `num_seeds`, `derived_seeds`, per-episode `{seed, exit_code, status}` in derivation order, best-effort `git_sha`; no timestamp/elapsed). Phase 5's `plot.py` must select episode files by numeric stem (e.g. glob `[0-9]*.json`) so it skips `_manifest.json`.

**TC25–TC27** (added to `python arena/arena.py arena/arena_v1.yaml --check`):
- TC25: seed derivation — determinism, 64-bit uniqueness, prefix property, master-sensitivity (pure computation).
- TC26: batch determinism + parallel-ordering — two same-master-seed `--jobs 1` runs produce byte-identical per-seed JSON and manifest; a `--jobs 3` run keeps the manifest in derivation order (completion order must not leak). Uses `arena_no_path.yaml` so each episode fails fast.
- TC27: failure accounting — a malformed (but existing) world makes every child exit non-zero; the batch reports the failures and itself exits non-zero.

## The planner family (Phase 6)

`planners/` holds the pluggable controllers. Phase 6 shipped the unified interface, the grid family, D* Lite, the reactive (DWA, APF) family, and the sampling (RRT, RRT*) family. The registry now holds 11 keys: `a_star_once`, `a_star_replan`, `dijkstra_once`, `dijkstra_replan`, `d_star_lite`, `dwa`, `apf`, `rrt_once`, `rrt_replan`, `rrt_star_once`, `rrt_star_replan`. Only the Phase 6b K-sweep remains deferred.

**The `Controller` interface** (`planners/_types.py`): a `name` attribute (the FAMILY name, e.g. `a_star_replan`), `reset(world_yaml, initial_snapshot, lidar0, state0) -> None` (build the static substrate and the t=0 plan; may raise `ValueError`/`RuntimeError`, which the runner records as `planner_error`), and `act(state, lidar) -> (2,1) action`. `run_episode.py` is now planner-agnostic: it calls `reset()` once at t=0, then `while not done: act()`. A mid-episode replan that fails inside `act()` must not raise (the controller keeps its last valid path), so only a t=0 plan failure yields `planner_error`.

**The registry** (`planners/_grid.py`): controller modules self-register into `ALGORITHMS` at import (via `register(name, cls)`); importing the `planners` package is what populates it. `build_controller(name, replan_k)` validates the pair and constructs the instance; `algorithm_label(name, replan_k)` returns the results-dir label.

**The grid planners shipped**: `a_star_once`, `a_star_replan`, `dijkstra_once`, `dijkstra_replan`. Dijkstra is A* with a zero heuristic (`heuristic_fn = staticmethod(lambda *_: 0.0)`), so it reuses the same `astar_search` and grid machinery — only the heuristic differs. The `_once` controllers plan once on the STATIC occupancy grid (analytic line-of-sight pipeline from `manual_astar`, no lidar fold) and follow that path forever; the `_replan` controllers (`PathFollowingController`) re-search the lidar-folded grid every K acts. `d_star_lite` also ships as the incremental planner (see **D* Lite** below); it is not a `_once`/`_replan` family.

**`--replan-k`**: required for the `_replan` families (`a_star_replan`, `dijkstra_replan` — the `REPLAN_FAMILIES` set), rejected for `_once` and `d_star_lite`. Results land in `results/<world_stem>/<family>_k<K>/` (e.g. `a_star_replan_k5/`); `algorithm_label` folds the cadence into the label so different K values do not collide. `run_experiment` forwards `--replan-k` to each child episode and records it in `_manifest.json` as `replan_k`.

**The lidar->grid fold** (`lidar_to_occupancy`): memoryless — it folds the current lidar frame onto a COPY of the static grid each time (no accumulation across replans; the static cells are never mutated). After t=0 the replanners are lidar-only (Mission-faithful: `initial_snapshot` is ignored by design because lidar0 already encodes those obstacles). Beam bearings are recovered as `np.linspace(angle_min, angle_max, number)` from the YAML `lidar2d` sensor block, mirroring how irsim lays the beams. A replan re-searches from the robot's CURRENT cell to the goal; a failed mid-episode replan is swallowed and the last valid path is kept, so only the t=0 plan failing produces `planner_error`.

**The `_replan` families' follower commitment (resolved).** `PathFollowingController` used to rebuild the `WaypointFollower` on every K-th act unconditionally, so `a_star_replan` / `dijkstra_replan` could not cleanly traverse even the static, traffic-free world: at frequent K (5, 25) the re-extracted waypoints jittered one or two cells per replan and the heading-gated speed schedule starved forward motion into a timeout; at infrequent K (100) the robot committed to a stale waypoint segment long enough to drive into a static wall (collision). The fix ported `DStarLiteController`'s commitment horizon into `PathFollowingController.act()`: it keeps re-searching every K for fresh knowledge but swaps the follower only when the follower is finished or its immediate committed segment (robot -> current target waypoint) is blocked in the last fold. With it, `a_star_replan` (~85.8 s) and `dijkstra_replan` (~85.7 s) now reach the arena_v1 `--no-traffic` goal, and `rrt_replan` / `rrt_star_replan` traverse too. The commitment-horizon logic now lives in TWO places (`PathFollowingController` and `DStarLiteController`); that duplication is acknowledged and tracked in a GitHub issue for consolidation (alongside stronger-follower alternatives), not resolved here.

**The reactive family** (`dwa`, `apf`): velocity output, no global plan, so `reset()` never raises `planner_error` (it loads the goal and the lidar beam geometry from the YAML and stores them). Both reject `--replan-k`. They are expected to stall or crash in arena_v1's corridors (that is the experimental signal), so their TCs do not require reaching the goal.
- `dwa` (`DWAController`, `planners/dwa.py`): Dynamic Window Approach. Samples an acceleration-bounded window of `(v, ω)`, forward-simulates each candidate over a short rollout, scores by goal heading + lidar clearance + speed, and drives the best feasible command. The collision band adds the robot radius to the lidar return (lidar is center-to-surface in this harness), so candidates that would clip the body are rejected.
- `apf` (`APFController`, `planners/apf.py`): Khatib 1986 artificial potential fields. An attractive pull to the goal plus a repulsive push from live lidar returns within an influence radius, combined into a clamped `(v, ω)`.

**The sampling family** (`planners/rrt.py`, `planners/rrt_star.py`): RRT and RRT*, each with a `_once` and a `_replan` variant. Hand-rolled and grown from a single numpy `Generator` so traces stay byte-identical. `_once` plans on the static grid with `default_rng(RRT_SEED)`; `_replan` re-grows on the lidar fold every K acts via the `PathFollowingController._plan` hook with `default_rng(RRT_SEED + self._k)`, so successive replans explore fresh samples yet stay deterministic. `rrt_replan` / `rrt_star_replan` are in `REPLAN_FAMILIES` (require `--replan-k`); `rrt_once` / `rrt_star_once` reject it. At `RRT_SEED=5`, `rrt_once` reaches the arena_v1 `--no-traffic` goal at ~73.0 s and `rrt_star_once` at ~70.7 s. RRT* adds choose-parent + rewire; node positions match RRT for a given seed (only parent pointers and costs change), and at seed 5 rewiring cuts the planned cost from 78.0 m to 70.9 m. `rrt_points_to_waypoints` shortcuts the continuous tree path with the same line-of-sight bisection the grid planners use.

**The sampling family's collision-LOS speedup (issue #10).** The per-edge collision check in `rrt_plan` / `rrt_star_plan` (and RRT*'s choose-parent + rewire) routes through `_segment_clear_fast` — an allocation-free scalar line-of-sight helper defined in `rrt.py` and imported by `rrt_star.py`. It reproduces the frozen `_grid.segment_is_clear_grid`'s exact accept/reject bool but drops the per-sample numpy boxing (`world_to_grid`'s `np.clip`/`np.floor`/`np.asarray`): it uses `math.sqrt(dx*dx+dy*dy)` for length (NOT `math.hypot`, which flips the sample count on ~17% of inputs), `math.ceil` for the sample count, and a `min/max`+`math.floor` clamp that clip-then-reads the clamped cell with NO out-of-bounds rejection (matching that `world_to_grid` always clips). Both planners also keep an incremental preallocated node-position buffer (`_nearest_index_in_array`, plus a buffer-typed `_near_node_indices`) instead of rebuilding `np.asarray(nodes)` per iteration. The pair keeps every trace byte-identical (node positions, parent structure, and planned cost are unchanged — guarded permanently by TC47) while cutting the planner grow time ~8.5x (RRT) / ~13.2x (RRT*), so `rrt_replan` / `rrt_star_replan` now reach a terminal sim state within the per-episode wall on replan-heavy seeds that previously blew it under ~20-obstacle traffic. CAVEAT: the runner's `wallclock_per_step` metric times only `Arena.step` (irsim), NOT the planner's `act()`, so it does NOT reflect this speedup — the gain shows in total episode wall time and the timeout rescue. Making the runner time `act()` (or adding a planner-time metric) is a deferred follow-up, outside this change's rrt/rrt_star-only scope.

**D* Lite** (`planners/d_star_lite.py`, `DStarLiteController`): the incremental planner. ONE registry entry, no `_once`/`_replan` split (Mission.md: D* Lite is inherently incremental), and it REJECTS `--replan-k` — it is not a replan family, so it is not in `REPLAN_FAMILIES`. Every act it does the cheap edge-cost BOOKKEEPING — fold the live lidar onto a copy of the static grid, diff that against the working occupancy to get the CHANGED cells, mutate `self._cells` in place at those positions, `move_start(current_cell)` (unconditionally, O(1)), and `update_cells(changed)` when cells flipped. But the expensive tree settle (`compute_shortest_path` + `extract_path` + follower rebuild) is DEFERRED to the moment a fresh path is actually needed: when the waypoint follower is finished OR its immediate segment (robot -> current target waypoint) is no longer clear in the live fold. That deferral is exactly what D* Lite's `k_m` machinery exists to support — `move_start` accumulates the heuristic drift into `k_m` so stored keys stay comparable across many batched `update_cells`, and a single settle at demand-time folds all of those batched edge changes into the same optimum a from-scratch A* would find (proved by TC46). The perf motivation: the repaired `g`/`rhs` tree is only consumed at re-extraction (rare on a clear run), so settling per tick — ~89% of `act()`'s wallclock under ~20-obstacle traffic — was pure waste, and it blew the 600 s per-episode wallclock wall on 9 of 50 batch episodes. The search core (`DStarLiteSearch`) is a hand-rolled optimized (`k_m`-based) Koenig-Likhachev D* Lite over the boolean grid in pure cell space — internally a flat, padded grid (a permanently-occupied border ring makes out-of-bounds moves cost inf with no bounds check), flat `g`/`rhs` lists of native floats, 4-tuple heap entries, and an occupancy mirror — with the SAME cost model as `astar_search` (8-connected, octile step cost, no corner-cutting), so it recovers the same optimal path cost a fresh A* would. The load-bearing invariant is grid ownership: the search holds a REFERENCE to `self._cells`, and `act()` mutates that array in place at the flipped positions rather than rebinding it to the freshly folded array (rebinding would detach the search's view and desync both its occupancy mirror and its incremental costs). The occupancy mirror is re-synced ONLY inside `update_cells` (it re-reads each reported flip from the live ndarray), so the report-every-flip contract is now load-bearing for occupancy correctness, not just for the incremental invariants — a flip the caller never reports is never seen. The commitment horizon also avoids the heading whipsaw a per-tick follower rebuild causes — the inflation band repaints each tick, jittering the cell path one or two cells even on a static map, which starves forward speed and times the robot out. A mid-episode settle/extraction failure keeps the last valid follower (never rebuilt), so `act()` never raises; only the t=0 plan in `reset()` surfaces as `planner_error`.

**TC28–TC37** (added to `python arena/arena.py arena/arena_v1.yaml --check`):
- TC28: lidar->grid fold geometry — pose-dependent and memoryless (a finite beam blocks its hit cell, far cells stay free, an all-NaN scan reproduces the static grid, and the fold returns a new array without mutating the static cells).
- TC29: Dijkstra == A* optimal cost, and `dijkstra_once` reaches the goal through the runner.
- TC30: `a_star_replan` end-to-end (subprocess) — writes to the `a_star_replan_k5` labeled dir and every traffic-on trace line carries the 8-key schema.
- TC31: replan cadence — `compute_path` fires only on every K-th act, and each fold is memoryless (no obstacle leaks across replans).
- TC32: mid-replan failure fallback — a replan that raises does not propagate out of `act()`, and the existing follower object is kept (not rebuilt).
- TC33: `--replan-k` validation — required/forbidden per family, plus `name == registry key`, the `_k<K>` label, and `ALGORITHMS` membership.
- TC34: `a_star_once` parity through the new planner-agnostic loop — two same-seed `--no-traffic` runs produce byte-identical trace JSONL.
- TC35: D* Lite optimal static path (== A* cost) + reaches goal — the search recovers the same octile cost A* does, and `d_star_lite` drives the static map to the goal through the runner (subprocess).
- TC36: D* Lite incremental == from-scratch (binding block) — block a cell on the optimal path, then the incremental recomputed cost equals a fresh-A* oracle AND strictly increased (the binding incremental-correctness test).
- TC37: `d_star_lite` registered + rejects `--replan-k` + traffic e2e — it is a key in `ALGORITHMS`, `build_controller('d_star_lite', 5)` raises and `--replan-k` exits 2, and a traffic-on subprocess drive plans at t=0 with the 8-key trace schema per line.

**TC38–TC47** (added to `python arena/arena.py arena/arena_v1.yaml --check`):
- TC38: `dwa` traffic-on drive via runner (subprocess) — runs to a terminal state and every trace line carries the 8-key schema (goal-reaching not required).
- TC39: `apf` traffic-on drive via runner (subprocess) — runs to completion with the 8-key trace schema per line (goal-reaching not required).
- TC40: `rrt_once --no-traffic` on arena_v1 — `time_to_goal` is non-null within the recorded margin, and two same-seed runs produce byte-identical trace JSONL.
- TC41: `rrt_star_once --no-traffic` on arena_v1 — reaches the goal (blocking), plus a non-blocking in-process observation of `rrt_star_once`'s planned cost versus `rrt_once`'s (the RRT*-vs-RRT comparison; no `≤` assertion).
- TC42: `rrt_once` & `rrt_star_once` on `arena_no_path.yaml` — the sealed start makes each raise a no-path error, so `planner_error` is populated and no `trace.jsonl` is written.
- TC43: `--replan-k` validation for the 6 new keys — `dwa` / `apf` / `rrt_once` / `rrt_star_once` reject `-k`; `rrt_replan` / `rrt_star_replan` require it; `name == registry key`; the `_k<K>` label folds in; all are in `ALGORITHMS`.
- TC44: `rrt_replan` & `rrt_star_replan` traffic-on via runner — write to the `rrt_replan_k5` and `rrt_star_replan_k5` labeled dirs, exit 0, and emit the 8-key trace schema per line.
- TC45: commitment-horizon fix proof (binding gate) — `a_star_replan` and `dijkstra_replan` reach the arena_v1 `--no-traffic` goal, and the follower object identity is preserved across at least one replan tick on the clear run (proving the commitment actually held, not merely that the goal was reached).
- TC46: D* Lite deferred settle (in-process, no irsim/subprocess) — a counting spy over `compute_shortest_path` proves clear committed ticks and behind-the-robot changes never settle (yet the per-tick bookkeeping still diverges `self._cells` from the static grid), a return on the committed segment forces exactly one settle, and the deferred-batch incremental path then matches a fresh A* oracle on the same folded grid (batched update_cells + one settle == from-scratch).
- TC47: rrt-local LOS-helper equivalence (in-process, no irsim/subprocess) — a fixed-RNG stratified fuzz (OOB endpoints, sub-1e-9 degenerate, length-spread, in-bounds; >=10^5 segments on random grids with occupied cells on all four edges) asserting `_segment_clear_fast` returns the identical bool as the frozen `segment_is_clear_grid` for every segment, plus a `math.sqrt`-vs-`np.linalg.norm` length-formula guard against a future `math.hypot` swap.

## The Phase 5 plotter and batch driver

Phase 5 turns the per-episode result JSONs into the cross-algorithm comparison Mission.md asks for. Two entry points: `runners/plot.py` (read-only charting) and `runners/run_all.py` (the batch driver that produces the data the plotter reads). `matplotlib` was added to `requirements.txt` for the plotter.

**`runners/plot.py`** — a read-only plotter (`python -m runners.plot --world arena/arena_v1.yaml`). It reads ONLY the result JSONs (never irsim, never a sim): it loads every canonical algorithm's `<seed>.json` files for one world into per-algorithm summaries, writes a `summary.csv`, and renders 7 charts as PNGs into `results/<world_stem>/plots/` (gitignored, overridable with `--out-dir`). Nothing imports matplotlib until `ensure_matplotlib()` runs (the Agg headless backend), and `planners` is imported lazily, so the loader/classifier stay headless. The seven charts:
- **A1** — headline scatter: time-to-goal (x) vs failure rate (y), per-seed success dots plus a mean (star) and median (diamond) centroid per algorithm, one color per algorithm, side legend. The Mission deliverable ("down-left wins").
- **A3** — failure-breakdown stacked bars: success / crash / timeout / planner_error / DNF counts per algorithm (sum to `n_present`, expected 50).
- **A4** — time-to-goal box plot over successful times per algorithm, sorted by median ascending (0-success and single-success algorithms degrade to an annotation / lone point rather than a box).
- **B1** — seed-difficulty heatmap: 11 algorithms × 50 seeds aligned to the shared traffic-stream order (the manifest's `derived_seeds`, else sorted stems); successes shaded by time on a viridis colorbar, failures in flat categorical colors (crash / timeout / planner_error / DNF), so universally-hard seeds read as columns.
- **B2** — path-length box vs the Euclidean lower bound (`46*sqrt(2)` ≈ 65.05 m, labelled unreachable through the walls).
- **B3** — compute-cost bars: mean `wallclock_per_step` per algorithm, sourced from the serial `__wallclock__` pass; the figure footnote credits the serial pass when that subtree is present, else caveats that the bulk-pass numbers are `--jobs`-perturbed.
- **B4** — family-contrast panels (the three designed experiments): A* vs Dijkstra, once vs replan, reactive vs global, each a grouped failure-rate + median-time bar pair.

**Outcome classification & failure rate.** Each episode is classified into exactly one of success / crash / timeout / planner_error / DNF (precedence: planner_error → crash → timeout → success). `failure_rate = (crash + timeout + planner_error + dnf) / n_present`, with the denominator kept at the full seed count. **DNF** ("did-not-finish") is a fifth failure subtype that has no `<seed>.json`: an episode the batch killed at its per-episode wallclock wall, recorded in that label's `_manifest.json` as a `status="runner_error"` roster entry with no metrics JSON. When a label's manifest carries an `episodes` roster the loader treats it as the authoritative seed roster, so those killed seeds fold into the failure rate at denominator 50 rather than silently dropping. This extends Mission.md's original crash+timeout failure definition to also include planner_error and DNF.

**`--selfcheck`:** `python -m runners.plot --selfcheck` runs TC-P1..TC-P11 on synthetic JSON fixtures built in a `TemporaryDirectory` (no irsim, no real episodes): the classifier precedence, the loader over a numeric-stem tree (decoys + manifest skipped), the summary math, the partial/missing-dir robustness, all 7 chart renders, the B1 seed alignment, the matplotlib import guard, the B3 wallclock-source selection + fallback, the `run_all` canonical-set derivation, and the DNF roster. `--world` is optional when `--selfcheck` is given (the selfcheck gate runs before the `--world` requirement). Each TC runs in isolation so one failure never aborts the rest; the suite ends with an `N/11 passed` line and exits 0 only if all pass.

**`runners/run_all.py`** — the batch driver (`python -m runners.run_all --world arena/arena_v1.yaml`). It runs all 11 canonical planner labels via `run_experiment` subprocesses (one per planner, mirroring the existing two-tier subprocess pattern) in two passes:
1. A bulk pass at `--jobs N` over the full `--num-seeds` stream, writing `results/<world_stem>/<label>/` — the plotter's main input.
2. A serial wallclock mini-pass (`--jobs 1`, `--wallclock-seeds` seeds, default 5) writing `results/__wallclock__/<world_stem>/<label>/` — a clean uncontended `wallclock_per_step` that B3 reads. The children are handed `<results-dir>/__wallclock__` as their results-dir so `episode_out_dir`'s unconditional `<world_stem>/<label>` suffix lands the files where B3 looks (the stem is inserted once, never double-nested).

Replan families are forwarded `--replan-k 5` (the canonical `REPLAN_K`); `_CANONICAL_ORDER` is asserted against `ALGORITHMS` at import so a registry drift fails loud. The driver exits non-zero if any planner's batch had a runner failure (e.g. a wallclock-killed DNF seed makes its child exit non-zero), continuing past it and listing the failures at the end.

**The `results/__wallclock__/<world_stem>/<label>/` subtree** is a sibling of `<world_stem>` at the results root (NOT under the bulk world dir). It holds only the short serial wallclock pass, and `runners/plot.py`'s B3 reads `wallclock_per_step` from there; when it is absent B3 falls back to the bulk dir's wallclock with an on-figure `--jobs`-sensitivity caveat.

## The obstacle-speed-cap sweep (issue #11)

The dynamic-obstacle speed band is now a swept parameter, so the harness can measure how the obstacle-speed cap drives failure rate and time-to-goal per algorithm (the issue #11 question: does D* Lite's crash rate floor to zero once no obstacle can outrun the robot?). The plumbing is additive and default-preserving. The full 50-seed run and its verdict are the user's to launch; see `docs/plans/2026-06-23-obstacle-speed-sweep.findings.md` for how to run and read it.

**The `--speed-regime` knob:** `runners/run_episode.py` and `runners/run_experiment.py` take `--speed-regime {slow,matched,current,fast}` (default `current` = the Mission baseline), with raw `--speed-min-factor` / `--speed-max-factor` overrides for off-menu single runs. The override pair is mutually exclusive with `--speed-regime` and is both-or-neither; a bad or conflicting flag (lone min/max, unknown regime, non-positive min, max < min, regime + override) is rejected at parse time with exit 2, before any Arena is built. The four regimes are factors of robot top speed: `slow` 0.3–0.7, `matched` 0.3–1.0, `current` 0.3–1.5, `fast` 0.5–2.0. The band table, the resolver, and the shared CLI helpers (`SPEED_REGIMES`, `SPEED_REGIME_CAP`, `resolve_speed_factors`, `add_speed_args`, `resolve_speed_args`) live in the new stdlib-only `arena/speed_regimes.py` — irsim-free and matplotlib-free, so the headless plotter and its `--selfcheck` can import it.

**Determinism preserved:** `--speed-regime current` (and the default) is byte-identical to the prior baseline. The speed draw stays the 3rd `traffic_rng` draw with the same `uniform(lo, hi)` call (only the bounds become parameters), so TC17–TC24 are unchanged. `run_experiment`'s `_manifest.json` now records `speed_regime` / `speed_min_factor` / `speed_max_factor` as write-only provenance.

**The sweep driver** `runners/run_speed_sweep.py` (`python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus|all`, default `focus`): runs the canonical seeds across all 4 regimes for the selected planner set, shelling `run_experiment` once per (regime, planner) into a per-regime subtree `results/speed_<regime>/<world_stem>/<label>/` (the `__wallclock__`-style sibling convention). `focus` = `a_star_once, d_star_lite, dwa, apf`; `all` = the 11 canonical planners (imports `run_all.canonical_planner_set()`). Flags `--master-seed` / `--num-seeds` / `--jobs` / `--resume` / `--traffic` | `--no-traffic` (traffic ON by default). Mirrors `run_all`'s failure policy: continues past a child runner failure and exits non-zero if any (regime, planner) failed.

**The sweep plotter** `runners/plot_speed_sweep.py` (headless, read-only; `python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus|all`): reads the four `results/speed_<regime>/` subtrees via `plot.load_world_results` and writes `failure_rate_vs_cap.png`, `median_time_vs_cap.png`, and `speed_sweep_summary.csv` to `results/<world_stem>/speed_sweep_plots/` (x-axis = the max-cap factor 0.7/1.0/1.5/2.0, one line per present algorithm). Colors come from `plot.CANONICAL` built first then filtered to present planners, so a line keeps the same color across regimes. `python -m runners.plot_speed_sweep --selfcheck` runs 7 synthetic-fixture cases (no irsim, no real episodes) and exits 0 only if all pass; `--world` is optional when `--selfcheck` is given.

**TC48–TC52, TC-CLI, TC-FWD** (added to `python arena/arena.py arena/arena_v1.yaml --check`, growing it from 48 to 55 cases):
- TC48: regime table + resolver — `SPEED_REGIMES` is the exact 4 bands, `resolve_speed_factors("current",None,None)==(0.3,1.5)` and equals the spawner constants, both overrides return them, an unknown regime raises.
- TC49: spawner bound validation — `speed_min_factor<=0` and `speed_max_factor<speed_min_factor` each raise `ValueError`; `min==max` is allowed.
- TC50: baseline determinism preservation + draw-count guard — `Arena(traffic=True)` vs explicit `(0.3,1.5)` give byte-identical `dynamic_obstacles_sha256`, `--speed-regime current` trace == no-flag trace, `--speed-regime fast` differs, and the per-spawn `traffic_rng` draw count is asserted unchanged (3 per attempt).
- TC51: band wired + controlled-experiment property — at one seed two regimes' initial `initialize()` snapshots give identical spawn `(x,y)` and identical velocity direction per obstacle id, with only speeds scaled by the band (initial snapshot only; refills diverge once an obstacle despawns).
- TC52: non-baseline determinism across a despawn/refill — two same-seed `Arena(traffic=True, speed_min_factor=0.5, speed_max_factor=2.0)` runs give identical sha256 sequences over enough ticks to force a refill.
- TC-CLI: speed-flag CLI rejection — subprocess `run_episode` with regime+override / lone min / lone max / unknown regime / non-positive min / max < min each exits 2 and writes no `<seed>.json`.
- TC-FWD: `run_experiment` flag forwarding — the pure child-command builder emits `--speed-regime <regime>` (or the float overrides) in the child argv and the manifest carries the three provenance fields.

## The predictive D* Lite family (Phase 7 — experimental)

`planners/d_star_lite_predictive.py` and `planners/_predict.py` add a motion-aware D* Lite family that stamps each dynamic obstacle's predicted future footprint into the occupancy grid before the diff, so the planner routes behind crossing traffic rather than into where it is about to be. The family is **experimental**: neither key lands on the canonical-11 scatter, and the oracle variant is a go/no-go gate before the lidar estimator is built.

**The stamping seam (two hooks).** Prediction enters via the existing fold→diff→`update_cells` path in `DStarLiteController.act()` through two overridable hooks. The per-tick `_extra_blocked_cells(state, lidar, folded_new_cells) -> list[(row, col)]` ORs extra cells into the freshly folded array before the `diff_mask = self._cells != new_cells` line, so predicted cells reach `update_cells` and the deferred settle the same way a lidar-detected obstacle would. The settle-time `_settle_and_extract(position) -> Path | None` (the deferred `compute_shortest_path` + `extract_path` + waypoint pass) was lifted out of the inlined `act()` block into an overridable method so the predictive subclass can peel its stamp there without touching `act()`; the base method is byte-identical to the block it replaced. Both base hooks are no-ops for the plain planner (`_extra_blocked_cells` returns `[]`, `_settle_and_extract` runs the unchanged settle), so `d_star_lite` stays byte-identical to its pre-Phase-7 behavior (no search-core changes, no new invariants — TC46/TC57). `PredictiveDStarLiteController` subclasses `DStarLiteController` and overrides ONLY those two hooks; it does not reimplement `act()`, `reset()`, or the search. The grid-ownership invariant holds throughout: `self._cells` is mutated in place at the flipped positions, never rebound.

**The `_predict.py` substrate.** `planners/_predict.py`'s LOGIC is pure (plain floats + numpy in, deterministic output, no irsim/RNG calls, no set-iteration). It is NOT importable irsim-free, though: like every `planners` submodule, importing it runs the package `__init__` (which eagerly imports the controllers) and so pulls irsim — "pure" describes the computation, not the import, so headless tools still lazy-import `planners` symbols inside functions (see [[gotcha-planners-import-pulls-irsim]]). The shared grid geometry it builds on lives in the pure-logic `planners/_geometry.py`: `iter_disk_cells` (the ONE bounding-box disk-cell scan that both `_grid._mark_disk`'s lidar fold and `predict_blocked_cells`'s footprint use, so they can never drift) and `lidar_to_world_points` (the ONE finite-beam lidar→world projection shared by the fold, DWA, and the `LidarTracker`, with an optional near-rim `RANGE_MAX_DEADBAND` cut). The conflict gate's distance reuses `manual_astar.point_to_polyline_distance` (lazy-imported) rather than a private copy. It exports:
- `Track` — frozen dataclass `(id, x, y, vx, vy, radius)` in world frame.
- `Tracker` (Protocol) — `update(*, snapshot, state, lidar, dt) -> list[Track]`. Both implementations return tracks sorted by `id` for determinism.
- `OracleTracker` — reads `snapshot` (a tuple of `DynamicObstacleState` from `EpisodeInfo.dynamic_obstacles`) and ignores `state`/`lidar`/`dt`; velocities are exact from the truth seam.
- `predict_blocked_cells(tracks, planned_path, robot_xy, grid, inflation, horizon_steps, dt, *, geometry, exclusion_radius, corridor_half_width) -> list[(ThreatKey, list[(row, col)])]` — pure geometry/gate/grouping (the peel itself lives in the controller's `_settle_and_extract`, not here): for each track and lookahead step `k = 1..horizon_steps` it projects a future center and stamps a disk of radius `track.radius + inflation` (capsule) or `track.radius + inflation + CONE_GROWTH_PER_STEP * k` (cone). A track passes the **predicted-conflict gate** only if its predicted disk intersects a corridor of `corridor_half_width` around `planned_path` at some `k` — an obstacle that cannot reach the corridor over the horizon is skipped. Cells within `exclusion_radius` of the robot are removed (the robot exclusion zone). Output groups are sorted by `ThreatKey(ttc_steps, track_id)` ascending (soonest time-to-conflict first), each cell list sorted row-major. `PREDICT_DT = 0.1` s is the module constant (matches irsim `step_time`). Two calls on identical inputs return byte-identical output.
- Geometry choice: `"capsule"` uses a constant radius (correct for the oracle because obstacles move in exact straight lines); `"cone"` widens with `CONE_GROWTH_PER_STEP * k` to represent estimator uncertainty (intended for the future lidar variant).

**Per-tick stamp (cheap), settle-time fail-open peel (rare).** This split is the load-bearing perf rework, and it changed shape since the spec was first drafted: the peel is now SETTLE-TIME, not per-tick. The per-tick `_extra_blocked_cells` hook runs NO reachability search — it asks its `Tracker` for the tracks, calls `predict_blocked_cells` for the threat-ordered groups, applies the area cap, STORES the kept groups in `self._pending_groups` and the un-stamped fold in `self._last_fold`, and RETURNS the full stamp (the union of every kept group's cells) for the base to OR into the fold.
1. **Area cap** (`MAX_STAMP_CELLS = 6000`): `_apply_area_cap` keeps groups greedily while the cumulative cell count stays within the budget, allocating to the soonest-TTC tracks first and stopping at the first group that would overflow. It hard-bounds the per-tick stamp cost and the number of groups the settle-time peel may later drop.
2. **Settle-time fail-open peel** (in the overridden `_settle_and_extract`, which fires only when the follower finishes or its committed segment is blocked — D* Lite's existing commitment horizon): it first settles + extracts with the full stamp already committed. The base settle returns a path exactly when `g(start)` is finite and `None` exactly when the stamp sealed the grid, so the D* Lite search is itself the reachability oracle — there is no separate from-scratch A* probe. On a seal it peels groups farthest-future first (pops the least-imminent from the end of `_pending_groups`), un-stamps each dropped group's stamp-only cells via `_unstamp_group` (restoring them to their fold value in place and reporting them through `update_cells`, never erasing a real fold obstacle or a cell a kept group still needs), and re-settles incrementally until a path re-exists. If even zero predicted stamp leaves the grid unsolvable (a real static dead-end) it returns `None`, so the base keeps its last valid follower — `act()` never raises (AC6). The override is exception-guarded for the same reason, and the most-imminent protection is always retained.

**Why the peel moved to settle-time.** The first cut ran a from-scratch A* reachability probe on every stamp-bearing tick. Under ~20-obstacle traffic that probe cost ~1.8 s/tick and a full episode never terminated within the per-episode wall. Reusing the deferred-settle machinery instead — the `g(start) < inf` the incremental search already produces (the TC46 property: batched `update_cells` + one settle == a from-scratch A* on the same folded grid) — drops the per-tick cost to ~16 ms/tick, near plain `d_star_lite`, so a non-zero-horizon oracle now drives the arena_v1 traffic world to the goal. The settle (and therefore the peel) fires rarely on a clear run.

**The truth seam.** `planners/_types.py` adds two optional members to the `Controller` protocol: `wants_truth: bool = False` (default) and `observe_truth(self, snapshot: tuple) -> None` (default no-op). The runner, before each `act()`, calls `observe_truth(info.dynamic_obstacles)` ONLY when `controller.wants_truth` is set, using the snapshot from the SAME `reset()` or `step()` call that produced the corresponding `state`/`lidar`. `EpisodeInfo.dynamic_obstacles: tuple[DynamicObstacleState, ...]` is an additive field on `EpisodeInfo` (after `dynamic_obstacles_sha256` in field order), populated from the post-`_advance()` snapshot each tick; `()` when traffic is off or pre-reset. It is NOT included in `_trace_line` to preserve byte-identity.

**`d_star_lite_oracle` (shipped).** `DStarLiteOracleController` (`planners/d_star_lite_predictive.py`): sets `geometry = "capsule"`, `wants_truth = True`, uses `OracleTracker`. Registered under `"d_star_lite_oracle"`. A deliberate cheat (perfect live velocities) to measure the achievable motion-aware ceiling before building the lidar estimator. It is EXPERIMENTAL, excluded from `run_all`'s canonical 11.

**`d_star_lite_predictive` (shipped, experimental — net-harmful at v1).** `DStarLitePredictiveController` (`planners/d_star_lite_predictive.py`): the lidar-only variant — `geometry = "cone"`, `wants_truth = False`, fed by `LidarTracker` (`planners/_predict.py`), a frame-differencing velocity estimator (lidar→world points, static-return subtraction via the inflated grid, deterministic 8-connected grid-bucket clustering, sorted greedy nearest-neighbor association, `v = (centroid_now − centroid_prev)/PREDICT_DT`; no velocity clamp in v1; fully deterministic — TC64). Registered under `"d_star_lite_predictive"`, in `PREDICT_FAMILIES`/`EXPERIMENTAL_KEYS` (excluded from the canonical 11). The predictive base `PredictiveDStarLiteController` builds its tracker LAZILY on the first non-h0 `act()` (the `LidarTracker` needs the post-`reset()` grid + beam geometry); its `name` is blanked so the abstract base cannot shadow a registry key. **Finding (T14, `docs/plans/2026-06-27-predictive-d-star-lite.findings.md`):** on the arena_v1 10-seed quick read the oracle cuts failure 0.60 → 0.20 at h10, but the realizable lidar estimator at the same horizon is net-harmful (0.60 → 1.00) — its noisy velocity estimates over-segment obstacles and over-stamp the grid, detouring the robot into new crashes + timeouts. Motion-aware stamping works; estimating motion well enough from lidar is the unsolved wall. The v1 estimator-refinement levers (velocity clamp, tighter clustering, shorter horizon, smaller area cap) are out of the T11–T14 scope.

**`--predict-horizon` knob.** `runners/run_episode.py` and `runners/run_experiment.py` take `--predict-horizon <int steps>`, where T = steps × 0.1 s. **Required** for `PREDICT_FAMILIES`, **rejected** (exit 2) for all other algorithms. `--replan-k` is rejected (exit 2) for `PREDICT_FAMILIES`. Results land in `results/<world_stem>/d_star_lite_oracle_h<steps>/`; `algorithm_label` folds `_h<steps>` into the label. `run_experiment` forwards `--predict-horizon` to each child and records `predict_horizon` in `_manifest.json`. The horizon-0 path is a true no-op: no tracker side effect, no stamp — `d_star_lite_oracle` at `h0` produces a byte-identical trace to plain `d_star_lite` on the same seed (TC57).

**`EXPERIMENTAL_KEYS` and the canonical-set carve-out.** `planners/_grid.py` defines `PREDICT_FAMILIES = frozenset({"d_star_lite_oracle", "d_star_lite_predictive"})` and `EXPERIMENTAL_KEYS = PREDICT_FAMILIES`. `run_all.py`'s import-time assertion is relaxed to `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS`, so the experimental keys can be registered and used through the runner/sweep tooling without landing on the canonical-11 scatter.

**Horizon sweep tooling.** `runners/run_horizon_sweep.py` shells `run_experiment` once per horizon in `{0, 5, 10, 20}` for each key in `SWEEP_ALGORITHMS` (now both `d_star_lite_oracle` and `d_star_lite_predictive`), writing label dirs `<key>_h<steps>/`. `runners/plot_horizon_sweep.py` (headless, read-only; its own `SWEEP_ALGORITHMS` also lists both keys) reads those dirs and writes `failure_rate_vs_horizon.png`, `median_time_vs_horizon.png`, and `horizon_sweep_summary.csv` to `results/<world_stem>/horizon_sweep_plots/` (x-axis = T seconds = steps × 0.1; the oracle and lidar lines render together). `python -m runners.plot_horizon_sweep --selfcheck` runs synthetic-fixture cases (no irsim) and exits 0 only if all pass.

**Shared sweep helpers.** The horizon and speed sweep families used to be near-clones; the common code now lives in two helper modules both delegate to. `runners/_sweep_run.py` (stdlib-only, headless — never imports `run_all`/irsim) holds the subprocess driver loop: `SweepJob` / `SweepOutcome`, `add_common_sweep_args`, `run_jobs`, and `summarize` (the `[sweep i/N] … launching` / `exit=` prints, the failure roster, and the non-zero exit on any runner failure). Each driver keeps only its own `build_experiment_cmd` (the speed driver appends `--speed-regime` / `--replan-k`; the horizon driver appends `--predict-horizon`) and its axis-specific flag (`--algorithms` vs `--horizons`). `runners/_sweep_plot.py` (headless; reaches matplotlib only via `runners.plot.ensure_matplotlib`) holds `render_line_chart` (axis injected as tick positions/labels + x/color getters), `is_nan`, `make_selfcheck_record`, the matplotlib-guard TC body, and the `run_selfcheck_suite` harness; each plotter keeps its own `AlgoSeries`, loader, `build_series`, CSV columns, color map, and `_tc_s*` fixtures.

**Render overlay.** In `--render` mode only, `Arena.draw_prediction(cells, tracks)` paints the predicted footprint as a translucent cell overlay plus per-track velocity arrows on irsim's matplotlib axes, clearing the previous tick's artists so they do not accumulate. The runner reads `controller.last_predicted_cells` / `.last_tracks` after `act()` and calls it. Both debug attrs are initialized to `[]` (never `None`) in `__init__` so the overlay tolerates the pre-first-act case. The draw path is never entered when `render=False`, so headless traces are byte-identical with or without the overlay code.

**TC53–TC64** (added to `python arena/arena.py arena/arena_v1.yaml --check`, growing it from 55 to 68 cases):
- TC53: `predict_blocked_cells` capsule geometry on a synthetic track — a constant-radius disk train along v (the per-step disk cell count does not grow with the lookahead step, and distinct steps give distinct centers so the union is a real train); cells sorted row-major/deduped; byte-identical across two calls (AC4). Pure (synthesizes its own grid/track).
- TC54: `predict_blocked_cells` cone widening + exclusion zone + gate drop — the cone stamp strictly supersets the capsule stamp for the same track/horizon (radius grows with step); no stamped cell's center lies within `exclusion_radius` of the robot for either geometry (AC5); a track that never crosses the corridor is dropped by the gate. Pure.
- TC55: predicted-conflict gate is geometric over [0, T], not an instantaneous closing-course test — a shallow-angle fast crosser that is RECEDING from the robot at t=0 (a closing-rate gate would drop it) yet whose capsule sweeps into the planned-path corridor within the horizon is stamped with a finite TTC; a genuinely receding track is not. Pure.
- TC56: settle-time bounded peel — drives the oracle's `_settle_and_extract` directly after an in-process `reset()` (no irsim/subprocess, like TC46). The full stamp (a tiny most-imminent group + a farthest-future grid-spanning wall) is committed into `self._cells`, the base settle confirms the seal (`g(start)` infinite), and the override peels the wall farthest-future-first until a path re-exists while keeping the imminent group. Includes the shared-cell-survives case: a wall cell that also belongs to the retained imminent group must NOT be un-stamped. `self._cells` is mutated in place, never rebound.
- TC56b: genuine dead-end no-raise (AC6) — (a) a wall that is part of the un-stamped FOLD (not a stamp) keeps `_settle_and_extract` returning `None` even after peeling every predicted group (`_unstamp_group` refuses to erase a real fold obstacle); (b) full-pipeline `act()` under a dense stationary prediction returns a finite `(2,1)` action with a valid follower (the peelable seal re-solves). A fresh controller per check so (a)'s sealed grid never leaks into (b).
- TC57: `d_star_lite_oracle_h0` ≡ plain `d_star_lite` trace — byte-identical `trace.jsonl` on the same seed across BOTH `--no-traffic` and traffic-on (4 runner subprocesses), proving zero-horizon stamping is a true no-op baseline (AC2).
- TC58: `d_star_lite_oracle` `--predict-horizon 10` traffic-on e2e + determinism — runs to a terminal state, 8-key trace per line (incl. `dynamic_obstacles_sha256`), two same-seed runs byte-identical (AC3). Doubles as the de-facto guard that the per-tick cost stays near baseline (the settle-time peel, not a per-tick probe), so a full non-zero-horizon episode finishes inside the 600 s wall.
- TC59: `--predict-horizon` validation (subprocess exit codes) — required for the predict family (omitting it → exit 2), rejected for a non-predict family (exit 2), `--replan-k` rejected for the predict family (exit 2); each rejected run writes no `<seed>.json`; a valid `h0` run lands in the `d_star_lite_oracle_h0` label dir, and `algorithm_label` folds a non-zero horizon into `_h<steps>`.
- TC60: `EpisodeInfo.dynamic_obstacles` truth seam + tick alignment — `()` when traffic off/pre-reset, a length-20 tuple of `DynamicObstacleState` when traffic on; driving an Arena like the runner, the snapshot the oracle observes before each `act(state, lidar)` equals the `info.dynamic_obstacles` from the SAME `reset()`/`step()` that produced that `state`/`lidar` (no off-by-one); plain `d_star_lite` has a falsey `wants_truth` (never observed).
- TC61: `run_all` canonical assertion tolerates experimental keys — importing `planners` + `runners.run_all` does not raise and `set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS` (AC8).
- TC62: `plot_horizon_sweep --selfcheck` — runs the headless selfcheck suite (TC-S1..TC-S7) as a subprocess and asserts exit 0; synthetic-fixture cases only, no irsim (AC9).
- TC63: `d_star_lite_predictive` traffic-on e2e + determinism — the lidar variant clone of TC58: a `--predict-horizon 10` traffic run to a terminal state, 8-key trace per line, two same-seed runs byte-identical (the lidar tracker is deterministic through the runner).
- TC64: `LidarTracker` determinism across a multi-frame cluster-count change — in-process (no irsim), drives the tracker over a 5-frame synthetic lidar fixture whose cluster count walks `[1,1,1,2,1]` (an obstacle enters then leaves), twice on fresh instances, and asserts byte-identical `Track` sequences (`dataclasses.astuple`) plus a non-zero correctly-signed velocity after the move — exercising the association-stability hazard, not just a 2-frame diff.

## Conventions worth preserving

- `manual_astar.py` is written in a strict, dataclass-heavy style (frozen dataclasses, exhaustive `raise ValueError`s on bad input, type hints everywhere, no magic numbers in function bodies). New planner code in this file should match that style; the other scripts are deliberately looser.
- World YAML filenames spell "obstacle" correctly. The earlier "obstical" spelling was renamed — don't reintroduce it.
- Scratch worlds belong outside the repo or under the `_tmp_*` prefix (gitignored). World fixtures intended to live in the repo go in `tests/`.
