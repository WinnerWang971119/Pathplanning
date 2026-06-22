# Obstacle-Speed-Cap Sweep Plan (issue #11)

**Goal:** Make the dynamic-obstacle speed band a swept parameter so we can measure how the obstacle-speed cap drives failure rate and time-to-goal per algorithm — and specifically build the tooling that tests whether D* Lite's failures floor to zero once no obstacle can outrun the robot.

**Approach:** Parameterize the traffic spawner's speed band as two factors of robot top speed (defaulting to the current `0.3`/`1.5` constants, so the baseline stays byte-identical and TC17–TC24 are untouched). Define the four named regimes + the resolver + the shared CLI plumbing in a **new pure module `arena/speed_regimes.py`** (irsim-free, so the headless plotter can import it). Thread the band through `Arena` and the runners as a named regime (`--speed-regime slow|matched|current|fast`, the primary knob both consulted models recommended) with raw `--speed-min-factor`/`--speed-max-factor` overrides for off-menu single runs. A new `run_speed_sweep.py` shells `run_experiment` once per (regime, planner) into a per-regime results subtree (`results/speed_<regime>/...`, the established `__wallclock__`-style sibling convention), and a new `plot_speed_sweep.py` reuses `plot.py`'s loader per regime to chart failure-rate and median-time vs the speed cap, one line per algorithm. Deliver the tooling plus a small smoke run; the full 50-seed sweep and its confirm/refute verdict are the user's to launch.

## Scope

- **In scope:**
  - A new pure module `arena/speed_regimes.py` holding `SPEED_REGIMES`, `SPEED_REGIME_CAP`, `resolve_speed_factors()`, and the shared CLI helpers `add_speed_args()` / `resolve_speed_args()`. It imports only the standard library (no `arena.arena`, no `irsim`, no `matplotlib`) so the headless plotter and its `--selfcheck` can import it.
  - A speed-band parameter on `TrafficSpawner` (two float factors of robot top speed), defaulting to the current module constants, validated `0 < min <= max`.
  - Threading the band through `Arena.__init__` (two optional float params; `None` ⇒ spawner defaults) and through `runners/run_episode.py` + `runners/run_experiment.py` via the shared CLI helpers (`--speed-regime`, default unset⇒`current`; raw `--speed-min-factor`/`--speed-max-factor` override for off-menu single runs).
  - Speed-band provenance recorded in `run_experiment`'s `_manifest.json` (`speed_regime`, `speed_min_factor`, `speed_max_factor`).
  - New `runners/run_speed_sweep.py` driver: runs a selectable planner set (`--algorithms focus|all`) across the 4 named regimes over the canonical seed stream, partitioning output into `results/speed_<regime>/<world_stem>/<label>/`.
  - New `runners/plot_speed_sweep.py`: per-algorithm **failure-rate vs speed cap** and **median-time-to-goal vs speed cap** line charts (x = max-cap factor 0.7/1.0/1.5/2.0), reusing `plot.py`'s `load_world_results`; with a headless `--selfcheck` suite over synthetic fixtures.
  - New arena `--check` TCs (TC48–TC52) proving baseline determinism is preserved, bad bounds are rejected, the band is actually wired (faster cap ⇒ higher speeds, identical *initial* spawn positions across regimes), determinism holds at a non-baseline cap, and the CLI rejects bad/conflicting speed flags with exit 2.
  - A small smoke run (a few seeds, all 4 regimes, `focus` set) end-to-end, and a short findings note describing how to read the plot and run the full sweep.
  - Docs: CLAUDE.md + README + Mission.md Phase 7 note for the new knob, driver, and plotter.

- **Out of scope:**
  - Running the full 50-seed × 4-regime sweep to completion (multi-hour; the user launches it). We ship tooling + a smoke run only. **(User-authorized during planning: the execution-depth question was answered "tooling + smoke run; the user launches the full 50-seed sweep.")**
  - Drawing the final confirmed/refuted scientific conclusion from full data. The smoke run validates plumbing, not the hypothesis. **The deliverable here is the tooling that WOULD confirm or refute the D\* Lite hypothesis once the user runs the full sweep; producing the verdict is explicitly the user's step, not an acceptance criterion of this work.**
  - **Off-menu speed bands inside the *sweep*.** The raw `--speed-min-factor`/`--speed-max-factor` override is for single `run_episode`/`run_experiment` runs only; `run_speed_sweep` covers exactly the four named regimes. Extending the driver to accept ad-hoc bands is a possible follow-up, not in scope.
  - The Phase 7 traffic-**density** stretch goal (count, not speed) — explicitly a separate knob.
  - Any change to the trace/metrics JSON schema (the 7/8-key contract stays fixed; the regime lives in the manifest only).
  - Per-obstacle motion noise (`motion_rng` stays plumbed-but-unused, as today).
  - Folding the speed knob or a planner-subset into `run_all.py` (keeps its all-11 `_CANONICAL_ORDER` invariant clean).

## Decisions

- **Named-regime enum as the primary knob, raw floats as override** — Both consulted models (Opus and Codex) independently recommended a `--speed-regime` knob over threading two raw floats through every runner: it gives validated input, a self-documenting manifest field, and one token to forward. The user's Q3 answer required the bounds stay configurable for other bands, so the raw `--speed-min-factor`/`--speed-max-factor` override is added for single runs. The two raw floats live at the bottom (spawner/Arena) so off-menu caps stay reachable and determinism is defined at the float level.
- **Regime table + CLI plumbing live in a pure `arena/speed_regimes.py`** — `arena/dynamic.py` imports `arena.arena`, which imports `irsim`, so placing the constants there would pull irsim into the headless plotter and break its `--selfcheck` (no-irsim) guarantee. The pure module is the single source of truth imported by `arena/dynamic.py`, all three runners, the sweep driver, and the headless plotter (judge ruling, Codex critic).
- **CLI validation is a manual post-parse check, not an argparse mutex group** — a native `mutually_exclusive_group` cannot express "regime OR (min AND max together)"; it would make `--speed-min-factor` and `--speed-max-factor` exclusive with each other. `--speed-regime` defaults to `None` (so an explicit `--speed-regime current` is distinguishable from the unset default), and `resolve_speed_args(ns)` enforces the rules and calls `parser.error` (exit 2) on a violation (judge ruling, Codex critic).
- **Spawner change kept verbatim** — both models endorsed it. Defaulting the two factors to the existing constants and reusing the same `uniform(lo, hi) * robot_top_speed` call keeps the draw order/count identical, so the baseline is byte-identical and TC17–TC24 pass unchanged. The existing all-keyword `TrafficSpawner(...)` call in `arena.py` is unaffected by inserting the `*` keyword-only marker.
- **Per-regime results subtree `results/speed_<regime>/<world_stem>/<label>/`** — driver passes `--results-dir results/speed_<regime>` to children; `episode_out_dir`'s unconditional `<world_stem>/<label>` suffix lands files where `load_world_results("results/speed_<regime>", stem)` already reads them, with zero new path code (mirrors `__wallclock__`). `results/` is already gitignored, so the subtrees are too.
- **Dedicated `run_speed_sweep.py` driver, not a `run_all` extension** — Opus flagged that adding a planner-subset to `run_all` muddies its `_CANONICAL_ORDER == ALGORITHMS` drift guard. A dedicated driver shelling `run_experiment` per (regime, planner) reuses the proven subprocess/determinism/manifest machinery without touching that invariant.
- **Focus set = `a_star_once`, `d_star_lite`, `dwa`, `apf`** — the static baseline + the incremental planner the hypothesis is about + the two reactive planners whose degradation the issue calls out. `all` reuses the 11 canonical labels (the driver imports `run_all.canonical_planner_set()`; the headless plotter uses `plot.CANONICAL` instead, to stay irsim-free).
- **x-axis = max-cap factor (0.7/1.0/1.5/2.0)** — per the user's Q3 choice. `SPEED_REGIME_CAP` maps regime→max-factor for the x positions; regime names annotate the ticks.
- **In-process matrix runner rejected** — both models flagged that running many episodes in one process risks irsim global-state bleed (`id_iter` resets only on `make()`; reset warm-up clears flags) and breaks the byte-identical determinism the issue names as load-bearing. The subprocess-per-episode model is preserved.

## Acceptance Criteria

- [ ] AC1 — `TrafficSpawner(...)` accepts `speed_min_factor` / `speed_max_factor`; with the defaults (current constants) and a fixed seed, the `dynamic_obstacles_sha256` sequence is byte-identical to today's. (TC50)
- [ ] AC2 — `TrafficSpawner` raises `ValueError` for `speed_min_factor <= 0` and for `speed_max_factor < speed_min_factor`. (TC49)
- [ ] AC3 — `SPEED_REGIMES` defines exactly `slow=(0.3,0.7)`, `matched=(0.3,1.0)`, `current=(0.3,1.5)`, `fast=(0.5,2.0)`; `resolve_speed_factors("current", None, None) == (0.3, 1.5)` and equals the spawner's default constants. (TC48)
- [ ] AC4 — For a fixed seed, the **initial `initialize()` spawn snapshot** is identical across two regimes in obstacle spawn positions and velocity directions, with only obstacle speeds scaled by the band (Fast strictly faster than Slow). This invariant holds for the initial snapshot **only** — once an obstacle despawns and refills, the regimes draw a differing count of fresh `traffic_rng` values and the streams diverge, so later positions are not expected to match. (TC51)
- [ ] AC5 — `Arena(traffic=True)` (no speed args) and `Arena(traffic=True, speed_min_factor=0.3, speed_max_factor=1.5)` produce byte-identical `dynamic_obstacles_sha256` sequences; the full `--check` suite still passes with TC17–TC24 unchanged. (TC50 + full suite)
- [ ] AC6 — `python -m runners.run_episode --speed-regime current ...` produces a byte-identical trace JSONL to the same run with no speed flag; `--speed-regime fast` differs. (TC50)
- [ ] AC7 — Speed-flag CLI validation (exits 2, no episode output, no traceback): an explicit `--speed-regime` together with a raw override; a lone `--speed-min-factor` or `--speed-max-factor`; an unknown regime (`--speed-regime bogus`); a non-positive `--speed-min-factor` or a `--speed-max-factor < --speed-min-factor`. (TC-CLI)
- [ ] AC8 — `run_experiment` forwards the resolved speed flags to each `run_episode` child (provable by the pure command-builder), and records `speed_regime`, `speed_min_factor`, `speed_max_factor` in `_manifest.json`. (TC-FWD)
- [ ] AC9 — `python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus --num-seeds 3` writes `results/speed_{slow,matched,current,fast}/arena_v1/<label>/<seed>.json` for each focus planner. The driver completes; recorded in-sim outcomes (crash/timeout, which are child exit 0) are acceptable, and a wallclock-killed child is *reported* as a runner failure (driver exits non-zero) rather than treated as a plumbing defect — the AC is that the files are produced and the run is accounted for, not that every episode reached the goal.
- [ ] AC10 — `python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus` reads the four regime subtrees and writes a failure-rate-vs-cap PNG and a median-time-vs-cap PNG with one line per **present** algorithm (no empty lines for absent planners), plus a `speed_sweep_summary.csv`.
- [ ] AC11 — `python -m runners.plot_speed_sweep --selfcheck` runs its synthetic-fixture suite headlessly (no irsim, no real episodes) and exits 0 only if all cases pass.
- [ ] AC12 — Two same-seed sweep runs at a non-baseline regime produce byte-identical per-seed trace JSONL (determinism holds at any cap). (TC52)
- [ ] AC13 — Running the full `--check` suite prints all cases PASS, now including TC48–TC52 + the CLI/forward unit cases; the `--check` runtime help/count and every "48"/"48 PASS"/"48-case" mention in `arena/arena.py`'s docstring, CLAUDE.md, and README are bumped to the new count.
- [ ] AC14 — Hypothesis-readiness: the "failure rate vs speed cap" chart is structured so that, fed full-sweep data, it directly answers the D* Lite hypothesis (correct axes, per-algorithm series including D* Lite, all four regimes as distinct x positions). The smoke run renders this chart shape end-to-end on smoke data; it does **not** assert the hypothesis is confirmed or refuted (that needs the user's full sweep, per Out of scope).

## Data Model

```python
# arena/speed_regimes.py — PURE module (stdlib only; no arena.arena / irsim / matplotlib import).

# Each value is (min_factor, max_factor) of robot top speed. "current" reproduces
# the existing arena/dynamic.py SPEED_MIN_FACTOR / SPEED_MAX_FACTOR constants exactly.
SPEED_REGIMES: dict[str, tuple[float, float]] = {
    "slow":    (0.3, 0.7),
    "matched": (0.3, 1.0),
    "current": (0.3, 1.5),   # the Mission baseline
    "fast":    (0.5, 2.0),
}

# x positions for plot_speed_sweep: the "cap" is the max factor.
SPEED_REGIME_CAP: dict[str, float] = {k: v[1] for k, v in SPEED_REGIMES.items()}
```

## Contracts & Interfaces

Single source of truth for every cross-task seam. A task conforms to the exact name/signature/shape here.

### Shared module / tables (owner: T1)

- `arena/speed_regimes.py` — pure (stdlib only). Holds `SPEED_REGIMES`, `SPEED_REGIME_CAP`, `resolve_speed_factors`, `add_speed_args`, `resolve_speed_args`, and a `DEFAULT_REGIME = "current"`. Consumers: T2 (`arena/dynamic.py` re-imports the band constants), T3, T4, T5, T6. Invariant: imports nothing that pulls `irsim`/`matplotlib`, so AC11's headless selfcheck holds.
- `arena/dynamic.py::SPEED_MIN_FACTOR = 0.3`, `SPEED_MAX_FACTOR = 1.5` remain the spawner defaults (unchanged values); `arena/speed_regimes.py` mirrors them in `SPEED_REGIMES["current"]` and a TC asserts the two agree.

### Signatures

- `TrafficSpawner.__init__(self, env, robot, traffic_rng, motion_rng, dt, arena_w, arena_h, static_obstacles, *, speed_min_factor: float = SPEED_MIN_FACTOR, speed_max_factor: float = SPEED_MAX_FACTOR)` — owner: T1; consumer: T2. Validates `0 < speed_min_factor <= speed_max_factor`, else `ValueError`. The existing all-keyword `TrafficSpawner(...)` call in `arena.py` is unaffected by inserting `*`. The speed draw becomes `uniform(self._speed_min_factor, self._speed_max_factor) * self._robot_top_speed` — same call site, same draw order/count.
- `Arena.__init__(self, yaml_path, seed, render=False, timeout_s=DEFAULT_TIMEOUT_S, traffic=False, *, speed_min_factor: float | None = None, speed_max_factor: float | None = None)` — owner: T2; consumers: T3, T7. When both are `None`, omit them from the `TrafficSpawner(...)` kwargs entirely (do not pass `None`), so direct `Arena(traffic=True)` stays byte-identical. Both-set ⇒ pass through. One-sided (`ValueError`).
- `arena/speed_regimes.py::resolve_speed_factors(regime: str | None, min_override: float | None, max_override: float | None) -> tuple[float, float]` — owner: T1; consumers: T3, T4, T5. If both overrides given, validate (`0 < min <= max`) and return them; else look up `regime` (None ⇒ `DEFAULT_REGIME`); unknown regime ⇒ `ValueError` listing valid keys. (The regime-vs-override conflict and both-or-neither rules are enforced earlier, by `resolve_speed_args`.)
- `arena/speed_regimes.py::add_speed_args(parser: argparse.ArgumentParser) -> None` — owner: T1; consumers: T3, T4. Registers `--speed-regime` (`choices` = the 4 keys, `default=None`), `--speed-min-factor` (float, default None), `--speed-max-factor` (float, default None) as **plain** args (NOT a mutually-exclusive group).
- `arena/speed_regimes.py::resolve_speed_args(parser, ns) -> tuple[float, float]` — owner: T1; consumers: T3, T4. Manual post-parse validation: (a) reject an explicit regime together with either override (`parser.error`, exit 2); (b) require both overrides or neither (one alone ⇒ exit 2); (c) `0 < min <= max` (else exit 2); (d) `None` regime ⇒ `DEFAULT_REGIME`. Returns the resolved `(min, max)`.

### Runner CLI contract (T3 `run_episode`, mirrored by T4 `run_experiment` via the SAME shared helpers)

- Both runners call `add_speed_args(parser)` to register the flags and `resolve_speed_args(parser, ns)` to validate+resolve — no duplicated argparse logic between them.
- `run_episode` passes the resolved `(min, max)` to `Arena(..., speed_min_factor=min, speed_max_factor=max)`.
- `run_experiment` forwards the user's *original* speed flags verbatim to each `run_episode` child (so the child re-validates), AND resolves once for the manifest provenance fields.
- **No change** to the trace/metrics JSON schema.

### Results layout (owner: T5; consumer: T6)

- `results/speed_<regime>/<world_stem>/<label>/<seed>.json` and `…/_manifest.json` — produced by `run_experiment` children launched with `--results-dir results/speed_<regime>`. Read by `plot_speed_sweep` via `load_world_results("results/speed_<regime>", world_stem, replan_k=K)`.

### Manifest provenance (owner: T4)

- `run_experiment`'s `_manifest.json` gains `"speed_regime": str | null`, `"speed_min_factor": float`, `"speed_max_factor": float`. These are write-only provenance; the plotter derives its x-axis from `SPEED_REGIME_CAP[regime]`, not from the manifest floats.

### Planner sets

- `run_speed_sweep.FOCUS_SET = ("a_star_once", "d_star_lite", "dwa", "apf")` — non-replan, `replan_k=None`. Owner: T5.
- Driver `all`: T5 imports `run_all.canonical_planner_set()` (subprocess driver, irsim import is fine).
- Plotter `all`: T6 uses `plot.CANONICAL` (headless — must NOT import `run_all`, which pulls irsim).

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `arena/speed_regimes.py` (new, pure) | T1 | T2, T3, T4, T5, T6 (import only) |
| `arena/dynamic.py` | T1 | T2 |
| `arena/arena.py` (Arena class) | T2 | T3, T7 |
| `arena/arena.py` (TCs + `_run_checks` + module docstring count) | T7 | — |
| `runners/run_episode.py` | T3 | T4 (subprocess contract) |
| `runners/run_experiment.py` | T4 | T5 (subprocess contract) |
| `runners/run_speed_sweep.py` (new) | T5 | T8 (smoke) |
| `runners/plot_speed_sweep.py` (new) | T6 | T8 (smoke) |
| `CLAUDE.md`, `README.md`, `Mission.md` | T8 | — |

### Naming

- Regime keys: exactly `slow`, `matched`, `current`, `fast` (lowercase) everywhere — CLI choices, subtree names (`speed_<regime>`), manifest field, plot ticks.
- Subtree prefix: `speed_` (so `results/speed_fast/...`).

## Error Handling

- **Bad spawner bounds** (`min <= 0` or `max < min`): `TrafficSpawner.__init__` raises `ValueError` with the offending values — fail at construction, never silently draw negative/zero speeds.
- **One-sided Arena override**: `Arena.__init__` raises `ValueError`.
- **Unknown regime via API**: `resolve_speed_factors` raises `ValueError` listing valid keys.
- **Conflicting / malformed CLI speed flags** (explicit regime + override; a lone min or max; unknown regime; non-positive min; max < min): `resolve_speed_args` calls `parser.error`, argparse exits 2 — an explicit post-parse check, NOT a mutually-exclusive group, and BEFORE any `Arena` is constructed so a bound error never surfaces as a mid-run traceback.
- **Sweep child failure**: `run_speed_sweep` mirrors `run_all` — a non-zero `run_experiment` exit is recorded, the sweep continues to the next (regime, planner), and the driver exits non-zero at the end listing the failures. An in-sim crash/timeout is exit 0 (recorded in the metrics JSON), not a runner failure.
- **Plotter on a missing/empty regime subtree**: `load_world_results` warns-not-raises on missing label dirs; `plot_speed_sweep` treats a regime with zero present episodes for an algorithm as a gap in that line (annotated), never a crash. An entirely absent regime subtree is reported and the chart renders from the present regimes.
- **Plotter with no data at all**: exit 1 with a clear message (mirrors `plot.py`).

## Testing Strategy

**Levels:** Unit (regime table, validation, resolver, CLI helpers, command-builder), Integration (Arena/runner determinism, sweep e2e smoke), plus the existing `--check` regression gate.

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC48 | Regime table + resolver | Unit | `SPEED_REGIMES` has the exact 4 bands; `resolve_speed_factors("current",None,None)==(0.3,1.5)` and `==` the spawner constants; both overrides return them; unknown regime raises `ValueError` |
| TC49 | Spawner bound validation | Unit | `speed_min_factor<=0` and `speed_max_factor<speed_min_factor` each raise `ValueError`; `min==max` allowed |
| TC50 | Baseline determinism preservation + draw-count guard | Integration | `Arena(traffic=True)` vs explicit `(0.3,1.5)` give byte-identical `dynamic_obstacles_sha256` over N ticks; runner `--speed-regime current` trace == no-flag trace (byte-identical); `--speed-regime fast` differs; a guard asserts the per-spawn `traffic_rng` draw count is unchanged (3 per attempt) so a reordered/added draw fails here, not just on the hash |
| TC51 | Band wired + controlled-experiment property (initial snapshot only) | Unit | At one seed, two regimes' **initial `initialize()` snapshots** give identical spawn `(x,y)` and identical velocity DIRECTION (`atan2(vy,vx)`; `DynamicObstacleState` stores `vx,vy`, not a heading) per obstacle id; obstacle speeds scale with the band (Fast max speed > Slow max speed). Assertions are on the t=0 snapshot only — no stepping |
| TC52 | Non-baseline determinism (with a despawn/refill cycle) | Integration | Two same-seed `Arena(traffic=True, speed_min_factor=0.5, speed_max_factor=2.0)` runs give identical sha256 sequences over enough ticks to force ≥1 despawn+refill at the fast band |
| TC-CLI | Speed-flag CLI rejection | Unit | Subprocess `run_episode` with: regime+override; lone min; lone max; `--speed-regime bogus`; `--speed-min-factor 0`; `--speed-max-factor < min` — each returns exit code 2 and writes no `<seed>.json` |
| TC-FWD | `run_experiment` flag forwarding | Unit | The pure child-command builder emits `--speed-regime <regime>` (or the float overrides) in the child argv; the manifest dict carries `speed_regime`/`speed_min_factor`/`speed_max_factor` |
| TC-S1..n | `plot_speed_sweep --selfcheck` over synthetic per-regime JSON trees | Unit | Loader-per-regime aggregates failure_rate/median_time correctly; x = `SPEED_REGIME_CAP`; color map built from full `plot.CANONICAL` order THEN filtered (`n_present>0`) so colors stay stable across regimes; both charts render under Agg; matplotlib-absent guard exits 1; missing-regime gap handled |
| (regression) | Full `python arena/arena.py arena/arena_v1.yaml --check` | Integration | All cases PASS, TC17–TC24 byte-identical (proves the baseline is untouched) |

**Test data:** TC48/49/51 are pure/near-pure (a tiny static-obstacle list or the arena world; no full episode). TC50/52 drive `Arena` over a fixed tick count with zero actions, comparing sha256 sequences (mirrors TC20). TC-CLI/TC-FWD are subprocess/pure-function checks (no irsim sim). The plotter selfcheck builds synthetic `results/speed_<regime>/<stem>/<label>/<seed>.json` trees in a `TemporaryDirectory` (mirrors `plot.py`'s TC-P pattern) — no irsim.

**Run command:**
- TCs: `.venv\Scripts\Activate.ps1; python arena/arena.py arena/arena_v1.yaml --check`
- Plot selfcheck: `python -m runners.plot_speed_sweep --selfcheck`
- Smoke (AC9/AC10): `python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus --num-seeds 3` then `python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus`

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Pure regime module + spawner speed-band params | — | high | `arena/speed_regimes.py` (new), `arena/dynamic.py` | Create `arena/speed_regimes.py` (stdlib only) with `SPEED_REGIMES`, `SPEED_REGIME_CAP`, `DEFAULT_REGIME`, `resolve_speed_factors`, `add_speed_args`, `resolve_speed_args` per Contracts. In `arena/dynamic.py`: add keyword-only `speed_min_factor`/`speed_max_factor` to `TrafficSpawner.__init__` (defaults = existing `SPEED_MIN_FACTOR`/`SPEED_MAX_FACTOR`), validate `0 < min <= max` (else `ValueError`), store, and use them in the existing `_try_one_spawn` speed draw (same call site/order/count). Do NOT change the draw order or count. Satisfies AC1–AC4. |
| T2 | Thread band through `Arena` | T1 | med | `arena/arena.py` (Arena class only) | Add keyword-only `speed_min_factor`/`speed_max_factor` (default `None`) to `Arena.__init__`; when both `None`, omit from the `TrafficSpawner(...)` call so the default path is byte-identical; both-set ⇒ pass through; one-sided ⇒ `ValueError`. The existing all-keyword spawner call is otherwise untouched. No change to `reset()` RNG derivation. Satisfies AC5. |
| T3 | `run_episode` CLI knob | T1, T2 | med | `runners/run_episode.py` | Call `add_speed_args(parser)` + `resolve_speed_args(parser, ns)`; pass the resolved `(min,max)` to `Arena(...)`. Extend `RunnerArgs` + module docstring. No trace/metrics schema change. Satisfies AC6, AC7. |
| T4 | `run_experiment` forward + manifest | T3 | med | `runners/run_experiment.py` | Mirror the T3 flags via the SAME `add_speed_args`/`resolve_speed_args` helpers; forward the user's original speed flags verbatim to each `run_episode` child (in the pure command-builder); resolve once to record `speed_regime`/`speed_min_factor`/`speed_max_factor` in `_manifest.json`. Update `RunnerArgs` + docstring. Satisfies AC8. |
| T5 | `run_speed_sweep` driver | T4 | med | `runners/run_speed_sweep.py` (new) | New module mirroring `run_all`'s subprocess pattern: `--world`, `--algorithms {focus,all}` (default `focus`), `--master-seed`/`--num-seeds`/`--jobs`/`--resume`/`--traffic` passthrough. For each regime in `SPEED_REGIMES`, for each planner in the selected set, shell `python -m runners.run_experiment --speed-regime <regime> --results-dir results/speed_<regime> ...`. Record per-(regime,planner) exit; continue past failures; exit non-zero if any failed. `FOCUS_SET` per Contracts; `all` imports `run_all.canonical_planner_set()`. Satisfies AC9. |
| T6 | `plot_speed_sweep` plotter + selfcheck | T1 | med | `runners/plot_speed_sweep.py` (new) | New read-only plotter: `--world`, `--algorithms {focus,all}`, `--results-dir` (root), `--replan-k`, `--out-dir`, `--selfcheck`. Import `SPEED_REGIMES`/`SPEED_REGIME_CAP` from the pure `arena.speed_regimes` (NOT `arena.dynamic`); reuse `runners.plot.load_world_results`, `ensure_matplotlib`, `CANONICAL`, `_algorithm_color_map`. Build the color map from the FULL `plot.CANONICAL` order, THEN filter to `n_present>0` for drawing (so colors are stable across regimes). Per algorithm, build `failure_rate` and `median_time` series across regimes; render two line charts (x = `SPEED_REGIME_CAP`) + `speed_sweep_summary.csv`. `--selfcheck` builds synthetic per-regime trees in a `TemporaryDirectory` (TC-S*, headless). Satisfies AC10, AC11, AC14. |
| T7 | New `--check` TCs | T1, T2, T3 | med | `arena/arena.py` (TC section + `_run_checks` + module docstring) | Add `tc48`–`tc52`, `tc_cli`, `tc_fwd` per the Testing Strategy, register them in `_run_checks`, and update the `--check` runtime help text + the module-docstring count. TC50/TC52 mirror TC20's tick-and-compare-sha256 shape; TC-CLI runs `run_episode` subprocesses asserting exit 2. Satisfies AC2–AC8, AC12, AC13 (runtime portion). |
| T8 | Smoke run + findings note + docs | T1, T2, T3, T4, T5, T6, T7 | low | `CLAUDE.md`, `README.md`, `Mission.md`, `docs/plans/2026-06-23-obstacle-speed-sweep.findings.md` (new) | Run the full `--check` suite (confirm all PASS), the plot selfcheck, and the AC9/AC10 smoke; capture results. `grep` for `48`/`48 PASS`/`48-case` across CLAUDE.md + README + the arena docstring and bump every stale count. Write a short findings note: how to read the two charts, how to launch the full 50-seed sweep, what the smoke proves (plumbing, not the hypothesis), and the caveat that `a_star_once`'s crash rate is not cleanly speed-monotone (it never dodges, so it is a static baseline, not a hypothesis subject). Document the new knob/driver/plotter in CLAUDE.md + README, and add a Mission.md Phase 7 note that the speed-cap sweep tooling exists. |

Parallelism: T1 is the root. T2 and T6 can both start once T1 lands (T6 only needs the pure module + the `plot.py` loader + the layout contract). T3 waits on T1+T2; T4 on T3; T5 on T4; T7 on T1+T2+T3. T8 is the final integration/smoke/docs gate.

## Notes for Implementer

- **The single load-bearing invariant:** the speed draw must stay the *third* `traffic_rng` draw, made with the *same* `Generator.uniform(lo, hi)` call (only `lo`/`hi` become parameters). `uniform` consumes the same RNG bits regardless of bounds, so the baseline `(0.3, 1.5)` reproduces today's stream byte-for-byte. Do not reorder, add, or remove any draw, and do not branch the draw on the band. TC50 is the binding gate (it also guards the per-attempt draw count); if it fails, the change is wrong.
- **Initial-snapshot scope is real (don't over-claim):** overlap rejection is position-only and speed-independent, so for a fixed seed the *initial* `initialize()` population gets identical spawn positions/headings across regimes — only velocity magnitude differs. But this equality is guaranteed *only at t=0*: faster obstacles reach the despawn buffer sooner, so refills draw a differing number of `traffic_rng` values per regime and the streams diverge thereafter. Assert the controlled-experiment property on the initial snapshot only (AC4/TC51), never across a stepped episode.
- **`None`-means-default in Arena is load-bearing:** when both factors are `None`, omit them from the `TrafficSpawner(...)` kwargs entirely (don't pass `None` through). TC17–TC24 construct `Arena(traffic=True)` directly and must not change.
- **Headless boundary:** the plotter (`plot_speed_sweep`) and its `--selfcheck` must import speed constants from the pure `arena.speed_regimes`, and the `all` label list from `plot.CANONICAL` — never from `arena.dynamic` or `run_all` (both pull irsim). The driver (`run_speed_sweep`) is a subprocess launcher, so importing `run_all.canonical_planner_set()` there is fine.
- **Color stability:** `plot._algorithm_color_map` colors by `enumerate` index over the summaries you pass it. Build it once from the FULL `plot.CANONICAL`-ordered summaries, THEN filter to present algorithms for drawing — filtering first would shift indices and recolor lines differently per regime.
- **Don't touch the trace/metrics schema.** The regime is provenance — `_manifest.json` only.
- **Plotter reuse, don't reimplement:** import `load_world_results`, `AlgoSummary`, `ensure_matplotlib`, `CANONICAL`, and `_algorithm_color_map` from `runners.plot`. New logic is only "load four subtrees, line them up by regime cap, draw two line charts."
- **Cost expectation for the smoke:** D* Lite is the slowest planner and can approach the 600 s per-episode wall under the `fast` band; keep the smoke at ~3 seeds × 4 regimes on `focus`. The full 50-seed sweep is the user's to launch. State this in the findings note.
- **`all` must not drift:** the driver imports `run_all.canonical_planner_set()` for the 11-planner case rather than re-listing labels, so a registry change can't desync the sweep from the study.
- **Rollback:** the change is additive and default-preserving — reverting the new modules + the keyword params restores exact prior behavior. The full `--check` suite passing (with TC17–TC24 byte-identical) is the safety net.
