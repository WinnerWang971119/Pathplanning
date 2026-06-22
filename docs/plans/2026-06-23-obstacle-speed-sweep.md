# Obstacle-Speed-Cap Sweep Plan (issue #11)

**Goal:** Make the dynamic-obstacle speed band a swept parameter so we can measure how the obstacle-speed cap drives failure rate and time-to-goal per algorithm — and specifically test whether D* Lite's failures floor to zero once no obstacle can outrun the robot.

**Approach:** Parameterize the traffic spawner's speed band as two factors of robot top speed (defaulting to the current `0.3`/`1.5` constants, so the baseline stays byte-identical and TC17–TC24 are untouched). Thread the band through `Arena` and the runners as a **named regime enum** (`--speed-regime slow|matched|current|fast`, the primary knob both consulted models recommended) backed by a single lookup table in `arena/dynamic.py`, with a raw `--speed-min-factor`/`--speed-max-factor` override for off-menu bands. A new `run_speed_sweep.py` shells `run_experiment` once per (regime, planner) into a per-regime results subtree (`results/speed_<regime>/...`, the established `__wallclock__`-style sibling convention), and a new `plot_speed_sweep.py` reuses `plot.py`'s loader per regime to chart failure-rate and median-time vs the speed cap, one line per algorithm. Deliver the tooling plus a small smoke run; the full 50-seed sweep is the user's to launch.

## Scope

- **In scope:**
  - A speed-band parameter on `TrafficSpawner` (two float factors of robot top speed), defaulting to the current module constants, validated `0 < min <= max`.
  - A `SPEED_REGIMES` lookup table + `resolve_speed_factors()` resolver in `arena/dynamic.py` (single source of truth for the four named bands).
  - Threading the band through `Arena.__init__` (two optional float params; `None` ⇒ spawner defaults) and through `runners/run_episode.py` + `runners/run_experiment.py` as `--speed-regime` (default `current`) with a mutually-exclusive `--speed-min-factor`/`--speed-max-factor` override.
  - Speed-band provenance recorded in `run_experiment`'s `_manifest.json` (`speed_regime`, `speed_min_factor`, `speed_max_factor`).
  - New `runners/run_speed_sweep.py` driver: runs a selectable planner set (`--algorithms focus|all`) across the 4 regimes over the canonical seed stream, partitioning output into `results/speed_<regime>/<world_stem>/<label>/`.
  - New `runners/plot_speed_sweep.py`: per-algorithm **failure-rate vs speed cap** and **median-time-to-goal vs speed cap** line charts (x = max-cap factor 0.7/1.0/1.5/2.0), reusing `plot.py`'s `load_world_results`; with a `--selfcheck` suite over synthetic fixtures.
  - New arena `--check` TCs (TC48–TC52) proving baseline determinism is preserved, bad bounds are rejected, the band is actually wired (faster cap ⇒ higher speeds, identical spawn positions across regimes), and determinism holds at a non-baseline cap.
  - A small smoke run (a few seeds, all 4 regimes, `focus` set) end-to-end, and a short findings note describing how to read the plot and run the full sweep.
  - Docs: CLAUDE.md + README + Mission.md Phase 7 note for the new knob, driver, and plotter.

- **Out of scope:**
  - Running the full 50-seed × 4-regime sweep to completion (multi-hour; the user launches it). We ship tooling + a smoke run only.
  - Drawing the final confirmed/refuted scientific conclusion from full data (depends on the full run the user makes). The smoke run validates plumbing, not the hypothesis.
  - The Phase 7 traffic-**density** stretch goal (count, not speed) — explicitly a separate knob.
  - Any change to the trace/metrics JSON schema (the 7/8-key contract stays fixed; the regime lives in the manifest only).
  - Per-obstacle motion noise (`motion_rng` stays plumbed-but-unused, as today).
  - Folding the speed knob or a planner-subset into `run_all.py` (keeps its all-11 `_CANONICAL_ORDER` invariant clean).

## Decisions

- **Named-regime enum as the primary knob, raw floats as override** — Both consulted models (Opus and Codex) independently recommended a `--speed-regime` enum over threading two raw floats through every runner: it gives free `choices=` argparse validation (exit 2), a self-documenting manifest field, and one token to forward instead of two. The user's Q3 answer required the bounds stay configurable for other bands, so a mutually-exclusive `--speed-min-factor`/`--speed-max-factor` override is added on top. The two raw floats live at the bottom (spawner/Arena) so off-menu caps stay reachable and determinism is defined at the float level.
- **Spawner change kept verbatim from the lean** — both models endorsed it. Defaulting the two factors to the existing constants and reusing the same `uniform(lo, hi) * robot_top_speed` call keeps the draw order/count identical, so the baseline is byte-identical and TC17–TC24 pass unchanged. This is the load-bearing determinism property.
- **`SPEED_REGIMES` + `resolve_speed_factors` live in `arena/dynamic.py`** — single source of truth imported by all three runners + the driver, so the band table is never duplicated (Contracts rule 1).
- **Per-regime results subtree `results/speed_<regime>/<world_stem>/<label>/`** — driver passes `--results-dir results/speed_<regime>` to children; `episode_out_dir`'s unconditional `<world_stem>/<label>` suffix lands files where `load_world_results("results/speed_<regime>", stem)` already reads them, with zero new path code (both models flagged this; mirrors the `__wallclock__` convention). `results/` is already gitignored, so the subtrees are too.
- **Dedicated `run_speed_sweep.py` driver, not a `run_all` extension** — Opus flagged that adding a planner-subset to `run_all` muddies its `_CANONICAL_ORDER == ALGORITHMS` drift guard. A dedicated driver that shells `run_experiment` per (regime, planner) reuses the proven subprocess/determinism/manifest machinery without touching that invariant.
- **Focus set = `a_star_once`, `d_star_lite`, `dwa`, `apf`** — the static baseline + the incremental planner the hypothesis is about + the two reactive planners whose degradation the issue calls out. `all` reuses the 11 canonical labels (imported from `run_all`, not re-listed, to avoid drift).
- **x-axis = max-cap factor (0.7/1.0/1.5/2.0)** — per the user's Q3 choice ("speed cap"). A regime→max-factor map drives the x positions; regime names annotate the ticks.
- **In-process matrix runner rejected** — both models flagged that running many episodes in one process risks irsim global-state bleed (`id_iter` resets only on `make()`; reset warm-up clears flags) and breaks the byte-identical determinism the issue names as load-bearing. The subprocess-per-episode model is preserved.
- **`current` is the runner default regime** — so "no speed flag" and `--speed-regime current` are literally the same `(0.3, 1.5)` code path, which the determinism TC pins.

## Acceptance Criteria

- [ ] AC1 — `TrafficSpawner(...)` accepts `speed_min_factor` / `speed_max_factor`; with the defaults (current constants) and a fixed seed, the `dynamic_obstacles_sha256` sequence is byte-identical to today's. (TC50)
- [ ] AC2 — `TrafficSpawner` raises `ValueError` for `speed_min_factor <= 0` and for `speed_max_factor < speed_min_factor`. (TC49)
- [ ] AC3 — `SPEED_REGIMES` defines exactly `slow=(0.3,0.7)`, `matched=(0.3,1.0)`, `current=(0.3,1.5)`, `fast=(0.5,2.0)`; `resolve_speed_factors("current", None, None) == (0.3, 1.5)` and equals the spawner's default constants. (TC48)
- [ ] AC4 — For a fixed seed, two regimes produce **identical obstacle spawn positions and headings**, with obstacle speeds scaled by the band (Fast strictly faster than Slow). (TC51)
- [ ] AC5 — `Arena(traffic=True)` (no speed args) and `Arena(traffic=True, speed_min_factor=0.3, speed_max_factor=1.5)` produce byte-identical `dynamic_obstacles_sha256` sequences; the full 48-case `--check` suite still passes (TC17–TC24 unchanged). (TC50 + full suite)
- [ ] AC6 — `python -m runners.run_episode --speed-regime current ...` produces a byte-identical trace JSONL to the same run with no speed flag; `--speed-regime fast` differs. (TC50)
- [ ] AC7 — `--speed-regime` and `--speed-min-factor`/`--speed-max-factor` are mutually exclusive; supplying both exits 2; an unknown regime exits 2.
- [ ] AC8 — `run_experiment`'s `_manifest.json` carries `speed_regime`, `speed_min_factor`, `speed_max_factor`.
- [ ] AC9 — `python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus --num-seeds 3` writes `results/speed_{slow,matched,current,fast}/arena_v1/<label>/<seed>.json` for each focus planner and exits 0 on a clean smoke.
- [ ] AC10 — `python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus` reads the four regime subtrees and writes a failure-rate-vs-cap PNG and a median-time-vs-cap PNG with one line per **present** algorithm (no empty lines for absent planners), plus a `speed_sweep_summary.csv`.
- [ ] AC11 — `python -m runners.plot_speed_sweep --selfcheck` runs its synthetic-fixture suite headlessly (no irsim) and exits 0 only if all cases pass.
- [ ] AC12 — Two same-seed sweep runs at a non-baseline regime produce byte-identical per-seed trace JSONL (determinism holds at any cap). (TC52)
- [ ] AC13 — Running the full `--check` suite prints all cases PASS, now including TC48–TC52 (53 cases), and the `--check` help text/count are updated.

## Data Model

```python
# arena/dynamic.py — the band table and resolver (single source of truth).

# Each value is (min_factor, max_factor) of robot top speed. "current" reproduces
# the existing SPEED_MIN_FACTOR / SPEED_MAX_FACTOR module constants exactly.
SPEED_REGIMES: dict[str, tuple[float, float]] = {
    "slow":    (0.3, 0.7),
    "matched": (0.3, 1.0),
    "current": (0.3, 1.5),   # the Mission baseline; == (SPEED_MIN_FACTOR, SPEED_MAX_FACTOR)
    "fast":    (0.5, 2.0),
}

# x positions for plot_speed_sweep: the "cap" is the max factor.
SPEED_REGIME_CAP: dict[str, float] = {k: v[1] for k, v in SPEED_REGIMES.items()}
```

## Contracts & Interfaces

Single source of truth for every cross-task seam. A task conforms to the exact name/signature/shape here.

### Shared types / tables

- `arena/dynamic.py::SPEED_REGIMES: dict[str, tuple[float, float]]` — owner: T1; consumers: T3, T4, T5, T6.
- `arena/dynamic.py::SPEED_REGIME_CAP: dict[str, float]` — owner: T1; consumer: T6 (x-axis).
- Existing module constants `SPEED_MIN_FACTOR = 0.3`, `SPEED_MAX_FACTOR = 1.5` remain the spawner defaults (unchanged values).

### Signatures

- `TrafficSpawner.__init__(self, env, robot, traffic_rng, motion_rng, dt, arena_w, arena_h, static_obstacles, *, speed_min_factor: float = SPEED_MIN_FACTOR, speed_max_factor: float = SPEED_MAX_FACTOR)` — owner: T1; consumer: T2. Validates `0 < speed_min_factor <= speed_max_factor`, else `ValueError`. Stores them; the speed draw becomes `uniform(self._speed_min_factor, self._speed_max_factor) * self._robot_top_speed` (same call site, same draw order).
- `Arena.__init__(self, yaml_path, seed, render=False, timeout_s=DEFAULT_TIMEOUT_S, traffic=False, *, speed_min_factor: float | None = None, speed_max_factor: float | None = None)` — owner: T2; consumers: T3, T7. `None` ⇒ omit the kwarg when constructing `TrafficSpawner` (spawner uses its own defaults), so direct `Arena(traffic=True)` stays byte-identical. Both-None or both-set only (a one-sided override is a `ValueError`).
- `arena/dynamic.py::resolve_speed_factors(regime: str | None, min_override: float | None, max_override: float | None) -> tuple[float, float]` — owner: T1; consumers: T3, T4, T5. Resolution: if overrides given, return them (validated); else look up `regime` (default `"current"`) in `SPEED_REGIMES`; unknown regime ⇒ `ValueError`. Mutual-exclusion (regime explicitly set AND overrides set) is enforced at the argparse layer, not here.

### Runner CLI contract (T3 `run_episode`, mirrored by T4 `run_experiment`)

- `--speed-regime {slow,matched,current,fast}` (default `current`).
- `--speed-min-factor FLOAT` + `--speed-max-factor FLOAT` (both-or-neither), in a `mutually_exclusive_group` with `--speed-regime`. Supplying a regime AND an override ⇒ argparse error (exit 2). One override without the other ⇒ exit 2.
- The runner resolves to `(min, max)` via `resolve_speed_factors` and passes both to `Arena(..., speed_min_factor=min, speed_max_factor=max)`.
- **No change** to the trace/metrics JSON schema.

### Results layout (owner: T5; consumer: T6)

- `results/speed_<regime>/<world_stem>/<label>/<seed>.json` and `…/_manifest.json` — produced by `run_experiment` children launched with `--results-dir results/speed_<regime>`. Read by `plot_speed_sweep` via `load_world_results("results/speed_<regime>", world_stem, replan_k=K)`.

### Manifest provenance (owner: T4; consumer: T6 optional)

- `run_experiment`'s `_manifest.json` gains `"speed_regime": str | null`, `"speed_min_factor": float`, `"speed_max_factor": float`.

### Planner sets (owner: T5)

- `FOCUS_SET = ("a_star_once", "d_star_lite", "dwa", "apf")` — non-replan, `replan_k=None`.
- `all` reuses `run_all.canonical_planner_set()` (imported, not re-listed) for the 11 `(algorithm, replan_k, label)` tuples.

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `arena/dynamic.py` | T1 | T2, T3, T4, T5, T6 (import only) |
| `arena/arena.py` (Arena class) | T2 | T7 |
| `arena/arena.py` (TCs + `_run_checks`) | T7 | — |
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
- **One-sided Arena override** (`speed_min_factor` set, `speed_max_factor` None or vice-versa): `Arena.__init__` raises `ValueError` — avoids a half-applied band.
- **Unknown regime via API**: `resolve_speed_factors` raises `ValueError` listing the valid keys.
- **Conflicting CLI flags** (`--speed-regime` + overrides, or a lone override): argparse exits 2 (mutually-exclusive group / explicit check), matching the runners' existing up-front-validation exit convention.
- **Sweep child failure**: `run_speed_sweep` mirrors `run_all` — a non-zero `run_experiment` exit is recorded, the sweep continues to the next (regime, planner), and the driver exits non-zero at the end listing the failures. An in-sim crash/timeout is exit 0 (recorded in the metrics JSON), not a runner failure.
- **Plotter on a missing regime subtree**: `load_world_results` already warns-not-raises on missing label dirs; `plot_speed_sweep` treats a regime with zero present episodes as a gap in that algorithm's line (annotated), never a crash. A regime subtree entirely absent is reported and the chart still renders from the present regimes.
- **Plotter with no data at all**: exit 1 with a clear message (mirrors `plot.py`).

## Testing Strategy

**Levels:** Unit (regime table, validation, resolver), Integration (Arena/runner determinism, sweep e2e smoke), plus the existing `--check` regression gate.

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC48 | `SPEED_REGIMES` contents + `resolve_speed_factors` precedence/defaults; `current == (0.3,1.5) ==` spawner constants; unknown regime raises | Unit | Exact band values; `resolve_speed_factors("current",None,None)==(0.3,1.5)`; override beats regime; `ValueError` on unknown key |
| TC49 | Spawner bound validation | Unit | `speed_min_factor<=0` and `speed_max_factor<speed_min_factor` each raise `ValueError`; `min==max` is allowed |
| TC50 | Baseline determinism preservation | Integration | `Arena(traffic=True)` vs explicit `(0.3,1.5)` give byte-identical `dynamic_obstacles_sha256` over N ticks; runner `--speed-regime current` trace == no-flag trace (byte-identical); `--speed-regime fast` differs |
| TC51 | Band is wired + controlled-experiment property | Unit | For one seed, two regimes give identical spawn `(x,y)` and heading per obstacle; obstacle speeds scale with the band (Fast max speed > Slow max speed) |
| TC52 | Non-baseline determinism | Integration | Two same-seed `Arena(traffic=True, speed_min_factor=0.5, speed_max_factor=2.0)` runs give identical sha256 sequences |
| TC-S1..n | `plot_speed_sweep --selfcheck` over synthetic per-regime JSON trees | Unit | Loader-per-regime aggregates failure_rate/median_time correctly; x = max-cap factor mapping; absent algorithms filtered (`n_present>0`); both charts render under Agg; matplotlib-absent guard exits 1; missing-regime gap handled |
| (regression) | Full `python arena/arena.py arena/arena_v1.yaml --check` | Integration | All 53 cases PASS, TC17–TC24 byte-identical (proves the baseline is untouched) |

**Test data:** TC48/49/51 are pure/near-pure (construct a spawner with a tiny static-obstacle list or use the arena world; no full episode). TC50/52 drive `Arena` over a fixed tick count with zero actions, comparing sha256 sequences (mirrors TC20). The plotter selfcheck builds synthetic `results/speed_<regime>/<stem>/<label>/<seed>.json` trees in a `TemporaryDirectory` (mirrors `plot.py`'s TC-P pattern) — no irsim.

**Run command:**
- TCs: `.venv\Scripts\Activate.ps1; python arena/arena.py arena/arena_v1.yaml --check`
- Plot selfcheck: `python -m runners.plot_speed_sweep --selfcheck`
- Smoke (AC9/AC10): `python -m runners.run_speed_sweep --world arena/arena_v1.yaml --algorithms focus --num-seeds 3` then `python -m runners.plot_speed_sweep --world arena/arena_v1.yaml --algorithms focus`

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Spawner speed-band params + regime table | — | high | `arena/dynamic.py` | Add keyword-only `speed_min_factor`/`speed_max_factor` to `TrafficSpawner.__init__` (defaults = existing `SPEED_MIN_FACTOR`/`SPEED_MAX_FACTOR`), validate `0 < min <= max` (else `ValueError`), store, and use them in the existing `_try_one_spawn` speed draw (same call site/order). Add `SPEED_REGIMES`, `SPEED_REGIME_CAP`, and `resolve_speed_factors(regime, min_override, max_override)` per Contracts. Do NOT change the draw order or count. Satisfies AC1–AC4. |
| T2 | Thread band through `Arena` | T1 | med | `arena/arena.py` (Arena class only) | Add keyword-only `speed_min_factor`/`speed_max_factor` (default `None`) to `Arena.__init__`; `None` ⇒ omit from the `TrafficSpawner(...)` call so the default path is byte-identical; both-set ⇒ pass through; one-sided ⇒ `ValueError`. No change to `reset()` RNG derivation. Satisfies AC5. |
| T3 | `run_episode` CLI knob | T1, T2 | med | `runners/run_episode.py` | Add `--speed-regime {slow,matched,current,fast}` (default `current`) in a mutually-exclusive group with `--speed-min-factor`/`--speed-max-factor` (both-or-neither). Resolve via `resolve_speed_factors` and pass to `Arena(...)`. Reject a lone override / conflicting flags (exit 2). No trace/metrics schema change. Update `RunnerArgs` + module docstring. Satisfies AC6, AC7. |
| T4 | `run_experiment` forward + manifest | T3 | med | `runners/run_experiment.py` | Mirror the T3 CLI flags; forward `--speed-regime` (or the float overrides) to each `run_episode` child verbatim. Resolve once to record `speed_regime`/`speed_min_factor`/`speed_max_factor` in `_manifest.json`. Update `RunnerArgs` + docstring. Satisfies AC8. |
| T5 | `run_speed_sweep` driver | T4 | med | `runners/run_speed_sweep.py` (new) | New module mirroring `run_all`'s subprocess pattern: `--world`, `--algorithms {focus,all}` (default `focus`), `--master-seed`/`--num-seeds`/`--jobs`/`--resume`/`--traffic` passthrough. For each regime in `SPEED_REGIMES`, for each planner in the selected set, shell `python -m runners.run_experiment --speed-regime <regime> --results-dir results/speed_<regime> ...`. Record per-(regime,planner) exit; continue past failures; exit non-zero if any failed. `FOCUS_SET` per Contracts; `all` imports `run_all.canonical_planner_set()`. Satisfies AC9. |
| T6 | `plot_speed_sweep` plotter + selfcheck | — | med | `runners/plot_speed_sweep.py` (new) | New read-only plotter: `--world`, `--algorithms {focus,all}`, `--results-dir` (root), `--replan-k`, `--out-dir`, `--selfcheck`. Reuse `runners.plot.load_world_results` once per regime subtree; build per-algorithm series of `failure_rate` and `median_time` across regimes; render two line charts (x = `SPEED_REGIME_CAP`, one line per algorithm with `n_present>0`) + `speed_sweep_summary.csv`. Lazy matplotlib import via `plot.ensure_matplotlib`. `--selfcheck` builds synthetic per-regime trees in a `TemporaryDirectory` (TC-S*). Conforms to the Contracts results layout. Satisfies AC10, AC11. |
| T7 | New `--check` TCs | T2, T3 | med | `arena/arena.py` (TC section + `_run_checks`) | Add `tc48`–`tc52` per the Testing Strategy, register them in `_run_checks`, and update the `--check` help text + count (48 → 53). TC50/TC52 mirror TC20's tick-and-compare-sha256 shape. Satisfies AC2–AC6, AC12, AC13. |
| T8 | Smoke run + findings note + docs | T1, T2, T3, T4, T5, T6, T7 | low | `CLAUDE.md`, `README.md`, `Mission.md`, `docs/plans/2026-06-23-obstacle-speed-sweep.findings.md` (new) | Run the full `--check` suite (confirm 53 PASS), the plot selfcheck, and the AC9/AC10 smoke; capture results. Write a short findings note: how to read the two charts, how to launch the full 50-seed sweep, and what the smoke proved (plumbing, not the hypothesis). Document the new knob/driver/plotter in CLAUDE.md + README, and add a Mission.md Phase 7 note that the speed-cap sweep tooling exists. |

Parallelism: T1 is the root. T2 and T6 can start once T1 lands (T6 only needs the `plot.py` loader + the layout contract, so it can run alongside T2–T5). T3 waits on T1+T2; T4 on T3; T5 on T4; T7 on T2+T3. T8 is the final integration/smoke/docs gate.

## Notes for Implementer

- **The single load-bearing invariant:** the speed draw must stay the *third* `traffic_rng` draw, made with the *same* `Generator.uniform(lo, hi)` call (only `lo`/`hi` become parameters). `uniform` consumes the same RNG bits regardless of bounds, so the baseline `(0.3, 1.5)` reproduces today's stream byte-for-byte. Do not reorder, add, or remove any draw, and do not branch the draw on the band. TC50 is the binding gate; if it fails, the change is wrong.
- **Why spawn positions are identical across regimes (a feature, exploit it in TC51):** overlap rejection (`_overlaps_robot_start`, `_overlaps_static`) is position-only and independent of speed, so for a fixed seed every regime gets the same spawn positions/headings — only velocity magnitude changes. This makes the sweep a clean controlled experiment; assert it.
- **`None`-means-default in Arena is load-bearing:** when both factors are `None`, omit them from the `TrafficSpawner(...)` kwargs entirely (don't pass `None` through) so the spawner's own defaults apply and `Arena(traffic=True)` is byte-identical. TC17–TC24 construct `Arena(traffic=True)` directly and must not change.
- **Don't touch the trace/metrics schema.** The regime is provenance — it belongs in `_manifest.json` only. Adding a key to the trace would break the determinism TCs and the plotter's fixed schema.
- **Plotter reuse, don't reimplement:** import `load_world_results`, `AlgoSummary`, `ensure_matplotlib`, `CANONICAL`, and the color map from `runners.plot`. The only new logic is "load four subtrees, line them up by regime cap, filter present algorithms, draw two line charts." Filter `n_present>0` before charting so a `focus` sweep doesn't draw 7 empty lines for the absent planners (the loader iterates all 11 and warns on missing dirs — that's expected/benign).
- **Cost expectation for the smoke:** D* Lite is the slowest planner (its traffic drive dominates wall time). Keep the smoke at ~3 seeds × 4 regimes on the `focus` set; the full 50-seed sweep is the user's to launch. State this in the findings note.
- **`all` set must not drift:** import `run_all.canonical_planner_set()` for the 11-planner case rather than re-listing labels, so a registry change can't desync the sweep from the study.
- **Rollback:** the change is additive and default-preserving — reverting the new modules + the keyword params restores exact prior behavior. The full `--check` suite passing at 53 (with TC17–TC24 byte-identical) is the safety net.
