# Filter impossible and degenerate seeds out of the comparison (issue #9)

**Goal:** Detect the "degenerate" traffic seeds that crash every planner in the first
few ticks regardless of what it plans, set them aside from the headline `failure_rate`
number, and record the decision reproducibly — so the cross-algorithm scatter measures
planner quality, not traffic that spawned on top of the start.

**Approach:** A read-only, headless, POST-HOC filter. A new `runners/filter_seeds.py`
CLI reads each planner's per-seed metrics JSON and (only on crash) its trace JSONL,
classifies each of the canonical 50 seeds, and writes a `results/<world_stem>/_seed_filter.json`
sidecar. The plotter (`runners/plot.py`) reads that sidecar, verifies it is fresh
against the current result bytes, and drops the flagged seeds from `failure_rate` and
the numeric per-algorithm stats while still rendering them, marked, on the B1
seed-difficulty heatmap. The classification core lives in a pure, stdlib-only
`runners/_seed_filter.py` shared by both. No irsim, no re-run of any episode.

## Scope

- **In scope:**
  - A pure classification core (`runners/_seed_filter.py`): the criterion, the sidecar
    schema, sidecar read/write, and the freshness (content-hash) check. Stdlib + json
    only; no `planners`/irsim/matplotlib import.
  - A CLI labeler (`runners/filter_seeds.py`): reads the result tree for one world,
    classifies every seed against the declared planner set, writes the sidecar, prints a
    verdict, and ships a `--selfcheck` synthetic-fixture suite (no irsim).
  - Plotter integration (`runners/plot.py`): read + freshness-verify the sidecar, drop
    flagged seeds from `failure_rate` and numeric stats, keep them marked on B1, surface
    the filter status as an on-figure footnote, add raw+filtered columns to `summary.csv`,
    and a `--filter/--no-filter` toggle. New `--selfcheck` cases.
  - Docs: a CLAUDE.md section, a README entry + usage, and the plotter note on how
    filtered seeds are handled in `failure_rate` (the issue's "done" list).
- **Out of scope:**
  - The spawn-overlap geometry test and the clairvoyant baseline (issue's other two
    candidate signals) — the user's "only instant die for every algo" steer makes the
    post-hoc all-crash-within-4s criterion sufficient, and neither can run post-hoc: the
    trace stores only a sha256 of obstacle positions, not raw `(x, y, vx, vy)`, so they
    would need to re-derive each seed's snapshot from irsim. (Kept as possible future
    cross-check TCs; see Notes.)
  - Adding a `crash_step` field to the runner's 7-field metrics schema (Codex's
    "metrics-native" ideal) — it would touch the determinism-audited `run_episode` and
    require backfilling every existing result tree, violating the post-hoc/read-only
    steer. The classifier is structured so a future `crash_step` field can replace the
    trace read without changing the criterion.
  - Changing the canonical master seed, the 50-seed count, seed derivation, or the arena
    traffic model. The filter never rejects a seed before it runs — it only labels
    already-recorded results.
  - Baking the filter into arena/seed derivation (the issue's alternative "where it
    lives" — rejected in favor of the post-hoc pass).

## Decisions

- **Criterion — "every algo dies instantly":** a seed is `degenerate` iff, for **all
  required planners**, the episode is a crash whose first crashed trace line is at
  `1 <= step <= WINDOW_STEPS` (step 0 is a hardcoded `crashed=False` sentinel, so the
  earliest observable crash is step 1). — User steer: intersection of all-fail and
  immediate-crash; both consultants concur the criterion must be explicit.
- **Window = 4 s of sim time = `WINDOW_STEPS = 40`** (`round(window_seconds / STEP_TIME)`,
  `STEP_TIME = 0.1` s mirroring irsim's `step_time`). — User chose 4 s. `--window-seconds`
  overrides; the resolved seconds/steps are recorded in the sidecar.
- **Required planner set = all 13** (11 canonical + `d_star_lite_oracle_h<H>` +
  `d_star_lite_predictive_h<H>`, `H` = `--predict-horizon`, default 10). — User chose
  "canonical 11 + 2 experimental". Made an **explicitly declared, recorded** list
  (`--planners`), per both consultants, so a canonical-only tree can be filtered
  deliberately with the reduced set recorded rather than silently no-op'd.
- **Missing data => `indeterminate` => never dropped.** A seed is dropped only when every
  required planner has data AND all instant-crashed. A planner that reaches goal, times
  out, crashes after 4 s, or errors counts as "survived" and one survivor keeps the seed.
  — User chose "indeterminate — never drop".
- **Drop from the denominator, but report both.** The headline `failure_rate` and the
  numeric per-algorithm stats exclude dropped seeds (user's choice). `summary.csv` ALSO
  carries the raw full-50 `failure_rate_raw` + `n_dropped`, and A1 footnotes the drop, so
  the raw number is always recoverable. — User chose "drop"; both consultants require the
  raw number stay visible (dropping is order-preserving — every degenerate seed adds one
  failure to every planner — so rankings are unchanged; only the contaminated absolute
  level moves).
- **Filtered seeds stay visible+marked on B1 only.** Excluded from every numeric stat;
  the B1 heatmap renders their columns marked "degenerate (filtered)". — User's Q4 choice.
- **Sidecar-mediated seam, hardened with a content-hash freshness contract.** filter_seeds
  writes the sidecar; plot.py reads it and re-hashes the consulted metrics files; on any
  mismatch/missing it renders UNFILTERED with a loud footnote — never half-applies. —
  User chose sidecar-mediated (Phase 4). Fable endorsed with this contract; Codex vetoed
  the *loose* sidecar toward metrics-native, resolved (under `--auto`) by adopting the
  freshness contract, which equals Codex's own listed "VALIDATED SIDECAR" option, plus
  report-both and the declared planner set. Metrics-native set aside (no crash-step field
  exists; adding one breaks the read-only steer).
- **The all-13 rule is a loud no-op, never a silent one.** On the standard
  `run_all -> plot` tree only the 11 canonical dirs exist, so every seed is indeterminate.
  filter_seeds prints an explicit verdict and returns a **distinct exit code** naming the
  missing dirs; the sidecar records `global_status="indeterminate"` + `missing_labels`;
  plot.py's footnote confesses it. — Both consultants flagged this trap.
- **Assumptions confirmed** (Phase 3, user declined to correct any): `STEP_TIME = 0.1`;
  experimental data is a prerequisite to drop anything (canonical-only run is a recorded
  no-op unless `--planners canonical`); the criterion is crash-specific; filtered seeds
  leave all numeric stats.

## Acceptance Criteria

- [ ] **AC1** `import runners._seed_filter` and `import runners.filter_seeds` pull neither
  `matplotlib` nor `irsim` (verified by `python -c "import sys, runners.filter_seeds;
  print('matplotlib' in sys.modules, 'irsim' in sys.modules)"` printing `False False`).
- [ ] **AC2** For a world whose result tree has all required label dirs, `filter_seeds`
  labels every seed in the manifest `derived_seeds` roster as exactly one of
  `degenerate` / `kept` / `indeterminate` per the criterion, and writes `_seed_filter.json`.
- [ ] **AC3** A seed where all 13 required planners crash at `step <= 40` is `degenerate`;
  a single survivor (goal, timeout, late crash at `step > 40`, or `planner_error`) makes
  it `kept`; any missing required planner (with no survivor) makes it `indeterminate`.
- [ ] **AC4** Window boundary: a crash whose first crashed line is at `step == 40` counts
  as instant; at `step == 41` it does not; the `step == 0` sentinel is never a crash.
- [ ] **AC5** The sidecar records `schema_version`, `criterion_id`, `world_stem`,
  `window_seconds`, `step_time`, `window_steps`, the exact `required_labels`, `git_sha`,
  `global_status`, `missing_labels`, `seed_order`, `dropped_seeds`, per-seed `seeds`
  (verdict + per-planner `{status, crash_step|null}`), and `consulted_hashes`
  (`"<label>/<seed>.json" -> sha256`). Written sort-keyed with no timestamp, so two runs
  over an unchanged tree produce a byte-identical sidecar.
- [ ] **AC6** When a required label dir is entirely absent, every seed is `indeterminate`,
  `dropped_seeds == []`, `global_status == "indeterminate"`, `missing_labels` names the
  absent dirs, filter_seeds prints a loud verdict and exits with the distinct
  "nothing determinable" code (not 0).
- [ ] **AC7** `plot.py` with a fresh sidecar present drops `dropped_seeds` from
  `failure_rate` and the numeric stats of every algorithm; the dropped seeds do NOT appear
  in any A1/A3/A4/B2/B3/B4 count, but DO appear on B1 marked "degenerate (filtered)".
- [ ] **AC8** Stale-sidecar safety: if any consulted metrics file's current sha256 differs
  from the sidecar's recorded hash (or a recorded file is gone), plot.py applies NO drop,
  prints a loud stderr warning, and every figure footnote reads "filter: STALE, ignored".
- [ ] **AC9** `summary.csv` gains `n_present_raw`, `failure_rate_raw`, `n_dropped`
  columns; for an unfiltered/absent-sidecar run they equal the filtered values with
  `n_dropped == 0`.
- [ ] **AC10** `--no-filter` makes plot.py ignore any sidecar (full 50 denominator);
  absent sidecar behaves the same, with the footnote reading "filter: absent".
- [ ] **AC11** `python -m runners.filter_seeds --selfcheck` runs the TC-F suite on
  synthetic fixtures (no irsim) and exits 0 iff all pass; `--world` is optional when
  `--selfcheck` is given.
- [ ] **AC12** `python -m runners.plot --selfcheck` still passes, now including the new
  drop/stale/absent/raw-column cases; the matplotlib-at-import guard still holds.
- [ ] **AC13** Determinism preserved: a `plot` run with `--no-filter` (or no sidecar) is
  byte-identical to the pre-change plotter output for the same tree (the filter is purely
  additive when off).

## Contracts & Interfaces

Single source of truth for every seam. The three code tasks (T1 core, T2 CLI, T3 plotter)
meet only here.

### Shared types & constants — `runners/_seed_filter.py` (owner: T1)

```python
SEED_FILTER_NAME = "_seed_filter.json"   # underscore prefix => skipped by plot's [0-9]*.json glob
STEP_TIME = 0.1                          # s; mirrors irsim step_time (documented constant)
DEFAULT_WINDOW_SECONDS = 4.0
SCHEMA_VERSION = 1
CRITERION_ID = "all_required_instant_crash/v1"

PlannerStatus = str   # "instant_crash" | "survived" | "missing"
SeedVerdict   = str   # "degenerate"    | "kept"     | "indeterminate"

@dataclass(frozen=True)
class PlannerEvidence:
    label: str
    status: str            # PlannerStatus
    crash_step: int | None # first crashed step when status=="instant_crash", else None

@dataclass(frozen=True)
class SeedVerdictRow:
    seed: int
    verdict: str                     # SeedVerdict
    planners: tuple[PlannerEvidence, ...]  # sorted by label

@dataclass(frozen=True)
class SeedFilter:
    schema_version: int
    criterion_id: str
    world_stem: str
    window_seconds: float
    step_time: float
    window_steps: int
    required_labels: tuple[str, ...]
    git_sha: str | None
    global_status: str               # "ok" | "indeterminate"
    missing_labels: tuple[str, ...]
    seed_order: tuple[int, ...]
    dropped_seeds: tuple[int, ...]
    rows: tuple[SeedVerdictRow, ...]
    consulted_hashes: dict[str, str] # "<label>/<seed>.json" -> sha256
```

### Signatures (owner: T1; consumers noted)

- `window_steps(window_seconds: float, step_time: float = STEP_TIME) -> int` —
  `round(window_seconds / step_time)`. Consumers: T2.
- `crash_step_from_trace(lines: Iterable[dict]) -> int | None` — first parsed trace line
  whose `crashed` is truthy returns its `step`; `None` if none. Pure. Consumers: T2.
- `classify_planner(*, present: bool, crashed: bool, crash_step: int | None,
  window_steps: int) -> str` — returns a `PlannerStatus`. `instant_crash` iff
  `present and crashed and crash_step is not None and 1 <= crash_step <= window_steps`;
  `missing` iff `not present`; else `survived`. Pure. Consumers: T2.
- `classify_seed(evidence: Sequence[PlannerEvidence]) -> str` — `kept` if any `survived`;
  else `indeterminate` if any `missing`; else `degenerate`. Pure. Consumers: T2.
- `write_seed_filter(obj: SeedFilter, path: str | Path) -> None` — JSON, `sort_keys=True`,
  trailing newline, no timestamp. Consumers: T2.
- `read_seed_filter(path: str | Path) -> SeedFilter | None` — parse; `None` on
  absent/unreadable/schema-mismatch (warns to stderr). Consumers: T3.
- `sidecar_is_fresh(obj: SeedFilter, results_root: str | Path, world_stem: str) ->
  tuple[bool, list[str]]` — re-hash each `consulted_hashes` path under
  `<results_root>/<world_stem>/`; return `(all_match, stale_or_missing_paths)`.
  Consumers: T3.
- `file_sha256(path: str | Path) -> str | None` — hex digest, `None` if unreadable.
  Consumers: T1(write via T2), T3.

### Sidecar JSON — the on-disk contract (producer: T2, consumer: T3)

Path: `<results_dir>/<world_stem>/_seed_filter.json`. Shape = `SeedFilter` serialized
(dataclasses -> dict, tuples -> lists). `consulted_hashes` keys are POSIX-relative
`"<label>/<seed>.json"` so they are stable across OSes.

### `runners/plot.py` seams (owner: T3)

- `load_world_results(results_dir, world_stem, *, replan_k=DEFAULT_REPLAN_K,
  expected=DEFAULT_EXPECTED_SEEDS, dropped_seeds: frozenset[int] = frozenset()) ->
  WorldResults` — new `dropped_seeds` keyword; excludes those seeds from the FILTERED
  counts. `per_seed` stays complete (dropped seeds keep their real outcome for B1).
- `AlgoSummary` gains `n_present_raw: int` and `failure_rate_raw: float`.
  `n_present`/`failure_rate` become the FILTERED (headline) values; every existing chart
  reading them auto-adopts the drop. Success `times`/`path_lengths` are unaffected
  (degenerate seeds are crashes, never successes).
- `WorldResults` gains `degenerate_seeds: tuple[int, ...]` and
  `filter_status: str` (`"applied" | "stale" | "indeterminate" | "absent" | "off"`).
- `SUMMARY_CSV_COLUMNS` gains `n_present_raw`, `failure_rate_raw`, `n_dropped`
  (inserted after the existing `failure_rate`).
- CLI: `--filter/--no-filter` (default: apply a fresh sidecar if present).

### File ownership

| File | Owner | Consumers |
|------|-------|-----------|
| `runners/_seed_filter.py` (new) | T1 | T2, T3 |
| `runners/filter_seeds.py` (new) | T2 | — |
| `runners/plot.py` (modify) | T3 | — |
| `CLAUDE.md`, `README.md` (modify) | T4 | — |

### Naming (exact strings that must match across tasks)

- Sidecar filename: `_seed_filter.json` (the `SEED_FILTER_NAME` constant — never a literal).
- `global_status` values: `"ok"`, `"indeterminate"`.
- `PlannerStatus`: `"instant_crash"`, `"survived"`, `"missing"`.
- `SeedVerdict`: `"degenerate"`, `"kept"`, `"indeterminate"`.
- `plot` `filter_status`: `"applied"`, `"stale"`, `"indeterminate"`, `"absent"`, `"off"`.
- Experimental label build: `algorithm_label("d_star_lite_oracle", None, H)` /
  `algorithm_label("d_star_lite_predictive", None, H)` (yields `..._h<H>`).

## Data Model

**Required-label construction (T2).** Lazy-import inside a function (importing `planners`
pulls irsim+matplotlib — see Notes):

```python
def required_labels(predict_horizon: int, replan_k: int, planner_set: str) -> list[str]:
    from planners import algorithm_label            # lazy — pulls irsim
    from runners.run_all import canonical_planner_set
    labels = [label for (_name, _k, label) in canonical_planner_set()]  # the 11, _k5 folded
    if planner_set == "all13":
        labels += [
            algorithm_label("d_star_lite_oracle", None, predict_horizon),
            algorithm_label("d_star_lite_predictive", None, predict_horizon),
        ]
    return labels
```

**Seed roster (T2).** Read from any required label's `_manifest.json` `derived_seeds`
(the canonical 50, in derivation order) — same source `plot.load_world_results` uses.
If no manifest is found, fall back to the sorted union of numeric stems present and note
it in the printed verdict.

**Evidence collection per (seed, label) (T2):**
1. `present = (<seed>.json exists)`. If absent -> `PlannerEvidence(label, "missing", None)`.
   No hash recorded for a missing file (it can't go stale).
2. Read `<seed>.json`; record its `file_sha256` under `consulted_hashes["<label>/<seed>.json"]`.
3. `crashed = bool(rec["crashed"])`. If not crashed -> `"survived"`.
4. If crashed: read `<seed>.trace.jsonl` (first `window_steps + 1` lines are sufficient,
   but reading the whole file is fine — crash traces terminate at the crash and are tiny),
   `crash_step = crash_step_from_trace(...)`. `classify_planner(...)` -> `instant_crash`
   iff `1 <= crash_step <= window_steps`, else `survived` (late crash, or trace
   missing/unreadable — cannot confirm instant, so conservative).

## Error Handling

- **Sidecar absent (plot):** no drop; `filter_status="absent"`; footnote "filter: absent".
- **Sidecar present but stale (plot):** no drop; loud stderr warning listing the
  stale/missing paths; `filter_status="stale"`; footnote "filter: STALE, ignored — re-run
  filter_seeds". Never partial-apply.
- **Sidecar `global_status=="indeterminate"` (plot):** `dropped_seeds` is `[]` anyway;
  `filter_status="indeterminate"`; footnote "filter present but indeterminate — nothing
  dropped; missing: <labels>".
- **`--no-filter` (plot):** `filter_status="off"`; footnote "filter: off".
- **Unreadable metrics/trace JSON (filter_seeds):** warn to stderr, treat that planner as
  `missing` for that seed (blocks a drop — conservative), continue.
- **No result tree / no roster (filter_seeds):** print an error, write no sidecar, exit 1.
- **All seeds indeterminate (filter_seeds):** write the sidecar with
  `global_status="indeterminate"`, print the missing dirs, exit the distinct
  "nothing determinable" code (see Notes for the code map).
- **CLI/validation error (either):** exit 2, before any file work.

## Testing Strategy

**Levels:** Unit (pure core), Integration (CLI + plotter over synthetic trees), all under
the two in-module `--selfcheck` suites — no irsim, no real episodes, mirroring the repo's
existing `plot --selfcheck`. No changes to `arena/arena.py --check` (keeps its ~50-min
suite and the script-mode import order untouched).

`runners/filter_seeds.py --selfcheck` (TC-F*):

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC-F1 | All 13 crash at step<=40 for a seed | Unit | seed verdict `degenerate`, in `dropped_seeds` |
| TC-F2 | One planner reaches goal on that seed | Unit | verdict `kept`, not dropped |
| TC-F3 | One planner crashes at step 41 (late) | Unit | that planner `survived`; verdict `kept` |
| TC-F4 | One required planner has no `<seed>.json` | Unit | that planner `missing`; verdict `indeterminate`; not dropped |
| TC-F5 | Window boundary: crash step 40 vs 41 vs 0 sentinel | Unit | 40 instant, 41 not, step-0 sentinel never a crash |
| TC-F6 | `planner_error` (no trace) on a seed | Unit | that planner `survived` (blocks drop) |
| TC-F7 | A whole required label dir absent | Integration | `global_status=="indeterminate"`, `missing_labels` set, distinct exit code, `dropped_seeds==[]` |
| TC-F8 | Sidecar round-trip write->read | Unit | `read_seed_filter(write(...))` equals the original |
| TC-F9 | Determinism | Integration | two runs over one fixture tree -> byte-identical sidecar |
| TC-F10 | `--planners canonical` on an 11-only tree | Integration | required set is the 11; a seed can be `degenerate`; recorded set is the 11 |
| TC-F11 | Freshness: mutate one metrics file after write | Unit | `sidecar_is_fresh(...)` returns `(False, [that path])` |
| TC-F12 | Headless import guard | Unit | neither matplotlib nor irsim in `sys.modules` after import |

`runners/plot.py --selfcheck` additions (TC-P12+):

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC-P12 | Fresh sidecar with 1 degenerate seed | Integration | that seed excluded from every algo's `failure_rate`/`n_present`; present+marked on B1; `n_dropped==1` |
| TC-P13 | Stale sidecar (mutated metrics) | Integration | no drop; `filter_status=="stale"`; charts still render |
| TC-P14 | Absent sidecar / `--no-filter` | Integration | full-50 denominator; `n_present_raw==n_present`, `n_dropped==0`; byte-identical to pre-change output (AC13) |
| TC-P15 | `summary.csv` raw+filtered columns | Unit | `n_present_raw`, `failure_rate_raw`, `n_dropped` present and correct |
| TC-P16 | Indeterminate-status sidecar | Integration | no drop; footnote confesses; no raise |

**Test data:** synthetic result trees built in a `TemporaryDirectory` — `<seed>.json`
metrics (reuse plot's `_make_record` shape), tiny `<seed>.trace.jsonl` files with a
`crashed` flag at a chosen step, and a `_manifest.json` carrying `derived_seeds`. Factory
helpers live in each module's selfcheck block.

**Run command:** `python -m runners.filter_seeds --selfcheck` and
`python -m runners.plot --selfcheck` (both exit 0 iff all pass).

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|-----------|------|-------|-------------|
| T1 | Pure classification core | — | med | `runners/_seed_filter.py` | Implement the constants, dataclasses, and pure functions in Contracts & Interfaces (`window_steps`, `crash_step_from_trace`, `classify_planner`, `classify_seed`, `write_seed_filter`, `read_seed_filter`, `sidecar_is_fresh`, `file_sha256`). Stdlib + json ONLY — no `planners`/irsim/matplotlib import anywhere in the module. Deterministic, sort-keyed sidecar, no timestamp. Satisfies AC1 (its half), AC3–AC5, AC8's hash primitive. |
| T2 | CLI labeler + `--selfcheck` | T1 | med | `runners/filter_seeds.py` | argparse CLI (`--world`, `--results-dir` default "results", `--replan-k` default 5, `--predict-horizon` default 10, `--window-seconds` default 4.0, `--planners {all13,canonical}` default all13, `--selfcheck`). Build `required_labels` (lazy-import `planners`/`run_all`), read the seed roster, collect per-(seed,label) evidence (metrics first, trace only on crash), classify, assemble the `SeedFilter`, write the sidecar, print a verdict line, and use the exit-code map (0 ok / distinct code when all-indeterminate / 1 no-tree / 2 CLI). Ship TC-F1..TC-F12 in-module. Stays headless (lazy-import). Satisfies AC1, AC2, AC6, AC11, and the TC-F table. |
| T3 | Plotter integration + `--selfcheck` | T1 | high | `runners/plot.py` | Thread `dropped_seeds` through `load_world_results`; add `AlgoSummary.n_present_raw`/`failure_rate_raw` (make `n_present`/`failure_rate` the filtered headline), `WorldResults.degenerate_seeds`/`filter_status`; read+freshness-verify the sidecar in `main` (import `read_seed_filter`/`sidecar_is_fresh`/`SEED_FILTER_NAME` from `runners._seed_filter` — a headless import); `--filter/--no-filter`; the status footnote on A1 (and where sensible other charts); B1 "degenerate (filtered)" column marking + legend entry; `SUMMARY_CSV_COLUMNS` raw/filtered/n_dropped. Add TC-P12..TC-P16 to the selfcheck registry. Preserve AC13 (off/absent == pre-change bytes) and the matplotlib-import guard. Satisfies AC7–AC10, AC12, AC13, and the TC-P table. |
| T4 | Docs | T1, T2, T3 | low | `CLAUDE.md`, `README.md` | New CLAUDE.md section "The degenerate-seed filter (issue #9)" (criterion, sidecar schema, freshness contract, the all-13 no-op trap + `--planners`, exit codes). README: add `runners/filter_seeds.py` to the layout + a usage blurb, and the plotter note on how filtered seeds are handled in `failure_rate` (the issue's "done" item). Match repo voice; no AI attribution. |

T2 and T3 both depend only on T1 and touch different files, so they run in parallel after
T1. T4 waits on all three (it documents final behavior).

## Notes for Implementer

- **Headless import discipline (load-bearing).** Importing `planners` transitively pulls
  irsim AND matplotlib (`planners -> manual_astar -> irsim -> matplotlib.pyplot / TkAgg`).
  `runners/_seed_filter.py` must import NOTHING from `planners`. `runners/filter_seeds.py`
  must lazy-import `algorithm_label` / `canonical_planner_set` INSIDE functions (as
  `plot.py` already does). `plot.py` importing from `runners._seed_filter` is safe (that
  module is planners-free). Guard with the `python -c "... 'matplotlib' in sys.modules ..."`
  check (AC1/AC12).
- **Freshness scope.** Hash only the consulted `<seed>.json` **metrics** files (not
  traces). `wallclock_per_step` is a `perf_counter` mean, so any re-run of any episode
  changes its metrics bytes and trips the hash; a trace cannot change without its metrics
  changing (same deterministic episode). This keeps the hash set to <=13x50 tiny files.
- **The step-0 sentinel.** Trace step 0 is the post-reset sentinel with `crashed=False`
  hardcoded, so `crash_step_from_trace` naturally returns the first real crash at step>=1;
  `classify_planner`'s `1 <= crash_step` lower bound is belt-and-suspenders.
- **Exit-code map (filter_seeds), documented in the module docstring:** `0` = sidecar
  written and the roster was determinable (>=1 seed `degenerate` or `kept`); `3` = sidecar
  written but every seed `indeterminate` (a required label dir is missing — the loud
  no-op); `1` = no result tree / no seed roster (nothing written); `2` = CLI/validation.
  `3` is distinct so a script can tell "filter ran but decided nothing" from "filter ran
  and kept everything".
- **Order-preserving drop (why this is safe).** A degenerate seed is a crash for every
  planner, so removing D such seeds maps each rate `F/50 -> (F-D)/(50-D)` monotonically —
  planner rankings cannot change, only the contaminated absolute level. Keep the raw
  numbers in `summary.csv` so the transformation is auditable.
- **Existing data reality (verification target).** `results/arena_v1/` currently holds the
  11 canonical dirs + `d_star_lite_predictive_h10` but NOT `d_star_lite_oracle_h10`, so a
  default `filter_seeds --world arena/arena_v1.yaml` will legitimately return
  `indeterminate` (exit 3, missing `d_star_lite_oracle_h10`). This is the correct loud
  no-op and is a good manual smoke: run it, confirm the verdict + exit code, then run
  `plot` and confirm the "indeterminate" footnote. To actually drop seeds, run the oracle
  at h10 first, or use `--planners canonical` to filter on the 11 deliberately.
- **Do not** reintroduce the `d_star_lite_2` stray dir or any `_tmp_*` scratch into the
  required set — required labels come only from `canonical_planner_set()` + the two
  horizon-suffixed experimental keys.
- **Future hooks (out of scope, do not build):** if a `crash_step` field is ever added to
  the metrics JSON, `classify_planner` can read it instead of the trace with no criterion
  change; the spawn-overlap geometry test (`Arena.initial_dynamic_snapshot`) could become
  a `--selfcheck`-adjacent cross-check TC someday.
- **Rollback:** the change is purely additive when the filter is off/absent (AC13) — two
  new files plus additive plot.py fields/flags. Reverting = delete the two new files and
  the plot.py additions; no result data or determinism guarantee is touched.
