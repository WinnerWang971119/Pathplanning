# Filter impossible and degenerate seeds out of the comparison (issue #9)

**Goal:** Detect the "degenerate" traffic seeds that crash every planner in the first
few ticks regardless of what it plans, set them aside from the headline `failure_rate`
number, and record the decision reproducibly — so the cross-algorithm scatter measures
planner quality, not traffic that spawned on top of the start.

**Approach:** A read-only, headless, POST-HOC filter. A new `runners/filter_seeds.py`
CLI reads each planner's per-seed metrics JSON and (only on crash) its trace JSONL,
classifies each of the canonical 50 seeds, and writes a `results/<world_stem>/_seed_filter.json`
sidecar. The plotter (`runners/plot.py`) reads that sidecar, verifies it is fresh
against the current result bytes AND covers the label set it is plotting, and drops the
flagged seeds from `failure_rate` and the numeric per-algorithm stats while still
rendering them, marked, on the B1 seed-difficulty heatmap. The classification core lives
in a pure, stdlib-only `runners/_seed_filter.py` shared by both. No irsim, no re-run of
any episode.

## Scope

- **In scope:**
  - A pure classification core (`runners/_seed_filter.py`): the criterion, the sidecar
    schema, sidecar read/write, the freshness (content-hash + label-coverage +
    absent-path) check, and the pure-stdlib required-label builder. Stdlib + json + hashlib
    only; NO `planners`/irsim/matplotlib import anywhere.
  - A CLI labeler (`runners/filter_seeds.py`): reads the result tree for one world,
    classifies every seed against the declared planner set, writes the sidecar, prints a
    verdict, and ships a `--selfcheck` synthetic-fixture suite (no irsim).
  - Plotter integration (`runners/plot.py`): read + freshness/coverage-verify the sidecar,
    drop flagged seeds from `failure_rate` and numeric stats, keep them marked on B1,
    surface the filter status as an on-figure footnote, add raw+filtered columns to
    `summary.csv`, and a `--filter/--no-filter` toggle. New `--selfcheck` cases.
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
- **Window = 4 s of sim time = `WINDOW_STEPS = 40`** (`round(window_seconds / step_time)`,
  `step_time = 0.1` s mirroring irsim's `step_time`). — User chose 4 s. `--window-seconds`
  overrides; `--step-time` overrides the tick size (default 0.1; see the step-time note in
  Notes); the resolved seconds/step-time/steps are recorded in the sidecar.
- **Required planner set = all 13** (11 canonical + `d_star_lite_oracle_h<H>` +
  `d_star_lite_predictive_h<H>`, `H` = `--predict-horizon`, default 10). — User chose
  "canonical 11 + 2 experimental". Made an **explicitly declared, recorded** list
  (`--planners {all13,canonical}`), per both consultants, so a canonical-only tree can be
  filtered deliberately with the reduced set recorded rather than silently no-op'd.
- **Missing data => `indeterminate` => never dropped.** A seed is dropped only when every
  required planner has data AND all instant-crashed. A planner that reaches goal, times
  out, crashes after 4 s, errors, or is a DNF (wallclock-killed — it provably ran past
  4 s) counts as "survived", and one survivor keeps the seed. — User chose "indeterminate
  — never drop".
- **Drop from the denominator, but report both.** The headline `failure_rate` and the
  numeric per-algorithm stats exclude dropped seeds (user's choice). It is a CONDITIONAL
  rate (conditioned on non-degenerate); A1 labels it as such ("degenerate seeds excluded;
  N dropped"). `summary.csv` ALSO carries the raw full-50 `failure_rate_raw` + `n_dropped`,
  and the A1 footnote states the raw number, so it is always recoverable and prominent. —
  User chose "drop"; both consultants require the raw number stay visible (dropping is
  order-preserving — every degenerate seed adds one failure to every planner — so rankings
  are unchanged; only the contaminated absolute level moves).
- **Filtered seeds stay visible+marked on B1 only.** Excluded from every numeric stat;
  the B1 heatmap renders their columns marked "degenerate (filtered)". — User's Q4 choice.
- **Sidecar-mediated seam, hardened with a content-hash + label-coverage freshness
  contract.** filter_seeds writes the sidecar; plot.py reads it, re-hashes the consulted
  metrics files, checks the roster is unchanged and recorded-absent files are still
  absent, and checks the label set it is plotting is covered by the sidecar's
  `required_labels`; on any mismatch it renders UNFILTERED with a loud footnote — never
  half-applies. — User chose sidecar-mediated (Phase 4). Fable endorsed with this
  contract; Codex vetoed the *loose* sidecar toward metrics-native, resolved (under
  `--auto`) by adopting the freshness contract, which equals Codex's own listed "VALIDATED
  SIDECAR" option, plus report-both and the declared planner set. **Why sidecar-as-input
  (not sidecar-as-report + plot-recomputes):** the user chose it in Phase 4; it keeps
  plot.py a metrics-JSON reader that never parses traces; and the sidecar IS the recorded
  provenance artifact the issue asks for. Metrics-native set aside (no crash-step field
  exists; adding one breaks the read-only steer).
- **The all-13 rule is a loud no-op, never a silent one.** On the standard
  `run_all -> plot` tree only the 11 canonical dirs exist, so every seed is indeterminate.
  filter_seeds prints an explicit verdict and returns a **distinct exit code** naming the
  missing dirs; the sidecar records `global_status="indeterminate"` + `missing_labels`;
  plot.py's footnote confesses it. — Both consultants flagged this trap.
- **Cross-manifest experiment consensus.** The seed roster is read from the first required
  label's manifest in CANONICAL order (matching `plot.load_world_results`); every found
  required-label manifest must agree on `derived_seeds`, `world_stem`, `traffic`, and the
  speed provenance (`speed_regime`/`speed_min_factor`/`speed_max_factor`). On disagreement
  filter_seeds warns naming the odd label and sets `global_status="indeterminate"` (refuses
  to mix experiments); a `git_sha` mismatch is a warning only (benign). — CR6 (judge
  SYNTHESIS).
- **Assumptions confirmed** (Phase 3, user declined to correct any): `step_time = 0.1`;
  experimental data is a prerequisite to drop anything (canonical-only run is a recorded
  no-op unless `--planners canonical`); the criterion is crash-specific; filtered seeds
  leave all numeric stats.

## Acceptance Criteria

- [ ] **AC1** `import runners._seed_filter` and `import runners.filter_seeds` pull neither
  `matplotlib` nor `irsim`, and calling the label builder / a full `--selfcheck` run pulls
  neither either (verified in a fresh subprocess:
  `python -c "import sys, runners.filter_seeds as f; f.build_required_labels(10, 5, 'all13');
  print('matplotlib' in sys.modules, 'irsim' in sys.modules)"` prints `False False`).
- [ ] **AC2** For a world whose result tree has all required label dirs, `filter_seeds`
  labels every seed in the manifest `derived_seeds` roster as exactly one of
  `degenerate` / `kept` / `indeterminate` per the criterion, and writes `_seed_filter.json`.
- [ ] **AC3** A seed where all required planners crash at `step <= 40` is `degenerate`;
  a single survivor (goal, timeout, late crash at `step > 40`, `planner_error`, or a DNF
  roster entry) makes it `kept`; any required planner with an absent metrics file (and no
  survivor) makes it `indeterminate`.
- [ ] **AC4** Window boundary: a crash whose first crashed line is at `step == 40` counts
  as instant; at `step == 41` it does not; the `step == 0` sentinel is never a crash.
- [ ] **AC5** The sidecar records `schema_version`, `criterion_id`, `world_stem`,
  `window_seconds`, `step_time`, `window_steps`, `replan_k`, `predict_horizon`, `traffic`,
  the speed provenance, the exact `required_labels`, `roster_is_canonical`, `git_sha`,
  `global_status`, `missing_labels`, `seed_order`, `dropped_seeds`, per-seed `rows`
  (verdict + per-planner `{status, crash_step|null}`), `consulted_hashes`
  (`"<label>/<seed>.json" -> sha256`), and `absent_files` (`"<label>/<seed>.json"` that were
  missing at write time). Written sort-keyed with no timestamp, so two runs over an
  unchanged tree produce a byte-identical sidecar.
- [ ] **AC6** When a required label dir is entirely absent, no seed is `degenerate`
  (`dropped_seeds == []`), `global_status == "indeterminate"`, and `missing_labels` names
  the absent dirs; seeds that have a surviving planner still classify as `kept`.
  filter_seeds prints a loud verdict and exits with the distinct "nothing determinable"
  code (not 0).
- [ ] **AC7** `plot.py` with a fresh, covering sidecar drops `dropped_seeds` from
  `failure_rate` and the numeric stats of every algorithm; the dropped seeds do NOT appear
  in any A1/A3/A4/B2/B4 count, but DO appear on B1 marked "degenerate (filtered)". **B3 is
  exempt** — its `wallclock_per_step` comes from the separate `__wallclock__` subtree
  (a different, short seed subset) and is a per-step timing sample, not an outcome, so a
  degenerate seed's step-timing stays valid.
- [ ] **AC8** Freshness/coverage safety: plot applies NO drop (and every A1 footnote reads
  "filter: STALE, ignored") when ANY of — a consulted metrics file's current sha256
  differs from the recorded hash, a recorded-absent file now exists, the manifest roster
  changed, or the label set plot is loading (from its own `--replan-k`/`--predict-horizon`)
  is not covered by the sidecar's `required_labels`. It prints a loud stderr warning naming
  the cause. Never partial-apply.
- [ ] **AC9** `summary.csv` gains `n_present_raw`, `failure_rate_raw`, `n_dropped`
  columns (after `failure_rate`); for an unfiltered/absent-sidecar run they equal the
  filtered values with `n_dropped == 0`.
- [ ] **AC10** `--no-filter` makes plot.py ignore any sidecar (full 50 denominator,
  footnote "filter: off"); an absent sidecar behaves the same with footnote "filter: absent".
- [ ] **AC11** `python -m runners.filter_seeds --selfcheck` runs the TC-F suite on
  synthetic fixtures (no irsim) and exits 0 iff all pass; `--world` is optional when
  `--selfcheck` is given.
- [ ] **AC12** `python -m runners.plot --selfcheck` still passes, now including the new
  drop/stale/absent/off/raw-column cases; the matplotlib-at-import guard still holds; and
  `runners.plot_speed_sweep --selfcheck` + `runners.plot_horizon_sweep --selfcheck` (which
  consume `plot.load_world_results` / `plot.CANONICAL`) still pass as a regression gate.
- [ ] **AC13** Determinism preserved (additive-when-off): within the NEW code, a
  `--no-filter` run and an absent-sidecar run produce byte-identical output as each other,
  and every numeric value then equals its raw value with `n_dropped == 0`. (The always-on
  footnote + three new `summary.csv` columns are present by design; there is no comparison
  to any pre-change golden output.)
- [ ] **AC14** Cross-manifest consensus (CR6): if two required-label manifests disagree on
  `derived_seeds`/`world_stem`/`traffic`/speed provenance, filter_seeds warns naming the
  odd label and sets `global_status="indeterminate"` (no drop); a `git_sha` mismatch warns
  only.
- [ ] **AC15** Label-parity guard (CR7): the pure-stdlib `build_required_labels` output
  equals `canonical_planner_set()` labels + the two horizon-suffixed experimental keys —
  asserted in a subprocess (where importing `planners` is allowed), so a future registry
  change fails loud instead of silently checking the wrong labels.

## Contracts & Interfaces

Single source of truth for every seam. The three code tasks (T1 core, T2 CLI, T3 plotter)
meet only here.

### Shared types & constants — `runners/_seed_filter.py` (owner: T1)

```python
SEED_FILTER_NAME = "_seed_filter.json"   # world-level sidecar (a SIBLING of the label dirs)
DEFAULT_STEP_TIME = 0.1                   # s; mirrors irsim step_time (documented; see Notes)
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
    replan_k: int
    predict_horizon: int
    traffic: bool
    speed_regime: str | None
    speed_min_factor: float
    speed_max_factor: float
    required_labels: tuple[str, ...]
    roster_is_canonical: bool        # True iff roster came from a manifest's derived_seeds
    git_sha: str | None
    global_status: str               # "ok" | "indeterminate"
    missing_labels: tuple[str, ...]
    seed_order: tuple[int, ...]
    dropped_seeds: tuple[int, ...]
    rows: tuple[SeedVerdictRow, ...]
    consulted_hashes: dict[str, str] # "<label>/<seed>.json" -> sha256 (present files)
    absent_files: tuple[str, ...]    # "<label>/<seed>.json" that were missing at write time
```

### Signatures (owner: T1; consumers noted)

- `build_required_labels(predict_horizon: int, replan_k: int, planner_set: str) -> list[str]`
  — PURE stdlib string logic (NO `planners` import). Rebuilds the 11 canonical labels from
  a frozen hand-listed order (folding `_k<replan_k>` for the replan families) and, for
  `"all13"`, appends `d_star_lite_oracle_h<H>` and `d_star_lite_predictive_h<H>`. A comment
  pins the source (`run_all._CANONICAL_ORDER` + `algorithm_label`'s `_k`/`_h` folding) as
  frozen; parity is guarded by AC15's subprocess check. Consumers: T2.
- `window_steps(window_seconds: float, step_time: float) -> int` —
  `round(window_seconds / step_time)`. Consumers: T2.
- `crash_step_from_trace(lines: Iterable[dict]) -> int | None` — first parsed trace line
  whose `crashed` is truthy returns its `step`; `None` if none. Pure. Consumers: T2.
- `classify_planner(*, present: bool, crashed: bool, crash_step: int | None,
  window_steps: int) -> str` — returns a `PlannerStatus`. `instant_crash` iff
  `present and crashed and crash_step is not None and 1 <= crash_step <= window_steps`;
  `missing` iff `not present` (absent METRICS file only); else `survived`. Pure. Consumers: T2.
- `classify_seed(evidence: Sequence[PlannerEvidence]) -> str` — `kept` if any `survived`;
  else `indeterminate` if any `missing`; else `degenerate`. (Per-seed precedence: a
  survivor is dispositive regardless of a missing planner.) Pure. Consumers: T2.
- `write_seed_filter(obj: SeedFilter, path: str | Path) -> None` — JSON, `sort_keys=True`,
  trailing newline, no timestamp. Consumers: T2.
- `read_seed_filter(path: str | Path) -> SeedFilter | None` — parse; `None` on
  absent/unreadable/`schema_version` mismatch (warns to stderr). Consumers: T3.
- `sidecar_is_fresh(obj: SeedFilter, results_root: str | Path) -> tuple[bool, list[str]]`
  — `world_stem` is taken from `obj`; re-hash each `consulted_hashes` path under
  `<results_root>/<obj.world_stem>/`, verify every `absent_files` path is still absent, and
  verify the roster (`derived_seeds` from the first CANONICAL-order required manifest)
  equals `obj.seed_order`; return `(all_ok, reasons)`. Consumers: T3.
- `sidecar_covers(obj: SeedFilter, plotted_labels: Sequence[str]) -> bool` — True iff every
  plotted label is in `obj.required_labels`. Consumers: T3.
- `file_sha256(path: str | Path) -> str | None` — hex digest, `None` if unreadable.
  Consumers: T1 (write via T2), T3.

### Sidecar JSON — the on-disk contract (producer: T2, consumer: T3)

Path: `<results_dir>/<world_stem>/_seed_filter.json` (world level; NOT inside a label dir,
so the plotter's `[0-9]*.json` episode glob — which runs only inside label dirs — never
sees it; the underscore prefix keeps it consistent with `_manifest.json`). Shape =
`SeedFilter` serialized (dataclasses -> dict, tuples -> lists). The per-seed field is
`rows` (matching the dataclass). `consulted_hashes`/`absent_files` keys are POSIX-relative
`"<label>/<seed>.json"` so they are stable across OSes.

### `runners/plot.py` seams (owner: T3)

- `load_world_results(results_dir, world_stem, *, replan_k=DEFAULT_REPLAN_K,
  expected=DEFAULT_EXPECTED_SEEDS, dropped_seeds: frozenset[int] = frozenset()) ->
  WorldResults` — new `dropped_seeds` keyword; excludes those seeds from the FILTERED
  counts. `per_seed` stays complete (dropped seeds keep their real outcome for B1).
  `WorldResults.degenerate_seeds` is set to `dropped_seeds ∩ seed_order`.
- `AlgoSummary` gains `n_present_raw: int` and `failure_rate_raw: float`.
  `n_present`/`failure_rate` become the FILTERED (headline) values; every existing chart
  reading them auto-adopts the drop. Success `times`/`path_lengths` are unaffected
  (degenerate seeds are crashes, never successes).
- `WorldResults` gains `degenerate_seeds: tuple[int, ...]`, `filter_status: str`
  (`"applied" | "stale" | "indeterminate" | "absent" | "off"`), and `n_dropped: int`.
  `main` decides `filter_status`/`n_dropped` from the sidecar checks and passes
  `dropped_seeds` into `load_world_results`; `load_world_results` populates the rest.
- `SUMMARY_CSV_COLUMNS` gains `n_present_raw`, `failure_rate_raw`, `n_dropped`
  (inserted after the existing `failure_rate`).
- CLI: `--filter/--no-filter` (default: apply a fresh, covering sidecar if present).
- The status footnote is rendered on **A1** (the headline scatter) — pinned there, not
  "every figure". A1's y-axis/annotation labels the filtered rate as conditional and the
  footnote states raw + filtered + `n_dropped` + the recorded planner set/criterion.

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
- The per-seed sidecar field is `rows` (NOT `seeds`).
- Experimental labels: `"d_star_lite_oracle_h<H>"`, `"d_star_lite_predictive_h<H>"`
  (built by `build_required_labels`, parity-checked against
  `algorithm_label("d_star_lite_oracle", None, H)` etc. in AC15's subprocess).

## Data Model

**Required-label construction (T1, pure stdlib — CR7/AC15).** No `planners` import on the
runtime path:

```python
_CANONICAL_ORDER = (   # frozen mirror of run_all._CANONICAL_ORDER
    "a_star_once", "a_star_replan", "dijkstra_once", "dijkstra_replan", "d_star_lite",
    "dwa", "apf", "rrt_once", "rrt_replan", "rrt_star_once", "rrt_star_replan",
)
_REPLAN_FAMILIES = frozenset({"a_star_replan", "dijkstra_replan", "rrt_replan", "rrt_star_replan"})

def build_required_labels(predict_horizon: int, replan_k: int, planner_set: str) -> list[str]:
    labels = [
        f"{name}_k{replan_k}" if name in _REPLAN_FAMILIES else name
        for name in _CANONICAL_ORDER
    ]
    if planner_set == "all13":
        labels += [f"d_star_lite_oracle_h{predict_horizon}",
                   f"d_star_lite_predictive_h{predict_horizon}"]
    return labels
```
AC15's subprocess test imports `planners`/`run_all` (allowed there) and asserts this equals
`[label for _n,_k,label in canonical_planner_set()]` + the two `algorithm_label(...)`
experimental labels, so any registry drift fails loud.

**Seed roster (T2 — CR6).** Read `derived_seeds` from the FIRST required label's
`_manifest.json` in `_CANONICAL_ORDER` order (matching `plot.load_world_results`). Then
enforce consensus: every OTHER found required-label manifest must agree on `derived_seeds`,
`world_stem`, `traffic`, and the speed provenance (`speed_regime`/`speed_min_factor`/
`speed_max_factor`); on disagreement, warn naming the odd label and set
`global_status="indeterminate"`. A `git_sha` mismatch is a warning only. If no manifest is
found, fall back to the sorted union of numeric stems present, set
`roster_is_canonical=False`, and note it in the printed verdict.

**Evidence collection per (seed, label) (T2):**
1. `present = (<seed>.json exists)`. If absent, first consult the label's manifest
   `episodes` roster: a `status == "runner_error"` entry for this seed is a DNF that
   provably ran past 4 s -> `PlannerEvidence(label, "survived", None)`. Otherwise ->
   `PlannerEvidence(label, "missing", None)` and record the path in `absent_files`. No hash
   is recorded for a missing/DNF file.
2. Read `<seed>.json`; record its `file_sha256` under `consulted_hashes["<label>/<seed>.json"]`.
3. `crashed = bool(rec["crashed"])`. If not crashed -> `"survived"`.
4. If crashed: read `<seed>.trace.jsonl`, `crash_step = crash_step_from_trace(...)`.
   `classify_planner(...)` -> `instant_crash` iff `1 <= crash_step <= window_steps`, else
   `survived` (late crash, or trace missing/unreadable — cannot confirm instant, so
   conservative).

## Error Handling

- **Sidecar absent (plot):** no drop; `filter_status="absent"`; footnote "filter: absent".
- **Sidecar present but stale/uncovering (plot):** no drop; loud stderr warning naming the
  cause (changed hash / newly-present absent file / roster change / label set not covered);
  `filter_status="stale"`; A1 footnote "filter: STALE, ignored — re-run filter_seeds".
  Never partial-apply.
- **Sidecar `global_status=="indeterminate"` (plot):** `dropped_seeds` is `[]` anyway;
  `filter_status="indeterminate"`; footnote "filter present but indeterminate — nothing
  dropped; missing: <labels>".
- **`--no-filter` (plot):** `filter_status="off"`; footnote "filter: off".
- **Unreadable METRICS JSON (filter_seeds):** warn to stderr, treat that planner as
  `missing` for that seed (blocks a drop — conservative), record it in `absent_files`,
  continue.
- **Unreadable/absent TRACE on a crashed record (filter_seeds):** the planner is
  `survived` (cannot confirm instant), per `classify_planner`; warn to stderr. (Distinct
  from an unreadable metrics file, which is `missing`.)
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
| TC-F1 | All required planners crash at step<=40 | Unit | seed verdict `degenerate`, in `dropped_seeds` |
| TC-F2 | One planner reaches goal on that seed | Unit | verdict `kept`, not dropped |
| TC-F3 | One planner crashes at step 41 (late) | Unit | that planner `survived`; verdict `kept` |
| TC-F4 | One required planner has no `<seed>.json` (not DNF) | Unit | that planner `missing`; in `absent_files`; verdict `indeterminate` |
| TC-F5 | Window boundary: crash step 40 vs 41 vs 0 sentinel | Unit | 40 instant, 41 not, step-0 sentinel never a crash |
| TC-F6 | `planner_error` (no trace) on a seed | Unit | that planner `survived` (blocks drop) |
| TC-F6b | DNF roster entry (manifest `status=="runner_error"`, no JSON) | Unit | that planner `survived`, verdict `kept` (W1) |
| TC-F7 | A whole required label dir absent | Integration | `global_status=="indeterminate"`, `missing_labels` set, distinct exit code, `dropped_seeds==[]`, seeds-with-a-survivor still `kept` (AC6) |
| TC-F8 | Sidecar round-trip write->read | Unit | `read_seed_filter(write(...))` equals the original; `read_seed_filter` returns `None` on a `schema_version` mismatch |
| TC-F9 | Determinism | Integration | two runs over one fixture tree -> byte-identical sidecar |
| TC-F10 | `--planners canonical` on an 11-only tree | Integration | required set is the 11; a seed can be `degenerate`; recorded set is the 11 |
| TC-F11 | Freshness + POSIX keys | Unit | mutate one metrics file -> `sidecar_is_fresh` returns `(False, [reason])`; every `consulted_hashes`/`absent_files` key contains `/` and no `\` (W7); a recorded-absent file that now exists -> not fresh; a roster change -> not fresh |
| TC-F12 | Headless guard (subprocess) | Integration | in a fresh subprocess, importing `filter_seeds` + calling `build_required_labels` leaves neither matplotlib nor irsim in `sys.modules` (AC1) |
| TC-F13 | `--window-seconds` / `--step-time` override | Unit | resolved `window_seconds`/`step_time`/`window_steps` recorded in the sidecar; a crash at a step inside the overridden window is `instant` |
| TC-F14 | Cross-manifest consensus (AC14) | Integration | two required manifests disagreeing on `derived_seeds`/`traffic`/speed -> `global_status="indeterminate"` + warning naming the odd label; `git_sha`-only mismatch -> warning, still `ok` |
| TC-F15 | Label parity (subprocess, AC15) | Integration | `build_required_labels(10,5,'all13')` equals `canonical_planner_set()` labels + the two `algorithm_label(...)` experimental labels |

`runners/plot.py --selfcheck` additions (TC-P12+):

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC-P12 | Fresh sidecar with 1 degenerate seed | Integration | that seed excluded from every algo's `failure_rate`/`n_present` AND from the A3 count layer (assert via reconstructed summaries + `WorldResults.degenerate_seeds`, mirroring TC-P6 — not by inspecting the PNG); present on B1's degenerate layer; `n_dropped==1`; `n_present_raw`/`failure_rate_raw` hold the full-50 values |
| TC-P13 | Stale sidecar | Integration | (a) a mutated metrics file, (b) a recorded-absent file now present, (c) the label set not covered by `required_labels` — each yields no drop, `filter_status=="stale"`, the A1 STALE footnote, and a stderr reason; charts still render |
| TC-P14 | Absent sidecar vs `--no-filter` | Integration | absent -> `filter_status=="absent"`; `--no-filter` -> `"off"`; both full-50 denominator; the two are byte-identical to each other and every numeric == raw with `n_dropped==0` (AC13) |
| TC-P15 | `summary.csv` raw+filtered columns | Unit | `n_present_raw`, `failure_rate_raw`, `n_dropped` present and correct for a filtered and an unfiltered run |
| TC-P16 | Indeterminate-status sidecar | Integration | no drop; footnote confesses + names missing; no raise |

**Test data:** synthetic result trees built in a `TemporaryDirectory` — `<seed>.json`
metrics (reuse plot's `_make_record` shape), tiny `<seed>.trace.jsonl` files with a
`crashed` flag at a chosen step, and a `_manifest.json` carrying `derived_seeds` + an
`episodes` roster (for DNF cases). Factory helpers live in each module's selfcheck block.

**Run command:** `python -m runners.filter_seeds --selfcheck` and
`python -m runners.plot --selfcheck` (both exit 0 iff all pass), plus
`python -m runners.plot_speed_sweep --selfcheck` and
`python -m runners.plot_horizon_sweep --selfcheck` as the downstream regression gate (AC12).

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|-----------|------|-------|-------------|
| T1 | Pure classification core | — | med | `runners/_seed_filter.py` | Implement the constants, dataclasses, and pure functions in Contracts & Interfaces (`build_required_labels`, `window_steps`, `crash_step_from_trace`, `classify_planner`, `classify_seed`, `write_seed_filter`, `read_seed_filter`, `sidecar_is_fresh`, `sidecar_covers`, `file_sha256`). Stdlib + json + hashlib ONLY — no `planners`/irsim/matplotlib import anywhere. Deterministic, sort-keyed sidecar, no timestamp. Satisfies AC1 (its half), AC3–AC5, AC8's primitives. |
| T2 | CLI labeler + `--selfcheck` | T1 | med | `runners/filter_seeds.py` | argparse CLI (`--world`, `--results-dir` default "results", `--replan-k` default 5, `--predict-horizon` default 10, `--window-seconds` default 4.0, `--step-time` default 0.1, `--planners {all13,canonical}` default all13, `--selfcheck`). Build labels via T1's `build_required_labels` (NO planners import on the runtime path), read the seed roster with the CR6 consensus check, collect per-(seed,label) evidence (metrics first, trace only on crash, DNF roster -> survived), classify, assemble the `SeedFilter`, write the sidecar, print a verdict line, and use the exit-code map (0 ok / distinct code when all-indeterminate / 1 no-tree / 2 CLI). Ship TC-F1..TC-F15 in-module. Stays headless. Satisfies AC1, AC2, AC6, AC11, AC14, AC15, and the TC-F table. |
| T3 | Plotter integration + `--selfcheck` | T1 | high | `runners/plot.py` | Thread `dropped_seeds` through `load_world_results`; add `AlgoSummary.n_present_raw`/`failure_rate_raw` (make `n_present`/`failure_rate` the filtered headline), `WorldResults.degenerate_seeds`/`filter_status`/`n_dropped`; in `main` read + freshness/coverage-verify the sidecar (import `read_seed_filter`/`sidecar_is_fresh`/`sidecar_covers`/`SEED_FILTER_NAME` from `runners._seed_filter` — a headless import), computing `plotted_labels` from the CLI `--replan-k`/`--predict-horizon`; `--filter/--no-filter`; the A1 status footnote + conditional-rate label + raw number; B1 "degenerate (filtered)" column marking + legend entry; `SUMMARY_CSV_COLUMNS` raw/filtered/n_dropped. Update the `run_selfcheck` docstring/banner from "TC-P1..TC-P11"/"N/11" to the new count, and add TC-P12..TC-P16. Preserve AC13 and the matplotlib-import guard. Satisfies AC7–AC10, AC12, AC13, and the TC-P table. |
| T4 | Docs | T1, T2, T3 | low | `CLAUDE.md`, `README.md` | New CLAUDE.md section "The degenerate-seed filter (issue #9)" (criterion, sidecar schema, freshness+coverage contract, the all-13 no-op trap + `--planners`, exit codes, and that the sidecar lives under gitignored `results/` so reproducibility rests on determinism + re-running filter_seeds). README: add `runners/filter_seeds.py` to the layout + a usage blurb including the EXACT two `run_experiment` commands that produce the `d_star_lite_oracle_h10` / `d_star_lite_predictive_h10` dirs the default `--planners all13` needs, and the plotter note on how filtered seeds are handled in `failure_rate` (the issue's "done" item). Match repo voice; no AI attribution. |

T2 and T3 both depend only on T1 and touch different files, so they run in parallel after
T1. T4 waits on all three (it documents final behavior).

## Notes for Implementer

- **Headless import discipline (load-bearing, CR7).** Importing `planners` transitively
  pulls irsim AND matplotlib (`planners -> manual_astar -> irsim -> matplotlib.pyplot /
  TkAgg`), and `run_all` imports `planners` at module top — so even a lazy
  `from runners.run_all import canonical_planner_set` pulls irsim at call time. Therefore
  `filter_seeds` builds its labels with the pure-stdlib `build_required_labels` and imports
  NOTHING from `planners` on its runtime path. `run_all`/`algorithm_label` are imported
  ONLY inside the AC15 subprocess parity test. `plot.py` importing from
  `runners._seed_filter` is safe (that module is planners-free). Verify with the AC1
  subprocess check.
- **Freshness scope — metrics only (CF1, judge upheld Fable).** Hash only the consulted
  `<seed>.json` **metrics** files, NOT traces. `wallclock_per_step` is a `perf_counter`
  mean, so any re-run of any episode changes its metrics bytes and trips the hash; a trace
  is a deterministic function of the same episode and cannot change without its metrics
  changing. The freshness contract adds (not trace hashing) the roster check, the
  absent-file-still-absent check, and — on the plot side — the `sidecar_covers` label-set
  check, which together close the "wrong labels / files appear later / roster changed"
  holes (CR1).
- **The step-0 sentinel.** Trace step 0 is the post-reset sentinel with `crashed=False`
  hardcoded, so `crash_step_from_trace` naturally returns the first real crash at step>=1;
  `classify_planner`'s `1 <= crash_step` lower bound is belt-and-suspenders.
- **Step-time assumption (CW8).** `step_time` defaults to 0.1 s (irsim's default, matching
  `PREDICT_DT` and every shipped world). The manifest does NOT record step_time, so a world
  using a non-default tick must pass `--step-time`; the resolved value is recorded in the
  sidecar. All arena_v1/v2 worlds use 0.1, so the default is correct for the study.
- **Exit-code map (filter_seeds), documented in the module docstring:** `0` = sidecar
  written and the roster was determinable (>=1 seed `degenerate` or `kept`); `3` = sidecar
  written but every seed `indeterminate` (a required label dir is missing, or a consensus
  mismatch — the loud no-op); `1` = no result tree / no seed roster (nothing written);
  `2` = CLI/validation. `3` is distinct so a script can tell "filter ran but decided
  nothing" from "filter ran and kept everything".
- **Order-preserving drop (why this is safe).** A degenerate seed is a crash for every
  planner, so removing D such seeds maps each rate `F/50 -> (F-D)/(50-D)` monotonically —
  planner rankings cannot change, only the contaminated absolute level. The filtered rate
  is CONDITIONAL (on non-degenerate); A1 labels it so and keeps the raw full-50 number in
  the footnote + `summary.csv`, so the transformation is auditable.
- **B3 exemption (AC7).** `_load_wallclocks` returns bare timing values with no seed
  mapping, from the separate `__wallclock__` subtree (a different short seed subset outside
  the freshness hash set). A crashed episode's per-step wallclock is still a valid timing
  sample, so B3 is deliberately NOT filtered — do not add a seed-aware wallclock loader for
  this issue.
- **Existing data reality (verification target).** `results/arena_v1/` currently holds the
  11 canonical dirs + `d_star_lite_predictive_h10` but NOT `d_star_lite_oracle_h10`, so a
  default `filter_seeds --world arena/arena_v1.yaml` will legitimately return
  `indeterminate` (exit 3, missing `d_star_lite_oracle_h10`). This is the correct loud
  no-op and is a good manual smoke: run it, confirm the verdict + exit code, then run
  `plot` and confirm the "indeterminate" footnote. To actually drop seeds, run the oracle
  at h10 first (`python -m runners.run_experiment --algorithm d_star_lite_oracle
  --predict-horizon 10 --world arena/arena_v1.yaml`), or use `--planners canonical` to
  filter on the 11 deliberately.
- **Do not** reintroduce the `d_star_lite_2` stray dir or any `_tmp_*` scratch into the
  required set — required labels come only from `build_required_labels`.
- **Future hooks (out of scope, do not build):** if a `crash_step` field is ever added to
  the metrics JSON, `classify_planner` can read it instead of the trace with no criterion
  change; the spawn-overlap geometry test (`Arena.initial_dynamic_snapshot`) could become
  a `--selfcheck`-adjacent cross-check TC someday.
- **Rollback:** the change is purely additive when the filter is off/absent (AC13) — two
  new files plus additive plot.py fields/flags. Reverting = delete the two new files and
  the plot.py additions; no result data or determinism guarantee is touched.
