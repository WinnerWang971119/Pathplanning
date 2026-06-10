# Phase 5 — Scatter Plot & Analysis Charts Plan

**Goal:** Produce the Mission's headline time-to-goal vs failure-rate scatter plus six analysis
charts from the 11-planner × 50-seed experiment, driven by a new read-only `runners/plot.py`
and fed by a new batch driver that runs every planner against the canonical seeds.

**Approach:** Two new files, no edits to the existing runners. `runners/run_all.py` iterates the
canonical planner set and shells out to the existing `run_experiment` once per planner — a
parallel bulk pass for the correctness metrics plus a short serial `--jobs 1` pass for a clean
`wallclock_per_step` (B3). `runners/plot.py` is pure read-and-render: a shared loader walks
`results/<world_stem>/<label>/[0-9]*.json`, classifies every episode, and one function per chart
writes a PNG into `results/<world_stem>/plots/`. matplotlib is declared in `requirements.txt`. The
structure is deliberately singular (read-only plotter + separate compute driver) so the existing
determinism/CLI conventions hold; coupling compute into the plotter or moving to a notebook was
rejected for that reason.

**Note on the plotter's location.** Mission.md line 142 and README.md line 310 name the
deliverable `results/plot.py`. This plan relocates it to `runners/plot.py` because `results/*` is
gitignored (`!results/.gitkeep` is the only exception), so a `results/plot.py` source file could
not be committed; `runners/` is already where the harness code lives (`run_episode.py`,
`run_experiment.py`, `_layout.py`) and keeps `results/` as data-only output. T8 updates both doc
references to `runners/plot.py`.

## Scope

- **In scope:**
  - `requirements.txt`: declare the existing-but-undeclared `matplotlib` dependency (it is
    already installed in the venv but not pinned in requirements; no seaborn).
  - `runners/plot.py` — read-only. Shared metrics loader + outcome classifier + per-algorithm
    summary; seven chart functions (A1, A3, A4, B1, B2, B3, B4); a `--selfcheck` mode mirroring
    `arena.py --check`; a `summary.csv` alongside the PNGs. CLI over a single world. Invoked
    `python -m runners.plot ...` (module style, like the other runners).
  - `runners/run_all.py` — batch driver: runs the 11 canonical planner labels via
    `run_experiment` (replan families at K=5), parallel bulk pass + a serial wallclock mini-pass
    landing in a results-root `__wallclock__` subtree that B3 reads.
  - Actually run the full batch and generate the seven charts from real `arena_v1` data (the
    user chose "I run it"); the PNGs + `summary.csv` live locally under `results/`.
  - Docs: README status table (Phase 4 + 5) and its `results/plot.py` reference → `runners/plot.py`;
    CLAUDE.md new "Phase 5" section; Mission.md Phase 5 status and its `results/plot.py` reference;
    this plan doc.
- **Out of scope:**
  - Editing `runners/run_episode.py` / `runners/run_experiment.py` internals — the new modules
    only shell out to / import from them.
  - Committing PNGs/CSV anywhere: `results/` stays gitignored; the deliverable images are viewed
    locally. No `docs/figures/`, no `.gitignore` exception for generated artifacts.
  - **A2 Pareto-frontier chart** (deselected by the user).
  - Phase 6b K-sweep — replan families are plotted at the single cadence K=5 only.
  - Multi-world plotting / new world fixtures — `plot.py` is parametrized by `--world` but only
    `arena_v1.yaml` is run and rendered this phase.
  - Phase 7 written analysis / statistical tests — the charts feed it; the insight write-up is a
    later phase.
  - Any non-matplotlib visualization dependency.

## Decisions

- **Charts: A1, A3, A4, B1, B2, B3, B4** — exactly the user's selection (A2 excluded). A1 is the
  Mission-required deliverable.
- **Plotter at `runners/plot.py`, not `results/plot.py`** — `results/*` is gitignored, so the
  Mission-named path is uncommittable; `runners/` is the repo's home for harness code (judge
  ruling on the gitignore contradiction). Docs are updated to the new path.
- **K=5** for the four `_replan` families (`a_star_replan_k5`, `dijkstra_replan_k5`,
  `rrt_replan_k5`, `rrt_star_replan_k5`) — matches every existing label/example until the 6b
  sweep picks a real best-K.
- **Failures are a count.** Per-seed dots on any time axis use successful seeds only; failures
  (crash + timeout + planner_error) are captured in the Y-axis failure rate and in the
  count-based charts (A3/B1). This matches Mission Phase 4's "time-to-goal distribution
  (successes only)".
- **Failure set extends Mission's definition.** Mission.md defines failure rate as
  `(crashes + timeouts) / 50`; this plan folds `planner_error` into failures too
  (`(crash + timeout + planner_error) / n_present`). A `reset()` failure is a real failure; on
  `arena_v1` with reachable goals `planner_error` should be ~0, so the divergence is negligible
  in practice but is called out here as a deliberate decision, not silent drift.
- **Phase 4 lands inside the plotter.** Mission Phase 4 ("per-algorithm aggregation") was never
  given its own artifact; the loader's per-algorithm summary + `summary.csv` IS that aggregation.
  T8 records this so Phase 4 → done is explicit, not implied.
- **Outcome precedence:** `planner_error` → `crash` → `timeout` → `success`. Exactly one class
  per episode. `success` ⇔ `time_to_goal` is non-null.
- **Output to `results/<world_stem>/plots/`** (gitignored, not committed); `summary.csv` beside
  the PNGs.
- **B3 wallclock source:** a serial `--jobs 1` mini-pass (default 5 seeds/planner) written to
  `results/__wallclock__/<world_stem>/<label>/` (a results-root sibling of `<world_stem>`, so the
  stem appears exactly once where `episode_out_dir` places it and the main loader never sees it);
  B3 reads that subtree, falling back to the bulk dir's perturbed wallclock with a caveat if the
  subtree is absent. This two-pass design is the user's explicit informed choice (they rejected
  both all-serial and noisy-parallel).
- **matplotlib only** — box/violin/heatmap/bars are all native to matplotlib; no seaborn.
- **`plot.py` is read-only**; `run_all.py` owns all compute.
- **Branch** off the current `phase6-reactive-sampling-planners` HEAD (already merged into
  `origin/main`); the local working tree is clean.

## Acceptance Criteria

- [ ] **AC1 — Dependency.** `matplotlib` is in `requirements.txt`; `plot.py` routes its import
  through `ensure_matplotlib()`, which exits with a clear "pip install -r requirements.txt"
  message (not a raw `ImportError`) when the module is absent. (Tested via TC-P8 with a patched
  `find_spec`.)
- [ ] **AC2 — Loader + classifier.** The loader enumerates the explicit canonical label list under
  `results/<world_stem>/<label>/`, globs only numeric-stem JSONs (`[0-9]*.json`), skips
  `_manifest.json` and any non-numeric file, parses all 7 metric fields, and classifies each
  episode into exactly one of {success, crash, timeout, planner_error} using the precedence
  above. A `planner_error`-tagged episode is classified `planner_error` even if other flags are
  false.
- [ ] **AC3 — Summary math.** Per algorithm: `failure_rate = (n_crash + n_timeout +
  n_planner_error) / n_present`; the success set is the episodes with non-null `time_to_goal`;
  median and mean of successful times are computed (NaN/empty handled when `n_success == 0`).
- [ ] **AC4 — A1 headline scatter.** Renders per-seed success dots (one color per algorithm),
  a **median** centroid and a **mean** centroid per algorithm at (centroid-time, failure_rate),
  a side legend, labeled axes, and a "down-left wins" annotation; saved as a PNG. An algorithm
  with 0 successes still appears (its failure_rate row) without raising.
- [ ] **AC5 — A3 failure-breakdown bars.** Per-algorithm stacked bars of success / crash /
  timeout / planner_error counts (summing to `n_present`), fixed per-outcome colors, legend.
- [ ] **AC6 — A4 time-to-goal box/violin.** Per-algorithm distribution of successful times,
  sorted by median; an algorithm with <2 successes degrades to plotted points (or is annotated),
  never raises.
- [ ] **AC7 — B1 seed-difficulty heatmap.** 11 algorithm rows × 50 seed columns, all rows aligned
  to one shared seed-column order; successes shaded by a continuous time colormap with a
  colorbar; failure cells drawn in distinct flat colors (crash / timeout / planner_error) with a
  legend.
- [ ] **AC8 — B2 path-length box.** Per-algorithm box of `path_length` over successes with a
  horizontal reference line at the straight-line Euclidean lower bound `46·√2 ≈ 65.05 m`, labeled
  as unreachable through the walls.
- [ ] **AC9 — B3 compute-cost bars.** Per-algorithm mean `wallclock_per_step` sourced from
  `results/__wallclock__/<world_stem>/<label>/`, sorted, with a footnote stating the metric's
  `--jobs` sensitivity and which subtree it came from (or that it fell back to the bulk dir).
- [ ] **AC10 — B4 family-contrast panels.** Small-multiple subplots isolating the designed
  experiments: A\* vs Dijkstra, once vs replan, reactive vs global — grouped bars on failure rate
  and median time.
- [ ] **AC11 — Robust to partial data.** A missing label dir, a label with <50 episode JSONs, a
  malformed JSON file, and a 0-success algorithm all produce a stderr warning and a rendered
  chart annotated with the actual N — never a traceback. A 0-success algorithm still appears in
  A1's failure_rate row and in A3/B1 with its N-annotation rendered. A world with no readable
  data at all exits non-zero with a clear message.
- [ ] **AC12 — Driver.** `python -m runners.run_all --world arena/arena_v1.yaml` runs all 11
  canonical labels via `run_experiment` subprocesses (replan families forwarded `--replan-k 5`),
  traffic on, master seed + num-seeds forwarded; a `--jobs N` bulk pass plus a serial
  `--wallclock-seeds M` pass into `results/__wallclock__/<world_stem>/<label>/`. `--resume` is
  scoped per pass (bulk resume probes the bulk dir; wallclock resume probes the wallclock dir),
  so each pass skips only its own completed seeds.
- [ ] **AC13 — Selfcheck.** `python -m runners.plot --selfcheck` passes TC-P1…TC-P10 on synthetic
  fixtures in a `TemporaryDirectory`, rendering every chart to a non-empty PNG, and exits 0.
- [ ] **AC14 — Output location & tracking.** All generated PNGs + `summary.csv` land under
  `results/<world_stem>/plots/`; `git status` shows nothing new tracked under `results/` (still
  gitignored). The plotter itself ships as `runners/plot.py` and IS tracked.
- [ ] **AC15 — Real artifacts.** The full `arena_v1` batch is actually run and the seven charts +
  `summary.csv` are generated from real data; the images are surfaced to the user.
- [ ] **AC16 — Docs.** README status table (Phase 4 → done via aggregation, Phase 5 → done) and
  its `results/plot.py` reference → `runners/plot.py`; CLAUDE.md gains a "Phase 5" section
  documenting `plot.py` + `run_all.py` + the chart set + the `__wallclock__` subtree +
  `--selfcheck`; Mission.md Phase 5 marked landed with its `results/plot.py` reference updated.

## Data Model

```python
# Outcome of one episode JSON.
OUTCOMES = ("success", "crash", "timeout", "planner_error")

def classify_outcome(rec: dict) -> str:
    # Precedence is load-bearing: a reset() failure writes planner_error with the
    # other flags false and time_to_goal null, so it must be checked first.
    if rec.get("planner_error") is not None:
        return "planner_error"
    if rec.get("crashed"):
        return "crash"
    if rec.get("timed_out"):
        return "timeout"
    if rec.get("time_to_goal") is not None:
        return "success"
    # Defensive only: run_episode ALWAYS writes one of the flags or a non-null
    # time_to_goal, so this branch is unreachable from real output. TC-P1 still
    # asserts it as a guard against malformed/hand-authored records.
    return "planner_error"

@dataclass(frozen=True)
class AlgoSummary:
    label: str                 # results dir label, e.g. "a_star_replan_k5"
    display: str               # legend name, e.g. "A* replan (K=5)"
    family: str                # "grid" | "incremental" | "reactive" | "sampling"
    n_present: int             # episode JSONs found (warn if != expected 50)
    n_success: int
    n_crash: int
    n_timeout: int
    n_planner_error: int
    failure_rate: float        # (crash+timeout+planner_error)/n_present
    times: tuple[float, ...]        # successful time_to_goal values
    path_lengths: tuple[float, ...] # path_length over successes
    wallclocks: tuple[float, ...]   # wallclock_per_step (from __wallclock__ subtree for B3)
    per_seed: dict[int, str]   # seed -> outcome, for the B1 heatmap (aligned columns)

# Canonical planner set rendered this phase (label order = chart row/column order):
CANONICAL = [
    ("a_star_once",        None, "grid",        "A* once"),
    ("a_star_replan",      5,    "grid",        "A* replan (K=5)"),
    ("dijkstra_once",      None, "grid",        "Dijkstra once"),
    ("dijkstra_replan",    5,    "grid",        "Dijkstra replan (K=5)"),
    ("d_star_lite",        None, "incremental", "D* Lite"),
    ("dwa",                None, "reactive",    "DWA"),
    ("apf",                None, "reactive",    "APF"),
    ("rrt_once",           None, "sampling",    "RRT once"),
    ("rrt_replan",         5,    "sampling",    "RRT replan (K=5)"),
    ("rrt_star_once",      None, "sampling",    "RRT* once"),
    ("rrt_star_replan",    5,    "sampling",    "RRT* replan (K=5)"),
]
# Label is algorithm_label(name, k); seed-column order for B1 is the manifest's
# derived_seeds (the shared traffic stream), read once from any present _manifest.json,
# falling back to sorted numeric stems if no manifest is found.
STRAIGHT_LINE_IDEAL_M = 46.0 * (2.0 ** 0.5)   # (2,2)->(48,48) ≈ 65.05 m Euclidean;
                                              # UNREACHABLE through the corridor walls.
```

## Interfaces

```
# runners/plot.py
python -m runners.plot --world arena/arena_v1.yaml [--results-dir results]
    [--replan-k 5] [--charts a1,a3,a4,b1,b2,b3,b4] [--out-dir <dir>]
python -m runners.plot --selfcheck            # synthetic-fixture unit + smoke suite, exit 0/1

# runners/run_all.py
python -m runners.run_all --world arena/arena_v1.yaml
    [--master-seed 20260605] [--num-seeds 50] [--jobs N]
    [--wallclock-seeds 5] [--results-dir results] [--resume] [--traffic|--no-traffic]
# Bulk:      run_experiment per canonical label at --jobs N, --results-dir <results>
#            -> child episode_out_dir writes results/<stem>/<label>/.
# Wallclock: re-run the first --wallclock-seeds seeds per label at --jobs 1,
#            --results-dir <results>/__wallclock__
#            -> child episode_out_dir writes results/__wallclock__/<stem>/<label>/.
```

`episode_out_dir(results_dir, world_stem, label)` always appends `<world_stem>/<label>`, so the
wallclock pass passes `<results>/__wallclock__` as its results-dir and the child reinserts the
stem, yielding `results/__wallclock__/<world_stem>/<label>/` — the exact path B3 reads.

## Error Handling

- **Missing label dir / <50 JSONs:** warn to stderr, render with the present N, annotate the
  actual count on the chart. No raise.
- **Malformed / unreadable JSON:** skip that file with a stderr warning; continue.
- **0-success algorithm:** omitted from A1 dots / A4 box / B2 box (still present, with its
  N-annotation, in A1's failure_rate row and in A3, B1, B3). No raise.
- **No readable data for the world at all:** print a clear message and exit non-zero (nothing to
  plot).
- **`__wallclock__` subtree absent:** B3 falls back to the bulk dir's `wallclock_per_step` and
  prints a caveat (on the figure and to stderr) that the values are parallel-perturbed.
- **matplotlib import failure:** `ensure_matplotlib()` catches it at startup → friendly install
  hint, exit non-zero.
- **`run_all` child runner failure:** inherits `run_experiment`'s policy — continue past the
  failed planner, list it in the end summary, exit non-zero if any planner's batch failed.

## Testing Strategy

**Levels:** Unit (loader / classifier / summary math / source selection / driver derivation over
synthetic JSON trees) + smoke (each chart writes a non-empty PNG to a temp dir). Mirrors the
`arena.py --check` convention via a `plot.py --selfcheck` entry point; no pytest runner is
introduced (the repo has none). Negative paths use monkeypatch, the same technique the arena
`--check` suite already uses (e.g. TC2b patches `get_lidar_scan`). The selfcheck is NOT
import-light — TC-P5 renders all seven charts, so it loads matplotlib and needs a writable temp
dir; it is a correctness gate, not a millisecond smoke test.

| ID     | Test Case | Type | Expected Behavior |
|--------|-----------|------|-------------------|
| TC-P1  | `classify_outcome` precedence | Unit | success / crash / timeout / planner_error each classified correctly; a record with `planner_error` set AND `crashed=true` classifies `planner_error`; an all-false/null record classifies `planner_error` (defensive branch) |
| TC-P2  | Loader over a synthetic tree | Unit | Reads numeric-stem JSONs, skips `_manifest.json` and a `notes.txt`; `n_present` counts only episode files; a label dir with 3 of 50 files warns and reports `n_present=3` |
| TC-P3  | Summary math | Unit | On a known set (e.g. 3 success times {10,20,60}, 1 crash, 1 timeout) `failure_rate=2/5`, `median=20`, `mean=30`, `n_success=3` |
| TC-P4  | Partial / missing data | Unit | Missing label dir → warning + skipped from dot charts, present in count charts; no exception raised anywhere in the load path |
| TC-P5  | Chart smoke (all 7) | Smoke | Each of A1/A3/A4/B1/B2/B3/B4 writes a non-empty PNG to a temp dir from a synthetic 11-label fixture that includes a 0-success algorithm and a <50-seed algorithm |
| TC-P6  | B1 column alignment | Unit | The heatmap matrix has 11 rows × (seed-count) columns; every row maps to the same seed→column index from the manifest order; a failure cell carries its categorical color code |
| TC-P7  | Malformed JSON + no-data world | Unit | A malformed JSON is skipped with a warning; a world stem with zero readable files makes `plot.py` exit non-zero with the "nothing to plot" message |
| TC-P8  | matplotlib import guard | Unit | `ensure_matplotlib()` with `importlib.util.find_spec` patched to return `None` prints the "pip install -r requirements.txt" hint and exits non-zero (negative-path content, not just guard presence) |
| TC-P9  | B3 wallclock source selection | Unit | Build `results/__wallclock__/<stem>/<label>/` at the EXACT path `run_all` produces, holding distinct sentinel `wallclock_per_step` values, AND a bulk `results/<stem>/<label>/` with DIFFERENT sentinels. Assert B3's summary reports the `__wallclock__` sentinels, not the bulk ones. Then delete the subtree and assert B3 falls back to the bulk values AND emits the caveat. |
| TC-P10 | `run_all` canonical-set derivation | Unit | `canonical_planner_set()` returns the 11 expected `(algorithm, replan_k, label)` tuples; the 4 replan families carry `replan_k=5` and `_k5` labels, the other 7 carry `replan_k=None`; the constructed child command for a replan family includes `--replan-k 5` and a non-replan family omits it (built without launching any episode) |

**Test data:** synthetic result trees built in a `TemporaryDirectory` (write hand-made
`<seed>.json` dicts + a `_manifest.json` with `derived_seeds`), no irsim needed. TC-P10 imports
`runners.run_all.canonical_planner_set` + its command builder (pure functions, no subprocess).
The real end-to-end validation is AC15 (the actual `arena_v1` batch).
**Run command:** `python -m runners.plot --selfcheck`

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Loader + classifier + summary scaffolding | — | med | `requirements.txt`, `runners/plot.py` | Declare `matplotlib` in requirements. Create `runners/plot.py` with: `ensure_matplotlib()` import guard via `importlib.util.find_spec` (AC1); `classify_outcome` (AC2, precedence + defensive comment per Data Model); the `CANONICAL` label table + `algorithm_label`-derived dir labels; `load_world_results(results_dir, world_stem, expected=50)` globbing `[0-9]*.json`, skipping `_manifest.json`/non-numeric, parsing 7 fields, building `AlgoSummary` per label incl. `per_seed` + seed-column order from `_manifest.json` `derived_seeds` (fallback sorted stems); summary math (AC3); `summary.csv` writer; argparse CLI skeleton (`--world/--results-dir/--replan-k/--charts/--out-dir/--selfcheck`) resolving `<world_stem>` + out-dir `results/<stem>/plots/`. Wallclock loaded from `results/__wallclock__/<stem>/<label>/` when present, else bulk-dir fallback (AC9 source). Satisfies AC1–AC3, AC14 (placement/out-dir), part of AC9/AC11. Do NOT import irsim. |
| T2 | Core charts A1, A3, A4 | T1 | med | `runners/plot.py` | Implement `chart_a1_scatter` (per-seed success dots one color/algorithm + median & mean centroids at the algo failure_rate, side legend, labeled axes, down-left annotation, 0-success algos shown without raising — AC4), `chart_a3_failure_bars` (stacked success/crash/timeout/planner_error to `n_present`, per-outcome colors, legend — AC5), `chart_a4_time_box` (box or violin of successful times, sorted by median, <2-success degrades to points — AC6). Each writes a PNG into the out-dir and is dispatched by `--charts`. Satisfies AC4–AC6. |
| T3 | Analysis charts B1, B2, B3, B4 | T2 | med | `runners/plot.py` | Implement `chart_b1_heatmap` (11×seed matrix aligned to the shared seed order; successes via continuous colormap + colorbar; failure cells distinct flat colors + legend — AC7), `chart_b2_pathlen_box` (path_length boxes + `STRAIGHT_LINE_IDEAL_M` reference line labeled unreachable — AC8), `chart_b3_compute_bars` (mean `wallclock_per_step` from the `__wallclock__` subtree, sorted, `--jobs` caveat footnote + fallback note — AC9), `chart_b4_family_panels` (small-multiple subplots: A* vs Dijkstra, once vs replan, reactive vs global; grouped bars on failure_rate + median time — AC10). Same-file dependency on T2. Satisfies AC7–AC10. |
| T4 | Batch driver `run_all.py` | — | med | `runners/run_all.py` | New module. Pure `canonical_planner_set() -> list[(algorithm, replan_k, label)]` derived from `planners.ALGORITHMS` + `REPLAN_FAMILIES` (K=5 for replan families) + a pure child-command builder (both unit-tested by TC-P10). Bulk pass: launch `python -m runners.run_experiment` **subprocess** per label at `--jobs N`, `--results-dir <results>` (NOT in-process `main()` — match the existing run_experiment→run_episode subprocess pattern and avoid accumulated `sys.path`/import side effects), forwarding `--master-seed/--num-seeds/--traffic/--resume`. Wallclock pass: re-run the first `--wallclock-seeds M` seeds per label at `--jobs 1`, `--results-dir <results>/__wallclock__` (child reinserts the stem → `results/__wallclock__/<stem>/<label>/`). `--resume` scoped per pass. Print a per-label tally; exit non-zero if any label's batch runner-failed. Satisfies AC12. |
| T5 | `--selfcheck` test suite | T3, T4 | med | `runners/plot.py` | Implement `--selfcheck`: build synthetic result trees (incl. a 0-success algo, a <50-seed algo, a malformed file, and a `results/__wallclock__/<stem>/<label>/` subtree with sentinel wallclocks) in a `TemporaryDirectory`, run TC-P1…TC-P10 (Testing table) as plain asserts — TC-P10 imports `runners.run_all.canonical_planner_set` + the command builder — render every chart to the temp dir asserting non-empty PNGs, print PASS/FAIL per TC, exit 0/1. Same-file dependency on T3; imports T4. Satisfies AC11 (assertions), AC13. |
| T6 | Smoke-gate then run the full batch | T4 | med | (no repo files; generates `results/`) | Activate `.venv`. FIRST run a fast smoke gate: `python -m runners.run_all --world arena/arena_v1.yaml --num-seeds 1 --wallclock-seeds 1 --jobs 2` and verify BOTH `results/arena_v1/<label>/1*.json` and `results/__wallclock__/arena_v1/<label>/` populated at the expected paths (catches any wallclock-path regression before the multi-hour run). THEN the full run: `python -m runners.run_all --world arena/arena_v1.yaml --jobs <cores-2> --wallclock-seeds 5`. 11 planners × 50 traffic episodes is multi-hour wall-clock (the existing 45-TC check alone is ~50 min and is a fraction of this); budget hours, run unattended. Confirm all 11 label dirs + the `__wallclock__` subtree populated; check the exit code + end summary. The heavy compute the user opted to run. |
| T7 | Generate + eyeball the 7 charts | T5, T6 | med | (no repo files; generates `results/<stem>/plots/`) | `python -m runners.plot --world arena/arena_v1.yaml`. Verify all 7 PNGs + `summary.csv` under `results/arena_v1/plots/`, open them, sanity-check axes/legend/annotations/centroids, confirm B3 read the `__wallclock__` subtree (no fallback caveat), and surface the images to the user (AC15). Re-run `--selfcheck` to confirm green. |
| T8 | Docs | T7 | low | `README.md`, `CLAUDE.md`, `Mission.md` | README status table: Phase 4 → done (per-algorithm aggregation lands in `plot.py`), Phase 5 → done with `runners/plot.py` + `runners/run_all.py`; update the README `results/plot.py` reference to `runners/plot.py`. CLAUDE.md: add a "Phase 5" section (the seven charts, the read-only `plot.py`, the `run_all.py` driver, the `results/__wallclock__` subtree, `--selfcheck`, output under `results/<stem>/plots/`). Mission.md Phase 5: mark landed (A2 deferred, K=5, 6b pending) and update its `results/plot.py` reference. Follow the user's git/prose rules (no AI attribution, no tell-words, no em-dashes) for any committed text. Satisfies AC16. |

**Parallel waves:** **W1** = T1, T4. **W2** = T2 (after T1). **W3** = T3 (after T2). **W4** =
T5 (after T3, T4) and T6 (after T4) in parallel. **W5** = T7 (after T5, T6). **W6** = T8.

## Notes for Implementer

- **Single-file plotter, serial chart tasks.** T2/T3/T5 all edit `runners/plot.py`, so they are a
  strict chain (T1→T2→T3→T5); only T4 (`run_all.py`) runs alongside. Do not split the chart
  functions into separate modules — the file is small enough and the loader is shared.
- **Plotter is `runners/plot.py`, not `results/plot.py`.** `results/*` is gitignored; the harness
  code convention is `runners/`. Invoke as `python -m runners.plot`. Generated PNGs/CSV still land
  under `results/<stem>/plots/` (gitignored).
- **Failures never enter a time axis.** A1 dots, A4 boxes, and B2 boxes draw successes only;
  failures live in the failure_rate (A1 Y), the stacked bars (A3), and the heatmap cells (B1).
  This is the user's "failures are a count" decision and Mission Phase 4's successes-only rule.
- **Failure rate includes planner_error** (extends Mission's crash+timeout); ~0 on arena_v1 but
  documented as a deliberate decision.
- **B1 column order is the shared traffic stream.** Read `derived_seeds` from any present
  `_manifest.json` so all 11 rows align to the same seed columns — that alignment is the whole
  point of the heatmap (it exposes universally-hard streams). Fall back to sorted numeric stems
  only if no manifest exists.
- **B3 reads `results/__wallclock__/<stem>/<label>/`, not the bulk dir.** The bulk pass runs
  parallel, so its `wallclock_per_step` is contention-perturbed; the serial mini-pass is the
  trustworthy source. The `__wallclock__` tree is a results-root sibling of `<world_stem>` (the
  stem appears once, inserted by `episode_out_dir`), so the main loader globbing under
  `results/<world_stem>/` never sees it. If the subtree is missing, fall back with an on-figure
  caveat rather than failing.
- **`run_all` wallclock pass uses `--results-dir <results>/__wallclock__`** so the child's
  `episode_out_dir` reinserts the stem. Do NOT pass `<results>/<stem>/__wallclock__` — that
  double-nests the stem and B3 would never find the files (the path-nesting bug the smoke gate in
  T6 guards against).
- **0-success is expected for several planners.** `_once` planners and the reactive pair will
  fail most traffic seeds (the experimental signal). Charts must render them without dropping the
  algorithm entirely — that absence is itself a finding, so keep them in A1's failure-rate row and
  in A3/B1 with the N-annotation.
- **Output stays gitignored.** Do not add a `.gitignore` exception for generated artifacts and do
  not copy PNGs into a tracked dir; the user chose local-only artifacts. T8 documents where they
  land, nothing more.
- **Determinism caveat carries to the figure, not just the docs.** B3's footnote must state the
  `wallclock_per_step` `--jobs` sensitivity so a reader of the PNG alone is not misled.
- **Rollback:** every task is an additive new file except T1's one-line `requirements.txt` edit
  and T8's doc edits; deleting `runners/plot.py` + `runners/run_all.py` and reverting those two
  edits fully restores the prior state. Generated artifacts under `results/` are gitignored, so
  there is nothing to revert there.
