# Predictive DWA v2 — Braking-Inevitability + Cost-to-Go: quick-read findings

**What ran:** a 10-seed × {0, 5, 10, 20}-horizon quick read for plain `dwa` and the
four rebuilt predictive-DWA keys, traffic ON, `arena_v1`, canonical seed stream
(master 20260605, first 10 seeds). Every one of the 170 episodes ran to a terminal
state — 0 runner failures. This is the T8 deliverable of
`2026-07-10-predictive-dwa-braking.md` (AC12); the full 50-seed / horizon / speed
sweeps remain the user's to launch.

Chart: `2026-07-10-predictive-dwa-braking.findings.png` (failure rate vs horizon).

## Measured baseline

`plain dwa`: **failure_rate 0.50** (5/10 reach the goal), median success time 85.9 s.
This is the number both AC12 gates are read against.

## Failure rate by horizon (10 seeds)

| key | h0 | h5 | h10 | h20 |
|---|---|---|---|---|
| `dwa_predictive` (lidar + global, **canonical**) | 1.00 | 0.90 | 0.90 | 0.80 |
| `dwa_predictive_oracle` (oracle + global) | 1.00 | 0.80 | 1.00 | 0.80 |
| `dwa_predictive_paper` (lidar, braking-only) | 0.50 | 0.50 | 0.70 | 0.60 |
| `dwa_predictive_paper_oracle` (oracle, braking-only) | 0.50 | 0.40 | 0.70 | 0.70 |

Median success time (s), `nan` = no success that cell:

| key | h0 | h5 | h10 | h20 |
|---|---|---|---|---|
| `dwa_predictive` | nan | 73.0 | 73.0 | 78.5 |
| `dwa_predictive_oracle` | nan | 79.4 | nan | 99.0 |
| `dwa_predictive_paper` | 85.9 | 85.9 | 86.0 | 99.1 |
| `dwa_predictive_paper_oracle` | 85.9 | 85.9 | 85.9 | 109.8 |

## 2×2 ablation (best nonzero-horizon failure rate)

|  | oracle | lidar |
|---|---|---|
| **paper-only** (braking, no field) | **0.40** @h5 | 0.50 @h5 |
| **paper+global** (braking + field) | 0.80 @h5 | 0.80 @h20 |

## Gate verdicts (judged on the ORACLE rows) — both PASS

- **Gate 1 — PASS.** `dwa_predictive_paper_oracle` best-horizon failure rate
  **0.40 (@h5) < 0.50** plain `dwa`. The emergency-braking inevitable-collision
  policy, on its own, beats plain DWA on the same 10 traffic streams. The Missura
  soft-braking layer is the real improvement over the old blanket space-time reject
  (which measured net-harmful at 0.70–0.80 before this rebuild).
- **Gate 2 — PASS.** `dwa_predictive_oracle` best-horizon failure rate
  **0.80 (@h5) < 1.00** its own global-only h0 cell. Adding the braking layer to the
  field-guided controller improves it over field-only guidance.

AC12 is satisfied: both gates pass on the oracle rows.

## The load-bearing finding: the global cost-to-go guidance is net-harmful

The gates pass, but the honest headline is the opposite of the design hypothesis.
The cost-to-go field was added to cure local-minima timeouts; instead it **raises**
failure rate sharply:

- Every paper+global cell (0.80–1.00) is worse than plain `dwa` (0.50) and worse
  than its paper-only sibling (0.40–0.70).
- The global-only baseline (h0, field with no braking) fails **every** seed
  (1.00) for both the lidar and oracle variants — the field heading alone drives
  the robot deterministically along the geodesic straight into crossing traffic
  with no yielding, so gate 2's "improvement" is measured against a floor.
- The canonical key `dwa_predictive` (lidar + global) is therefore currently
  **worse than plain `dwa`** (best 0.80 vs 0.50). The braking-only ablation, which
  ships as the experimental `dwa_predictive_paper*`, is the variant that actually
  helps.

So the win is entirely the braking-inevitability policy; the static geodesic field,
as weighted here, over-commits the robot to the shortest-path heading and defeats
the yielding the soft term is trying to produce. Gate 2 passing is real but weak —
it clears a global-only baseline that is itself a regression.

Horizon shape: the braking benefit is sharpest at **h5** (0.40 oracle, and the field
variants' best) and decays or reverses by h10/h20 — a longer inevitability window
makes the ICS test reject more forward candidates, over-braking into stalls/timeouts
(paper-only median time climbs 85.9 s → 109.8 s at h20). Short horizons yield; long
horizons freeze.

## Lidar rows (reported, not gated — a perception finding)

Gates are judged on the oracle rows by design (oracle-first isolates policy from
perception). The lidar rows: `dwa_predictive_paper` best 0.50 (ties plain `dwa`),
`dwa_predictive` best 0.80 (harmed by the field, as above). The frame-differencing
estimator (even with the new smoothing + 2.0 m/s clamp) does not add value over
plain DWA on these 10 seeds in the braking-only config, and does not rescue the
field-harmed global config. This is a perception finding, not a spec blocker.

## Next-step levers (for the deferred full sweep + a v3)

1. **Down-weight or drop the global field for the canonical key.** The braking-only
   `paper` variant is the empirical winner; the strongest immediate lever is to make
   the field heading a much smaller nudge (or fall back to Euclidean) rather than the
   dominant heading term, or to reconsider the plan's "canonical = paper+global"
   choice in favor of paper-only.
2. **Prefer a short horizon (~h5, T=0.5 s).** The ICS test's benefit peaks there and
   inverts by h20; the canonical h10 is past the sweet spot.
3. **Make the field time-aware.** A static geodesic field cannot know a cell is only
   transiently blocked by a crosser; a decay/traffic-aware cost would stop it from
   steering into the very lanes the braking layer then has to freeze for.
4. **Estimator refinement (lidar).** Tighter clustering / a velocity floor before the
   lidar variant can beat plain DWA; separate from the policy question above.

None of these are in scope for this change — the code ships the four keys and the
gates pass; this document records that the braking policy is the genuine advance and
the global guidance needs rework before the canonical key is worth its main-scatter
slot.
