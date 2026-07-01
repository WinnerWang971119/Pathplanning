# Brainstorm: moving-obstacle-aware planning

**Seed (verbatim):** Think ways to improve the D* and other to prevent crashing to obstacles. I think they should encounter the fact that the object is "moving"

**Context summary:**
- The arena's global planners (D* Lite, a_star/dijkstra/rrt _replan) fold the
  *current* lidar frame onto a COPY of the static grid each act — memorylessly.
  Moving obstacles are therefore baked in as STATIC walls at their instantaneous
  position. The planner has no notion of velocity.
- Project memory already records the empirical consequence: replanning does NOT
  lower per-second crash risk vs single-plan ([[project-replanning-no-safer-than-single-plan]]);
  only DWA's velocity-space avoidance halves hazard (0.48 vs ~0.9/100s). This is
  exactly the user's intuition — the global planners ignore that obstacles move.
- Hard constraint from Mission.md: after t=0 the planner sees ONLY lidar
  (static map + initial obstacle positions at t=0). Obstacle vx/vy live in the
  spawner (`DynamicObstacleState.vx/vy`) but are NOT exposed to the planner — so
  velocity must be ESTIMATED from successive lidar frames, or a deliberate oracle
  channel must be added for research.
- `motion_rng` is plumbed but unused; lidar bearings reconstructed via np.linspace
  ([[gotcha-lidar-angle-increment-mismatch]]); lidar is center-to-surface
  ([[gotcha-lidar-center-relative]]). Determinism (byte-identical traces) is a
  load-bearing harness invariant any change must preserve.

**Clarified framing:** Build a NEW, measurable D* Lite variant (lidar-only, no
oracle) that estimates crossing-obstacle velocity from successive lidar frames
and uses it to avoid where obstacles WILL be, so it stops crashing into traffic.
It must (a) fit the existing `Controller` interface and register as a new
algorithm key, (b) preserve byte-identical determinism, (c) ideally move
down-left (lower crash rate, comparable time) vs plain `d_star_lite` on the
time-to-goal-vs-crash scatter. D* Lite is the flagship to prove it on; generalize
to A*/RRT replan later. Velocity must come from the lidar stream only — the
spawner's true vx/vy stays hidden (Mission-faithful). Decision forks resolved:
measurable variant + lidar-estimate-only + D*-Lite-first.

## Perspectives (round 1)
### Critic (strongest first)
1. **Kinematics cap the achievable win.** Robot top speed 1.0 m/s, turn rate 1.0 rad/s; obstacles 0.3-1.5 m/s so ~1/3 move FASTER than the robot, mean ~0.9 m/s. A 90deg turn takes ~1.57s during which a 1.5 m/s obstacle covers 2.4 m. There is an un-evadable crash floor (likely 15-25%). MUST pre-compute this floor before building anything.
2. **Data association is genuinely unsolved with this lidar.** 360 anonymous ranges, NaN=no return, no intensity/ID, ~20 obstacles + walls all in one scan, continuous despawn/respawn. A bad estimate injects PHANTOM obstacles into open corridor -> actively worse than the memoryless baseline (no graceful degradation).
3. **Ego-motion corrupts every estimate.** 1.0 rad/s rotation x 0.1s = 0.1 rad -> ~1 m phantom shift at 10 m range = 10 m/s phantom velocity, dwarfing the 0.3-1.5 signal. Correctable via known pose but residual is comparable to signal exactly when maneuvering.
4. **Time-indexed cost breaks D* Lite's static-edge model + enforced determinism.** Either you do the memoryless fold with a shifted disk (gains little) or build a real space-time search (different algorithm, discards incremental advantage, per-tick settle returns -> the 600s-wall problem). Estimator float non-associativity flips trace bits -> fails the 55-case --check.
5. **Estimate-to-action latency is structurally too long.** >=2 frames to estimate + deferred settle + ~1.5s to physically turn = 1-2s pipeline lag; a 0.9 m/s obstacle moves 1-2 m in that time. Narrow-to-empty window where prediction is both accurate and early enough.
6. **This risks reinventing DWA badly.** Repo memory already says only velocity-aware DWA dodges. Global planner is the WORST place to add millisecond evasion. Likely-correct design = HYBRID: keep D* Lite global + thin DWA/VO local veto layer. The seed conflates "can lidar-velocity help a global planner" (probable null result) with "stop D* Lite crashing" (answer: local reactive layer).

### Feasibility — VERDICT: Conditional (possible within every constraint; one real determinism trap)
- No HARD BLOCKERS. Lidar-only is fine (velocity recoverable from frame differencing); determinism preserved if estimator is a pure deterministic fn of (state, lidar) + prior internal state; D* Lite absorbs predictive cells through the EXISTING update_cells seam (TC46 proves batched update_cells + one settle == from-scratch A*).
- Must-be-true: new registry key (reject --replan-k); act() never raises mid-episode; static-grid cost core untouched (velocity enters as changed CELLS, not a new edge-cost term); grid-ownership mutate-in-place + report-every-flip honored.
- Most feasible path: (a) estimate via reuse of dwa `_lidar_to_world_points` -> subtract static returns using self._static_cells -> deterministic grid-bucket clustering (hand-rolled, NOT sklearn) -> sorted greedy NN association -> (centroid_now-centroid_prev)/dt with speed gating; OR a cheaper per-cell occupancy-FLOW/decay field (no association). (b) Use it as PREDICTIVE SWEPT-SHADOW inflation stamped into the fold BEFORE the diff, via the existing _mark_disk + update_cells path — one-function insertion, no search rewrite. NOT space-time (total rewrite), NOT a new edge-cost.
- Effort: Medium (~200-300 line new planner file lifted from dwa+d_star_lite + 2-3 TCs). Riskiest unknown: deterministic AND accurate-enough cross-frame association while the robot moves. De-risk by prototyping the estimator as a pure function on synthetic 2-frame fixtures (TC46/47-style, no irsim) and proving byte-identical output BEFORE wiring it in.
- Strong recommendation: build the DWA-style TTC/VO veto layer FIRST as the lower-risk MVP (Small effort, determinism risk confined to estimator, doesn't touch the search).

### Thinker (branches)
1. **Two-frame range-rate stamp** — per-beam (r_t - r_{t-1})/dt after ego-motion compensation; zero data association; gate speed / inflate that bearing forward. Cheapest "see motion" layer.
2. **Cluster-and-track + constant-velocity line fit** — segment finite returns into clusters, sorted greedy NN association, least-squares line through centroid history (exploits the straight-line prior). Clean (vx,vy), no Kalman tuning.
3. **Per-cluster constant-velocity Kalman** — 4-state [x,y,vx,vy] KF, fixed noise; predict-through on occlusion; covariance -> uncertainty-scaled risk radius. Feeds branch 9.
4. **Predictive swept-volume "shadow" inflation** — mark a capsule from p to p+v*T into the fold; D* Lite routes behind traffic with zero search edits. Asymmetric forward-only inflation is the cheap version. (Feasibility's recommended planner-half.)
5. **Space-time D* Lite over (x,y,t) slices** — textbook-correct; each layer = static grid + predicted disks at t*dt; edges advance in time + optional wait self-edge. Correct but K-layer-deeper grid, big change.
6. **VO/RVO/ORCA veto on top of the global path (hybrid)** — D* Lite finds the route on the static map; a reactive cone filter projects the commanded (v,w) out of any velocity-obstacle cone. Highest crash-reduction-per-line; directly comparable to DWA.
7. **TTC-gated speed governor (dumb-but-effective)** — min forward TTC = range/closing_speed; below threshold scale v down / to zero. Reframes avoidance as SPEED control, a single scalar on D* Lite's output. May capture most of the win.
8. **Wait/yield primitive** — a v=0 hold when the predicted swept-volume shows the segment clears if the robot just pauses; swerving into a different obstacle is a real failure mode at 20-obstacle density.
9. **Chance-constrained / risk-map planning** — per-cell collision-probability field from KF covariance; penalize/threshold cells above prob bound delta. Naturally trades a longer path for lower crash = "down-left"; gives a sweepable knob delta.
10. **Intent/flow learning over the episode** — accumulate a coarse traffic-flow field from edge-spawn structure, bias the global cost toward low-traffic corridors BEFORE any obstacle is visible. Strategic, not reactive; cheap online stats.
11. **Reachability / inevitable-collision-state shielding** — prune commands leading to states with no evasive option (preserve the robot's OWN escape options). Distinct philosophy from prediction; matters because some obstacles are faster.
12. **WILD CARD: oracle-velocity ceiling study + new TTC-margin metric** — register `d_star_lite_oracle` reading true (vx,vy) to bound how much crash reduction is even achievable, so the lidar-only gap is interpretable; add a closest-approach/min-TTC metric so "barely survives" differs from "comfortable margins."
13. **Optical-flow on the range image** — Lucas-Kanade / phase-correlation on the 360 range vector across frames -> angular flow -> lateral velocity. Sidesteps association entirely; degrades gracefully in clutter.

## Live direction
**1 + 3, sequenced.** First build the `d_star_lite_oracle` ceiling study (Thinker
#12) to measure the achievable crash-reduction headroom with true vx/vy; the
oracle also gives the predictive-stamp logic a tested home BEFORE lidar
estimation is added. Then build the lidar-only **predictive swept-shadow
stamping** variant (Thinker #4): estimate velocity from frames, stamp each
obstacle's future swept-capsule into the fold via the existing `update_cells`
seam so D* Lite routes BEHIND traffic. User wants to go deeper on #3 (the
stamping design sub-space).

## Perspectives (round 2) — Thinker on the stamping design sub-space
SHADOW GEOMETRY: 1 end-disk-only (cheapest, conservative "go behind the nose") · 2 full Bresenham swept capsule (literal, highest over-block) · 3 time-sliced disk train (K _mark_disk calls; K is an aggression knob via overlap/gaps) · 4 **forward-asymmetric** (stamp only the leading hemisphere along v; trailing wake stays open = "route behind"; halves area) · 5 thin leading-edge curtain (one short arc d_lead ahead; minimal, near-zero wall-in risk).
HORIZON: 6 robot-reachability-scaled T (= dist_to_encounter / robot_speed; far traffic gets reach only where the robot would actually meet it) · 7 closing-line-intersection horizon (truncate the capsule at where the obstacle ray crosses the committed segment; nothing for non-crossing rays).
CONFIDENCE/SAFETY: 8 **collision-course selectivity gate** (stamp only closing obstacles whose track passes near the committed segment — the biggest lever vs phantom walls, since most of the 20 aren't on a collision course) · 9 **confidence-modulated reach** (1-frame/jittery track -> reach 0; stable multi-frame -> full T; converts the worst failure mode into graceful fallback) · 10 plausibility-clamped velocity (clamp speed to 0.3-1.5 m/s, reject >1 step displacement, BEFORE stamping; guards association blow-ups) · 11 **stamp-then-verify-then-relax** (reachability probe after stamping; peel back farthest slices / lowest-confidence cluster until a path re-exists = fail-OPEN, can make robot cautious but never trap it).
COST/INTERACTION: 12 stamp into a SEPARATE overlay used only by `_immediate_segment_blocked` (prediction drives WHEN to replan without paying over-block cost in the search) · 13 area-budgeted stamp (hard per-tick cell ceiling, allocate to nearest-closing/lowest-TTC first; bounds over-block + update_cells cost deterministically) · 14 **T_lookahead as a swept knob** (mirror --speed-regime; T in {0,0.5,1,2}s over 50 seeds; T=0 IS plain d_star_lite so the sweep self-proves whether stamping helps + picks the down-left optimum).
DETERMINISM: branches 1/3/4/5 reuse the byte-stable `_mark_disk`; stamp must be applied to `folded` BEFORE the `diff_mask` line (d_star_lite.py:797) so predictions enter purely as changed cells. Failure-mode map: phantom walls -> 8/9/10/11; walled-in -> 6/7/11/13; over-block-around-fast -> 4/5/13.
**Thinker's recommended stack: 4 (forward-only) + 8 (closing-course gate) + 9 (confidence reach) + 11 (fail-open guard), wrapped by 14 (sweep the horizon).**

**USER REFINEMENT (2026-06-27): geometry = forward-only EXPANDING CONE**, not a
constant-width capsule. Narrow at the obstacle's current cell (we know where it
is now), fanning WIDER the further ahead it reaches (growing position
uncertainty + the obstacle could change course). Implement as a disk TRAIN
(branch #3) with GROWING radii along v: disk at p = obstacle_r + inflation, each
later disk at p + v*(k/K)*T slightly bigger, union = a funnel/cone. Reuses the
byte-stable `_mark_disk`, so determinism holds. This geometrically fuses #4
(forward-only) + #3 (disk train) + #9's uncertainty idea into the shadow SHAPE
itself.

**HARD SAFETY RULE (user, 2026-06-27): the stamp must never block the robot's own
cell / footprint.** Two real failure modes if it does:
- At t=0 `reset()`: the stamp/settle is NOT wrapped in try/except, so a stamp that
  seals the start makes `extract_path` raise -> the runner records the whole
  episode as `planner_error`. AVOIDED FOR FREE: velocity needs >=2 frames, so at
  t=0 there is no estimate -> reset() does the plain fold, no predictive stamp.
- Mid-episode `act()`: the settle/extract IS wrapped in try/except (swallowed), so
  it won't crash, BUT a stamp on the robot's cell makes `_immediate_segment_blocked`
  fire every tick -> replan thrash + heading whipsaw + the stale-path-keep means
  the robot may drive into the obstacle anyway (defeats the purpose).
- FIX: a robot EXCLUSION ZONE — clip the cone so no cell within radius R of the
  robot's current position is ever stamped. Semantically correct too: if the cone
  reaches the robot, that's an IMMINENT collision the global stamp can't fix
  (can't reroute around your own square) -> hand off to a reactive response
  (slow/stop), not a stamp. So clipping at the robot is both safe AND right.

## Killed / parked

## Decisions & debate verdicts
