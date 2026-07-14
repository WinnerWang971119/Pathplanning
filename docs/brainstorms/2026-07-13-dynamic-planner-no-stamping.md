# Brainstorm: 3rd-try dynamic path planner (no stamping)

**Seed (verbatim):** Ok bro. This is our 3rd try on how to make a actually good
dynamic pathplanner that works. But not using the stamping method. You can first
check why me faild the first 2 times and we /brainstorm

**Context summary:**
- Goal (Mission.md): move a diff-drive robot (2,2)->(48,48) through ~20-obstacle
  crossing traffic; scoring is time-to-goal vs crash-rate scatter — down-left wins.
  After t=0 the planner sees ONLY lidar (no true vx/vy). Byte-identical determinism
  is load-bearing. Must fit the `Controller` reset()/act() interface + register a key.
- **Try 1 — predictive D* Lite (stamping):** stamp each obstacle's predicted future
  footprint (growing cone) into the occupancy grid before the diff. Oracle (perfect v)
  WORKED (0.60->0.20 @h10). Lidar estimator NET-HARMFUL (0.60->1.00): frame-diff
  tracker reported 27 clusters from 20 obstacles, jittered velocities, stamp hit
  ~5128/6000 cells -> phantom-walled corridors. Structural fault: stamping collapses
  the time axis (a cell merely passed-through reads permanently blocked).
- **Try 2 — space-time predictive DWA:** advance each tracked obstacle inside the
  DWA rollout, check at matched time step. Blanket reject = freezing robot
  (0.50->0.80). Rebuilt to braking-inevitability + soft yield + cost-to-go field:
  braking mildly WON (0.40@h5 oracle), global field NET-HARMFUL (drove into traffic),
  lidar variant only TIED plain DWA. Insight: DWA already reasons in velocity space,
  so a hard prediction layer double-counts movers and freezes it.
- **Three recurring killers:** (1) perception is the wall — every oracle helped,
  every lidar estimator failed/tied; (2) over-conservatism / freezing in dense
  traffic; (3) a real kinematic crash floor — ~1/3 of obstacles are FASTER than the
  robot (0.3-1.5x top speed); the `fast` (2.0x) regime sits at 0.56-0.60 for all.
- **Memory:** replanning is NOT safer than single-plan; only velocity-aware DWA truly
  halves hazard (0.48/100s vs ~0.9). Commitment horizon needed for any replanner.
  Prior brainstorm (2026-06-27) already surfaced VO/ORCA veto, TTC speed governor,
  chance-constrained risk maps, reachability/ICS shielding, optical-flow estimation.

**Clarified framing:** Try 3 is a HYBRID controller — a global planner (D* Lite or
A*) produces the route on the static map, and a thin REACTIVE LOCAL layer
(velocity-obstacle / TTC-based) filters or scales the commanded (v, ω) to avoid
crossing traffic. Explicitly NOT grid stamping. Lidar-only after t=0 (no true
vx/vy), must fit the `Controller` reset()/act() interface, register a new algorithm
key, and preserve byte-identical determinism. The success bar is to OWN THE
DOWN-LEFT CORNER: a REALIZABLE (non-oracle) lidar controller that approaches the
D* Lite oracle's 0.20 failure rate at comparable time-to-goal (~86 s), i.e. clearly
beating plain DWA's 0.50 and landing best-or-near-best on the scatter. The
perception bet is left OPEN — the agents should argue whether to (a) design a policy
that tolerates crude/no velocity, or (b) invest in a genuinely better tracker.

## Perspectives (round 1)
### Critic
1. **Hybrid inherits the exact velocity estimator that killed both tries — and the local layer is MORE velocity-sensitive than stamping.** A VO/ORCA/TTC veto consumes estimated obstacle velocity vectors, the identical quantity `LidarTracker` produces (27 clusters from 20, jittery). Stamping smeared a noisy velocity over a footprint the fail-open peel could un-stamp; a hard VO half-plane / TTC threshold turns one bad estimate into an instant wrong veto with no recovery → strictly more fragile. "Perception bet left open" is false — every realizable estimator to date is net-harmful or neutral.
2. **VO/ORCA is the wrong tool for anonymous lidar + non-reactive obstacles.** ORCA assumes reciprocity (all agents share the policy); arena obstacles move dead-straight and never react, so the robot takes 100% of the burden — assuming they take half under-maneuvers into a crash. Collapses to plain VO, which still needs each obstacle as an individuated (pos, vel, radius) disc — exactly the data association lidar can't supply.
3. **A local veto over a global path just relocates the freezing-robot failure (0.50→0.80 in Try 2).** ~20 obstacles each carve a forbidden cone from (v,ω); with ~1/3 faster than the robot the cones are wide and jointly cover every forward command. A TTC governor scales v→0, and since population refills, slowing just extends exposure → timeout (folded into failure).
4. **The global+local split is architecturally incoherent for CROSSING traffic.** The global planner commits to static structure, but the danger is moving obstacles the static route ignores → the hybrid is really "plain DWA with a fixed heading suggestion." Try 2's cost-to-go field already measured that "global heading into traffic" idea net-harmful (1.00). Re-proposes the demoted `dwa_predictive` under a new name.
5. **Determinism is a landmine for VO/ORCA/TTC.** ORCA's half-plane LP intersects constraints in an order that decides ties; TTC's "most-threatening obstacle" is an argmin over floats where a sub-ULP diff flips the command and cascades into a different lidar hash. Achievable but a hidden tax the "thin layer" framing conceals.
6. **The 0.20 bar IS the oracle number — no non-oracle controller here has come near it.** Pattern is total: oracle 0.20 vs lidar 1.00 (D* Lite); oracle 0.40 vs lidar 0.50-tie (braking DWA). The kinematic floor caps the win (oracle itself is 0.20 not 0.0 because ~1/3 outrun the robot → genuine inevitable-collision states). Asking a realizable estimator to match the oracle asks it to pay zero perception cost on a task where perception is the dominant cost.
**Critic's strongest line:** #1+#2 — the hybrid re-approaches the wall that killed Tries 1&2 with a MORE brittle instrument (hard veto, false reciprocity, no fail-open).

### Feasibility — VERDICT: Conditional (no hard blockers; every constraint has a working precedent)
- **Must be true:** new registry key (predict-family if it takes a TTC/lookahead knob, plain like `dwa` otherwise — both reject the wrong flags for free); `act()` never raises (guard the veto, fall back to un-vetoed global cmd, model = `dwa_predictive.py:372`), only the t=0 global plan may raise; estimator (if any) pure-deterministic (existing `LidarTracker` already meets this); **a veto that vetoes must degrade to SLOWING, not freezing** (scaling/yield governor or ICS test, never any-conflict hard reject — the lesson the code already learned).
- **Hard blockers: NONE.** No structural reason a hybrid can't be built.
- **Key split on perception:**
  - (a) **VO/ORCA needs velocities FIRST** — defined over per-obstacle (pos, vel) constraints, so it can only run after the same data-association + velocity-estimation step that sank Try 1. Literature confirms real ORCA feeds a Kalman tracker and "perfect sensing assumption limits performance." VO/ORCA INHERITS the perception wall, doesn't sidestep it. Only pursue if committing to a better estimator (KF / const-velocity line-fit).
  - (b) **TTC/range-rate speed governor genuinely SIDESTEPS data association — the strongest realizable path.** TTC = range / closing-rate is computable PER BEAM from (r_t − r_{t−1})/dt after ego-motion compensation: no clustering, no association, no IDs → avoids the 27-clusters-from-20 over-segmentation that made frame-differencing net-harmful. Reuses the D* Lite global route unchanged; only scales the follower's linear velocity by a min-forward-TTC gate. Residual: ego-rotation injects phantom range-rate (1.0 rad/s × 0.1 s at 10 m ≈ large phantom closing speed) → compensate with known pose delta, gate on the forward sector during low-|ω| ticks.
  - (c) Determinism seam is clean: compute global (v,ω), apply a pure scalar governor before returning. One extra sqrt-reduction over the scan, no set-iteration.
- **On the 0.20 bar:** plausible but NOT guaranteed. Oracle hit 0.20 with perfect velocities; a governor that slows/stops (rather than swerving into a 2nd obstacle) attacks crashes without clean vx/vy. BUT **the "fast regime 0.56–0.60 floor" is a PROJECTION — no speed-sweep findings file exists in the repo; that run is still unlaunched.** Realistic `current`-regime target: "clearly beat 0.50, land in 0.25–0.40," with 0.20 a stretch needing governor + a modest estimator, not the governor alone.

### Thinker (branches)
### Thinker (branches)
1. **Range-image angular flow as the whole veto** — work on the raw 360-vector, derotate by shifting the ring by Δθ/increment, compute per-beam radial rate ṙ and angular flow θ̇; scale v down only on beams with strongly-negative ṙ AND near-zero θ̇ (constant-bearing = collision course). Never forms discrete objects → can't over-segment.
2. **CBDR collision-cone veto (the TCAS/sailor analogy)** — two objects collide iff bearing stays constant while range shrinks. Per beam compute bearing-rate + range-rate; veto a (v,ω) only when it fails to break constant-bearing against a closing return. Weaker/cheaper than VO cones, no vx/vy reconstruction, resists freezing (most returns have nonzero bearing-rate and pass).
3. **Collision-cone CBF safety filter (C3BF-QP)** — a Control Barrier Function computes a MINIMALLY-INVASIVE correction to the nominal (v,ω) keeping relative velocity out of each collision cone; tiny per-tick QP or closed-form 2-DOF projection. Theory-backed, minimally-invasive by construction = direct antidote to freezing. (Speculative on determinism: needs closed-form or fixed-iteration solver.) [arxiv 2209.11524, 2503.00606]
4. **Passive-safety / braking-ICS shield (ATTACKS THE FAST-REGIME FLOOR)** — don't predict; guarantee the robot can always STOP before contact given what it sees now. Cap v so v²/(2·a_brake) ≤ nearest-closing-clearance; veto any state with no stop-in-time. Provably passive-safe (a faster obstacle hitting a STOPPED robot is the obstacle's fault). The one branch built for un-evadable 1.5× crossers. [braking-ICS lit]
5. **Maneuverability-first: escape-corridor preservation (inverted assumption)** — the lever is the robot's OWN future freedom, not obstacle prediction. Score candidates by how many distinct escape gaps remain open at the rollout endpoint; prefer keeping ≥2 exits. Pure current-free-space geometry, no velocity estimation → can't be poisoned by bad estimates.
6. **Gap-seeking between movers ("cross behind, never in front")** — exploit the edge-spawn straight-line prior: safe crossing is always BEHIND an obstacle along its inward bearing. Build a per-tick wake map (trailing half-plane safe, leading penalized), bias local heading toward wakes. Needs only rough inward direction (from perimeter-spawn geometry), not speed. (Speculative: leans on arena spawn structure.)
7. **Two-tier speed governor + hard "freeze budget" (ATTACKS OVER-CONSERVATISM)** — pure scalar v-governor on the global heading (never touch ω), scaled by min-TTC, PLUS an anti-freeze integrator: track cumulative time below a speed floor; once a "stuck budget" trips, force a commit through the largest gap. Encodes "short yield wins, freezing loses" as explicit controller state. Cheap, deterministic, one scalar of hidden state.
8. **Occupancy-decay flow field, no association (ATTACKS OVER-SEGMENTATION)** — never cluster into objects; maintain a decaying per-cell occupancy with a directional hint (free→occupied gives displacement direction), blended with exponential decay so jitter averages out instead of spawning phantoms. Association-free middle ground between raw range-rate and full tracking. (Speculative: decay must be a fixed-point-safe deterministic recurrence.)
9. **Oracle-ceiling study for the veto layer itself (de-risking branch-0)** — build `hybrid_veto_oracle`: global D* Lite + reactive veto fed PERFECT vx/vy via the truth seam. Measures the ceiling of the HYBRID ARCHITECTURE specifically (distinct from the stamping/DWA oracles). If even the oracle veto can't approach 0.20@86s the architecture is wrong; if it can, the gap is purely perception and you know how much estimator quality to buy. OracleTracker already exists → cheap, high info. Should be sequenced first.
10. **Speed-adaptive commitment horizon coupling global↔local** — make D* Lite's commitment horizon (when it swaps followers) a function of the local threat level: low threat → commit long (smooth/fast); high threat → shorten so the global path re-forms around the reactive detour. Closes the open loop where the veto and the oblivious global path currently fight. Reuses existing commitment-horizon machinery.
11. **Yield-lane global planning: route through low-flow corridors a priori (strategic, pre-reactive)** — accumulate a coarse traffic-density estimate per region from lidar over the first seconds, bias the global cost to prefer historically-quiet corridors. Strategic avoidance under tactical avoidance; cheap slow-timescale stats, decoupled from the fast veto so they don't interfere.
12. **Matched-time rollout, but SOFT and ASYMMETRIC (fixes DWA-predictive's freeze)** — keep matched-time reasoning but (a) graded penalty ∝ 1/matched-time-gap instead of hard reject, and (b) asymmetric: heavily penalize being IN FRONT of an obstacle's heading at matched time, lightly penalize behind. Bakes "cross behind" into the space-time score. The braking policy that worked + directional asymmetry − the blanket reject that failed.

## Live direction
**USER PIVOT (2026-07-14):** rejected the 4 mechanism options (soft governor /
braking shield / oracle-first / CBF). New framing chosen by the user: **A* computes
the initial global path ONCE at t=0 (static map), then a purely LOCAL algorithm
that operates ONLY within the lidar sensing bubble handles the moving obstacles —
locally deviating off the A* line and rejoining it.** Classic global-planner +
local-planner split. The global half is ~done (`a_star_once` already plans the
static A* path at t=0); the open question is the LOCAL layer's rules. Thinker
re-summoned (round 2) to generate concrete local-layer designs. Presentation must
stay PLAIN-LANGUAGE (user asked to drop the jargon).

## Killed / parked
- VO/ORCA hard veto — Critic+Feasibility: needs per-obstacle velocity (the wall)
  and is a MORE brittle instrument than stamping (no fail-open, false reciprocity).
- The 4 mechanism options as *presented* — user rejected the framing; superseded by
  the A*-global + lidar-local pivot above (the governor / braking-shield / CBF ideas
  may still return as candidate LOCAL layers).

## Local-layer designs (Thinker round 2) — rides on a fixed t=0 A* path, reasons only in the lidar bubble
Zero-estimation (pure current-frame geometry): **1, 6, 7**. Same-beam range-diff / bearing-persistence only (no per-obstacle vx/vy vector): **3, 4, 8, 10**. Strongest prior evidence: **2**. All degrade to slowing/nudging, never a hard freeze.
1. **Follow-the-gap toward the A* carrot** — bubble the lidar, find widest safe gap, pick the gap by blend(gap width, angular closeness to next-waypoint bearing). Obstacle straddles the line → widest gap shifts off-line → robot slides around → carrot pull rejoins. NO velocity. Can't freeze (a gap almost always exists). Cleanest match to the user's idea. [FGM]
2. **DWA-as-local-layer, aimed at the A* carrot** — reuse shipped `DWAController`; only override `_heading_term` to aim at the next waypoint instead of the far goal → DWA becomes a local corridor follower. NO estimation. Strongest evidence (repo DWA already halves hazard). Lowest risk (proven code).
3. **TTC speed governor on an unchanged A* follower** — per-beam closing-time = range / (same-beam range-decrease); multiply the follower's v by a smooth factor that → crawl as min forward-arc TTC drops; leave ω untouched. Never leaves the path, only modulates speed. NO per-obstacle vel. Anti-freeze: v floors at a crawl + restore ramp after N stalled ticks.
4. **CBDR lateral nudge** — flag beams on constant bearing + decreasing range over 2-3 frames (the maritime "it will hit you" test); add a lateral ω away from flagged bearings on top of the follower heading. Only nudges the ~1/3 truly on a collision course, ignores the rest. NO velocity. Only nudges heading, never zeros v. Best threat-selectivity (anti-phantom-wall).
5. **Local rolling-window mini-A*** — fold the current lidar onto a small ~6-8 m patch around the robot, run a tiny A* from robot → the cell where the global path exits the window; rejoin is structural (local goal = path exit). Bounded window → a moving obstacle only distorts a few metres, never an arena-spanning phantom wall (Try-1's failure). Reuses `manual_astar` search wholesale.
6. **Elastic-band deformation of the A* waypoints** — treat the waypoint list as a rubber band: neighbor contraction (taut) + lidar-point repulsion (push off obstacles); 1-2 Gauss-Seidel steps/tick, then the existing follower drives the deformed band. Bows around traffic, springs back. Classic (untimed, position-only) band → NO estimator. Fixed-iteration → deterministic. Very literal "deviate + rejoin". [elastic band/TEB core]
7. **Admissible-gap navigation** — like #1 but pick the closest gap the robot can *actually steer through* given turn/accel limits (test a fixed arc set per gap, closest-first). Guarantees a feasible motion exists before committing → never picks an opening it must then stop in front of. NO velocity. [Admissible Gap / Closest Gap]
8. **Fast-regime "duck-behind-the-nose" governor (THE fast-crosser answer)** — accept ~1/3 outrun the robot; on a fast-crosser signature (fast decreasing range + bearing sweeping across the front) do a bounded stop-and-let-pass (cut v, hold heading, ≤~1.5 s) rather than swerve into it (a 1 m/s swerve loses to a 1.5 m/s crosser). Bounded yield window then escalates to the gap-follower. NO velocity vector (just range-rate + bearing-sweep). Layers on top of any of 1-7.
9. **Path-anchored potential field (speculative)** — reuse `apf`: repulsion from lidar returns + attraction to the nearest point on the A* segment ahead (a line tether, NOT the far goal → avoids Try-2's drive-into-traffic). Tangential escape on cancellation (orbit the blockage) fixes APF minima. NO velocity.
10. **Range-image optical-flow avoider (speculative)** — treat the 360-vector as a 1-D image; integer beam-shift cross-correlation → angular flow → steer to pass BEHIND the flow. NO association. Determinism needs integer-shift (no sub-pixel), de-rotate by known ω·dt first. Head-on (zero-flow) falls through to #3.

## Killed / parked (cont.)

## Decisions & debate verdicts
