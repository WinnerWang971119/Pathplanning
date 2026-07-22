"""planners/_predict.py — shared predictive substrate for motion-aware D* Lite.

The LOGIC here is pure: plain floats/ints + numpy in, deterministic output, no
irsim/RNG calls, no set-iteration leaking into output order. The shared grid
geometry it builds on lives in the pure ``planners._geometry`` module; the one
manual_astar helper it needs at runtime (``point_to_polyline_distance`` for the
conflict gate) is lazy-imported inside ``predict_blocked_cells``.

NOTE: "pure" describes the computation, NOT the import. Like every ``planners``
submodule, importing this one runs the package ``__init__`` (which eagerly imports
the controllers) and therefore pulls irsim + matplotlib — the documented
"importing planners pulls irsim + matplotlib" gotcha. A headless tool does NOT
become irsim-free by importing from here; it must still lazy-import ``planners``
symbols inside functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from planners._geometry import iter_disk_cells, lidar_to_world_points

if TYPE_CHECKING:
    # Annotation-only (stringified by `from __future__ import annotations`), so it
    # never runs at import time. The one runtime manual_astar symbol — the conflict
    # gate's point_to_polyline_distance — is lazy-imported inside predict_blocked_cells.
    from manual_astar import OccupancyGrid

# Prediction timestep in seconds.  Matches irsim step_time and DWA CONTROL_DT.
PREDICT_DT: float = 0.1

# Per-step radial growth (metres) applied to the cone geometry's stamp radius.
# The cone widens with the lookahead step to represent estimator uncertainty
# (zero for the oracle, nonzero for the lidar variant).  Chosen as roughly half
# a grid cell per step (GRID_RESOLUTION is 0.1 m, so ~0.05 m), a small default
# tuned later by the lidar variant (T12).  The capsule geometry ignores it.
CONE_GROWTH_PER_STEP: float = 0.05

# --- LidarTracker detector tunables ------------------------------------------

# Grid-bucket edge length (metres) for clustering surviving dynamic lidar
# returns into obstacle clusters.  Roughly one obstacle radius (the arena's
# dynamic obstacles are r=0.3 m circles), so a single obstacle's arc of returns
# falls into a few adjacent buckets that 8-connectivity then merges into one
# cluster, while two distinct obstacles a metre apart stay separate.
CLUSTER_RESOLUTION: float = 0.3

# Floor (metres) on a detection's estimated radius; also the floor on the seed
# of a track's radius EMA at birth.  A one-point cluster has zero extent, so
# without a floor its predicted disk would collapse to nothing; lidar only ever
# sees the near surface and under-estimates the true radius anyway (the cone
# geometry widens to cover that), so a small sane floor is all that is required.
MIN_TRACK_RADIUS: float = 0.15

# --- LidarTracker Kalman-filter MOT tunables ---------------------------------

# Constant-velocity process-noise intensity q (accel white noise, m^2/s^3),
# feeding the continuous CV process noise
# Q(q, dt) = [[q*dt^3/3, q*dt^2/2], [q*dt^2/2, q*dt]].  Sized so the filter
# follows a ~1 m/s straight-line mover without lag while damping merge-frame
# centroid spikes.
KF_PROCESS_NOISE: float = 4.0

# Centroid measurement variance r (m^2), ~ (0.14 m)^2 for the near-surface
# centroid jitter of a clustered lidar arc.  Must stay positive: the scalar
# innovation variance is Sm = P00 + r and the Kalman gain divides by it
# (checked at construction, so no per-update division guard is needed).
KF_MEASUREMENT_NOISE: float = 0.02

# Birth covariance on the position states (m^2): a first detection's centroid
# is trusted to within roughly two grid cells.
KF_INITIAL_POSITION_VARIANCE: float = 0.05

# Birth covariance on the velocity states (m^2/s^2): deliberately wide, so the
# first gated hits move the velocity estimate freely instead of dragging a
# zero-velocity prior.
KF_INITIAL_VELOCITY_VARIANCE: float = 4.0

# Fastest obstacle speed the tracker must be able to follow (m/s).  Deliberately
# 2.0 — the ``fast`` obstacle-speed regime's max cap (0.5..2.0 * robot top
# speed) — NOT the ``current`` regime's 1.5: a gate sized to 1.5 would make
# every genuine fast-regime crosser fail association each frame, coast out, and
# die, silently un-tracking exactly the traffic the issue-#11 speed sweep spawns.
MAX_PLAUSIBLE_SPEED: float = 2.0

# Centroid/cluster jitter allowance (metres) added to the physical per-frame
# displacement bound when sizing the association gate.
GATE_SLACK: float = 0.15

# Maximum detection-centroid-to-PREDICTED-track-position distance (metres) for
# a valid association.  A fixed Euclidean physics gate (NOT a Mahalanobis /
# innovation-covariance gate): the fastest plausible mover displaces at most
# MAX_PLAUSIBLE_SPEED * PREDICT_DT per frame beyond its prediction, plus slack
# for centroid jitter (~0.35 m total).
ASSOCIATION_GATE_DISTANCE: float = MAX_PLAUSIBLE_SPEED * PREDICT_DT + GATE_SLACK

# Consecutive gated hits (counting the birth detection) required to promote a
# TENTATIVE track to CONFIRMED.  Tentative tracks are withheld from update()
# output, so a 1-2 frame spurious cluster is never emitted (or stamped).
CONFIRM_HITS: int = 3

# Consecutive misses a CONFIRMED track coasts on its prediction before
# deletion: it stays emitted (at its PREDICTED position) through misses
# 1..COAST_MISSES — <= 0.3 s at PREDICT_DT — then deletes on the next miss.
COAST_MISSES: int = 3

# Radius EMA weight: filtered = beta * measured + (1 - beta) * previous.
RADIUS_EMA_BETA: float = 0.3

# Hard cap (metres) on a track's filtered radius — 1.5x the arena's true 0.3 m
# mover.  Applied to the birth seed AND to every EMA update, so no reported
# Track.radius can exceed it even when a merge balloons the raw cluster radius
# (AC6).
RADIUS_MAX: float = 0.45

# A detection whose measured radius jumps above this ratio times the track's
# filtered radius is treated as a merge suspect: the radius EMA update is
# SKIPPED that frame (the position update still happens).  Never applied on the
# birth frame — there is no prior filtered radius to compare against.
RADIUS_MERGE_SUSPECT_RATIO: float = 1.5

# The near-rim no-hit deadband (RANGE_MAX_DEADBAND) the LidarTracker applies when
# projecting beams now lives with the shared projection in planners._geometry.


class ThreatKey(NamedTuple):
    """Sort key placing soonest-conflict tracks first, ties broken by id.

    Ascending order over (ttc_steps, track_id) is exactly threat order:
    the track whose predicted footprint first intersects the planned-path
    corridor comes first, and equal time-to-conflict ties resolve by stable
    obstacle id so the grouped output is deterministic.
    """

    ttc_steps: int
    track_id: int


@dataclass(frozen=True)
class Track:
    """A single tracked obstacle at the current tick."""

    id: int          # stable identity (oracle: obstacle id; lidar: per-episode birth-counter id)
    x: float
    y: float
    vx: float        # m/s, world frame
    vy: float
    radius: float


class Tracker(Protocol):
    """Interface for velocity-source adapters consumed by the predictive controller.

    OracleTracker.update reads *snapshot* (ignores state/lidar/dt) — velocities
    are exact/known from the live truth seam.

    LidarTracker reads *state* and *lidar* (ignores snapshot) — velocities are
    estimated by a constant-velocity Kalman-filter multi-object tracker over
    clustered dynamic lidar returns.

    Both return tracks sorted by id ascending to guarantee deterministic ordering.
    """

    def update(
        self,
        *,
        snapshot: object,
        state: object,
        lidar: object,
        dt: float,
    ) -> list[Track]:
        ...


class OracleTracker:
    """Trivial tracker that converts the live oracle snapshot to a Track list.

    Reads ``snapshot`` (a tuple of DynamicObstacleState records from
    EpisodeInfo.dynamic_obstacles).  ``state``, ``lidar``, and ``dt`` are
    accepted for protocol compatibility but are ignored — the velocities are
    exact and already present in the snapshot.

    The DynamicObstacleState type is NOT imported here; attributes are read
    via duck typing so this module stays decoupled from arena.
    """

    def update(
        self,
        *,
        snapshot: object,
        state: object,
        lidar: object,
        dt: float,
    ) -> list[Track]:
        """Return id-sorted Tracks from the oracle snapshot.

        Parameters
        ----------
        snapshot:
            A tuple of records each exposing .id, .x, .y, .vx, .vy, .radius.
            Pass ``()`` (or an empty tuple) when traffic is off.
        state, lidar, dt:
            Ignored by this implementation.
        """
        # Suppress "unused argument" intent — these are accepted for protocol
        # conformance only.
        del state, lidar, dt

        tracks = [
            Track(
                id=rec.id,
                x=rec.x,
                y=rec.y,
                vx=rec.vx,
                vy=rec.vy,
                radius=rec.radius,
            )
            for rec in snapshot  # type: ignore[union-attr]
        ]
        tracks.sort(key=lambda t: t.id)
        return tracks


@dataclass(frozen=True)
class _Cluster:
    """One connected-component cluster of dynamic lidar returns (internal).

    A per-frame DETECTION, not a track.  ``rep_cell`` is the smallest bucket
    key in the component — its deterministic representative, unique within a
    frame (distinct components have distinct rep_cells) — and fixes the frame's
    detection order: detections are ranked in ``rep_cell``-ascending order for
    the association sweep.  Cross-frame identity lives in the tracker's
    persistent ``_KTrack`` records, which carry monotonic birth-counter ids.
    """

    rep_cell: tuple[int, int]
    centroid_x: float
    centroid_y: float
    radius: float


@dataclass
class _KTrack:
    """Internal per-track record for the CV-KF multi-object tracker.

    Mutable by design: the tracker advances the Kalman state in place each
    frame.  ``id`` is a monotonic per-episode birth counter, assigned at birth
    and never reused.  The Kalman state is ``[x, y, vx, vy]`` with a DECOUPLED
    covariance: axis-symmetric noise makes the x and y axes two identical
    2-state filters sharing one covariance recursion, so a single symmetric
    2x2 ``[[p00, p01], [p01, p11]]`` (position variance, position-velocity
    cross term, velocity variance) serves both axes and the Kalman gain is one
    shared 2-vector.  ``radius`` is a separate capped EMA, not part of the KF
    state.  ``hits`` / ``misses`` / ``confirmed`` drive the N-hit/M-miss
    lifecycle.
    """

    id: int
    x: float
    y: float
    vx: float
    vy: float
    p00: float
    p01: float
    p11: float
    radius: float
    hits: int
    misses: int
    confirmed: bool


def _kf_predict(track: _KTrack, dt: float) -> None:
    """Time-advance one track's decoupled CV Kalman filter in place.

    Per axis: ``s <- F s`` with ``F = [[1, dt], [0, 1]]`` (position advances by
    velocity, velocity persists), and the shared 2x2 covariance advances by
    ``P <- F P F^T + Q(q, dt)`` with the continuous constant-velocity process
    noise ``Q = [[q*dt^3/3, q*dt^2/2], [q*dt^2/2, q*dt]]``.  Scalar mul/add
    only, in a fixed expression order — no matrices, no linear-algebra library
    call (AC3).
    """
    track.x += track.vx * dt
    track.y += track.vy * dt

    q = KF_PROCESS_NOISE
    p00 = track.p00
    p01 = track.p01
    p11 = track.p11
    track.p00 = p00 + dt * (2.0 * p01 + dt * p11) + q * dt * dt * dt / 3.0
    track.p01 = p01 + dt * p11 + q * dt * dt / 2.0
    track.p11 = p11 + q * dt


def _kf_update(track: _KTrack, measured_x: float, measured_y: float) -> None:
    """Fold one centroid measurement into a track's Kalman state in place.

    The measurement is position-only (``H = [1, 0]``) with scalar variance
    ``r = KF_MEASUREMENT_NOISE``, so the innovation variance is the SCALAR
    ``Sm = P00 + r`` and the shared gain is the 2-vector
    ``K = [P00/Sm, P01/Sm]``.  Both axes apply the same gain to their own
    position innovation; the covariance update ``P <- (I - K H) P`` reduces to
    three scalar assignments (symmetry is preserved exactly:
    ``p01 - K1*p00 == (1 - K0)*p01`` algebraically).  No matrix inverse
    anywhere (AC3).
    """
    innovation_variance = track.p00 + KF_MEASUREMENT_NOISE
    gain_position = track.p00 / innovation_variance
    gain_velocity = track.p01 / innovation_variance

    innovation_x = measured_x - track.x
    innovation_y = measured_y - track.y
    track.x += gain_position * innovation_x
    track.vx += gain_velocity * innovation_x
    track.y += gain_position * innovation_y
    track.vy += gain_velocity * innovation_y

    p00 = track.p00
    p01 = track.p01
    p11 = track.p11
    track.p00 = (1.0 - gain_position) * p00
    track.p01 = (1.0 - gain_position) * p01
    track.p11 = p11 - gain_velocity * p01


class LidarTracker:
    """Deterministic CV-KF multi-object tracker — the lidar-only ``Tracker``.

    Each tick the DETECTOR projects the live lidar frame to world points, drops
    returns that land on the STATIC inflated grid (walls / pillars and their
    inflation band), and clusters the surviving dynamic returns by grid bucket
    into per-frame detections.  The ESTIMATOR then carries persistent tracks
    across frames:

    1. Every track's constant-velocity Kalman filter is time-advanced by ``dt``
       (state AND covariance), so gating happens against PREDICTED positions,
       not last-seen ones.
    2. Detections associate to tracks by a prediction-gated, globally-sorted
       greedy assignment: every (track, detection) pair within
       ``ASSOCIATION_GATE_DISTANCE`` of the track's predicted position forms a
       ``(distance, track_id, detection_rank)`` triple; one global sort, then a
       greedy sweep consumes each track and each detection at most once.  The
       FULL triple is the sort key — a total order over the two int fields —
       because symmetric arena geometry produces exactly-equal float distances
       and a bare distance sort would leak nondeterministic tie order.
    3. N-hit/M-miss lifecycle: an unassociated detection births a TENTATIVE
       track, withheld from output; ``CONFIRM_HITS`` consecutive gated hits
       (counting the birth) promote it to CONFIRMED, the only state ``update``
       emits.  A tentative track dies on its first miss.  A confirmed track
       with no association COASTS — it emits its PREDICTED position (never a
       frozen one) through ``COAST_MISSES`` consecutive misses, then deletes.
       Coast-through-merge falls out of the tight gate: a merged blob's
       centroid fails both parents' gates, so both parents coast under their
       pre-merge birth-counter ids instead of being yanked or reborn.
    4. Radius is a capped EMA seeded at birth from the floored measured radius,
       with a merge-suspect gate that skips the radius update (position still
       updates) when the measured radius jumps past
       ``RADIUS_MERGE_SUSPECT_RATIO`` times the filtered one.

    Track ids are a monotonic per-episode birth counter, reset on CONSTRUCTION
    only — each episode builds a fresh tracker via ``_make_tracker``, so the
    per-episode reset is automatic; there is deliberately no per-``update``
    reset (mirrors the irsim ``id_iter`` reset-on-make convention).

    Determinism is the whole contract (the plan's "landmine"): the pipeline
    never iterates a ``set`` to build output or to order operations, draws no
    RNG, feeds every reduction a fixed sorted-order sequence, and breaks
    association ties by the total-order sort key above, so two ``update``
    sequences on byte-identical inputs return byte-identical ``Track`` lists.

    The first ``update`` on a fresh tracker births only tentative tracks, so it
    returns ``[]`` (cold start); an obstacle is first emitted on the frame its
    track is promoted to confirmed.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        bearings: np.ndarray,
        range_max: float = float("inf"),
    ) -> None:
        """Store the static grid, the per-beam bearings, and the no-hit range.

        Parameters
        ----------
        grid:
            The STATIC inflated occupancy grid.  Read-only here: ``grid.cells``
            is consulted to subtract static returns and is never mutated.
        range_max:
            The lidar's max range (the no-hit sentinel).  Returns within
            ``RANGE_MAX_DEADBAND`` of it are dropped as no-hits before clustering
            (see ``_lidar_to_world_points``).  Defaults to ``inf``, which disables
            the cut so a tracker built without a range stays byte-compatible.
        bearings:
            The per-beam angles, already recovered by the caller as
            ``np.linspace(angle_min, angle_max, number)`` (the exact recovery the
            rest of the harness uses — NOT ``i * angle_increment``).  No YAML is
            loaded and no bearings are recomputed here.
        """
        if grid is None:
            raise ValueError("LidarTracker requires a non-None OccupancyGrid.")

        bearings_array = np.asarray(bearings, dtype=float)
        if bearings_array.ndim != 1:
            raise ValueError(
                "bearings must be a 1-D array of beam angles, received shape "
                f"{bearings_array.shape}."
            )

        # Guard the Kalman update against a nonsensical tuning edit: the scalar
        # innovation variance Sm = P00 + r is the gain's divisor, and r > 0
        # keeps it strictly positive with no per-update division guard.
        if not KF_MEASUREMENT_NOISE > 0.0:
            raise ValueError(
                "KF_MEASUREMENT_NOISE must be positive, received "
                f"{KF_MEASUREMENT_NOISE!r}."
            )

        self._grid = grid
        self._bearings = bearings_array
        self._range_max = float(range_max)
        # Persistent track list, maintained in id-ascending (birth) order.
        self._tracks: list[_KTrack] = []
        # Monotonic per-episode birth counter for track ids.  Initialized HERE,
        # on construction, and never reset per update: each episode builds a
        # fresh tracker, so ids restart per episode automatically.
        self._next_track_id: int = 0

    def update(
        self,
        *,
        snapshot: object,
        state: np.ndarray,
        lidar: np.ndarray,
        dt: float,
    ) -> list[Track]:
        """Advance the tracker one frame; return CONFIRMED Tracks sorted by ``id``.

        Parameters
        ----------
        snapshot:
            Ignored — this tracker is lidar-only (accepted for protocol parity).
        state:
            ``(3,)`` ``[x, y, theta]`` robot pose in the world frame.
        lidar:
            ``(number,)`` range scan (NaN = no return), one entry per bearing.
            An empty / all-NaN frame is valid: every confirmed track takes a
            miss (coasting up to ``COAST_MISSES``), tentatives die, no
            exception.
        dt:
            Frame interval in seconds (the caller passes ``PREDICT_DT``); the
            Kalman prediction timestep.  Must be positive.
        """
        del snapshot  # lidar-only: the truth snapshot is ignored.

        state_array = np.asarray(state, dtype=float)
        if state_array.shape != (3,):
            raise ValueError(
                f"Expected (3,) [x, y, theta] state, received shape "
                f"{state_array.shape}."
            )

        lidar_array = np.asarray(lidar, dtype=float)
        expected_lidar_shape = (self._bearings.shape[0],)
        if lidar_array.shape != expected_lidar_shape:
            raise ValueError(
                f"Expected lidar of shape {expected_lidar_shape}, received "
                f"{lidar_array.shape}."
            )

        if not dt > 0.0:
            raise ValueError(f"dt must be positive, received {dt!r}.")

        world_points = self._lidar_to_world_points(state_array, lidar_array)
        dynamic_points = self._drop_static_returns(world_points)
        clusters = self._cluster(dynamic_points)
        return self._associate_and_build(clusters, float(dt))

    # --- Internal helpers ---------------------------------------------------

    def _lidar_to_world_points(
        self, state: np.ndarray, lidar: np.ndarray
    ) -> np.ndarray:
        """Project finite lidar returns to world-frame obstacle points.

        Delegates to the shared :func:`planners._geometry.lidar_to_world_points`
        (the one projection the family uses — formerly copy-pasted here, in DWA, and
        inline in the occupancy fold). Passing ``self._range_max`` enables the
        near-rim no-hit deadband: a beam coming back AT range_max with float jitter
        survives the Arena's ``>= range_max`` filter and would otherwise cluster into
        a phantom rim obstacle, so a RANGE_MAX_DEADBAND cut below range_max drops it
        (``range_max == inf`` keeps every finite beam). Returns an ``(N, 2)`` array,
        possibly empty, with the surviving points in beam order.
        """
        return lidar_to_world_points(
            state, lidar, self._bearings, range_max=self._range_max
        )

    def _drop_static_returns(self, points: np.ndarray) -> np.ndarray:
        """Drop every hit point whose grid cell is occupied in the static grid.

        Vectorized for speed but byte-equivalent to per-point
        ``world_to_grid`` + ``grid.cells[cell]`` (same ``np.floor`` / ``np.clip``
        clamp, same offset and resolution): a wall or pillar return lands on a
        ``True`` cell (the obstacle or its inflation band) and is removed,
        leaving only dynamic-obstacle returns.  Input beam ORDER is preserved.
        """
        if points.shape[0] == 0:
            return points

        cells = self._grid.cells
        rows, cols = cells.shape
        resolution = self._grid.resolution
        offset_x = float(self._grid.offset[0])
        offset_y = float(self._grid.offset[1])

        raw_col = (points[:, 0] - offset_x) / resolution
        raw_row = (points[:, 1] - offset_y) / resolution
        grid_col = np.clip(np.floor(raw_col), 0, cols - 1).astype(int)
        grid_row = np.clip(np.floor(raw_row), 0, rows - 1).astype(int)

        occupied = cells[grid_row, grid_col]
        return points[~occupied]

    def _cluster(self, points: np.ndarray) -> list[_Cluster]:
        """Deterministically cluster dynamic points into obstacle clusters.

        Buckets each point into an integer ``(row, col)`` cell at
        ``CLUSTER_RESOLUTION`` (row = y, col = x to match the grid convention),
        runs 8-connected connected components over the occupied buckets seeded in
        sorted bucket-key order, and reduces each component to a ``_Cluster``.
        Returns the clusters in component-seed order; the caller re-sorts by
        ``rep_cell`` before association.
        """
        if points.shape[0] == 0:
            return []

        # 1. Bucket points, preserving input (beam) order within each bucket.
        buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for index in range(points.shape[0]):
            point_x = float(points[index, 0])
            point_y = float(points[index, 1])
            bucket_key = (
                int(math.floor(point_y / CLUSTER_RESOLUTION)),
                int(math.floor(point_x / CLUSTER_RESOLUTION)),
            )
            buckets.setdefault(bucket_key, []).append((point_x, point_y))

        # 2. Connected components (8-connectivity) over the occupied buckets,
        #    seeding components in sorted bucket-key order so component output
        #    order is deterministic.  Flood-fill internal order is irrelevant
        #    because each component's members are re-sorted in _build_cluster.
        occupied_keys = set(buckets.keys())
        visited: set[tuple[int, int]] = set()
        clusters: list[_Cluster] = []
        for seed_key in sorted(occupied_keys):
            if seed_key in visited:
                continue
            component_keys: list[tuple[int, int]] = []
            stack = [seed_key]
            visited.add(seed_key)
            while stack:
                current = stack.pop()
                component_keys.append(current)
                current_row, current_col = current
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        neighbor = (current_row + delta_row, current_col + delta_col)
                        if neighbor in occupied_keys and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
            clusters.append(self._build_cluster(component_keys, buckets))

        return clusters

    def _build_cluster(
        self,
        component_keys: list[tuple[int, int]],
        buckets: dict[tuple[int, int], list[tuple[float, float]]],
    ) -> _Cluster:
        """Reduce one connected component to a centroid, radius, and rep cell.

        Member points are gathered and sorted by ``(bucket_key, x, y)`` into a
        fixed order, so the centroid ``np.mean`` reduces an identically-ordered
        array every run.  The radius is the max centroid-to-member distance,
        floored at ``MIN_TRACK_RADIUS``.  The representative cell is the
        smallest bucket key — the deterministic detection-order key.
        """
        member_points: list[tuple[tuple[int, int], float, float]] = []
        for bucket_key in component_keys:
            for point_x, point_y in buckets[bucket_key]:
                member_points.append((bucket_key, point_x, point_y))
        # Lexicographic sort by (bucket_key, x, y) — a fixed, reproducible order.
        member_points.sort()

        points_array = np.array(
            [[point_x, point_y] for (_, point_x, point_y) in member_points],
            dtype=float,
        )
        centroid = np.mean(points_array, axis=0)
        centroid_x = float(centroid[0])
        centroid_y = float(centroid[1])

        deltas = points_array - centroid
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        radius = max(float(distances.max()), MIN_TRACK_RADIUS)

        rep_cell = min(component_keys)
        return _Cluster(
            rep_cell=rep_cell,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            radius=radius,
        )

    def _associate_and_build(
        self, clusters: list[_Cluster], dt: float
    ) -> list[Track]:
        """Run one predict/associate/lifecycle cycle; return confirmed Tracks.

        The frame's clusters become detections in ``rep_cell``-ascending order
        (``detection_rank`` = index in that order).  Every track is first
        time-advanced by ``dt`` (KF predict), so the association gate and a
        coasting track's emitted position are both the PREDICTED state.
        Association is globally-sorted greedy over the gate-passing
        ``(distance, track_id, detection_rank)`` triples; lifecycle, birth, and
        radius rules follow the module constants above.  Output is confirmed
        tracks only, sorted by ``id`` ascending.
        """
        detections = sorted(clusters, key=lambda cluster: cluster.rep_cell)

        # 1. KF time-advance every track (state AND covariance) exactly once
        #    per frame.  After this, (track.x, track.y) IS the predicted
        #    position: the association gate reads it, an associated track's
        #    measurement update refines it, and a coasting track emits it
        #    unchanged — never a frozen last-seen position.
        for track in self._tracks:
            _kf_predict(track, dt)

        # 2. Prediction-gated, globally-sorted greedy association.  Build every
        #    gate-passing (distance, track_id, detection_rank) triple, sort
        #    ONCE, consume greedily (each track and each detection at most
        #    once).  THE determinism landmine: the sort key is the FULL triple
        #    — symmetric arena geometry produces exactly-equal float distances,
        #    and the two int fields make the order total, so ties can never
        #    resolve nondeterministically.
        candidate_pairs: list[tuple[float, int, int]] = []
        for track in self._tracks:
            for rank, detection in enumerate(detections):
                delta_x = detection.centroid_x - track.x
                delta_y = detection.centroid_y - track.y
                distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                if distance <= ASSOCIATION_GATE_DISTANCE:
                    candidate_pairs.append((distance, track.id, rank))
        candidate_pairs.sort()

        assignment: dict[int, int] = {}  # track id -> detection rank
        used_ranks: set[int] = set()     # membership checks only, never iterated
        for _distance, track_id, rank in candidate_pairs:
            if track_id in assignment or rank in used_ranks:
                continue
            assignment[track_id] = rank
            used_ranks.add(rank)

        # 3. Lifecycle sweep over the existing tracks, in id (list) order.
        survivors: list[_KTrack] = []
        for track in self._tracks:
            rank = assignment.get(track.id)
            if rank is not None:
                # Gated hit: fold the measurement in, then advance the
                # lifecycle.  CONFIRM_HITS consecutive gated hits (counting the
                # birth detection) promote a tentative track; the promotion
                # frame is also its first emitted frame.
                detection = detections[rank]
                _kf_update(track, detection.centroid_x, detection.centroid_y)
                track.hits += 1
                track.misses = 0
                if not track.confirmed and track.hits >= CONFIRM_HITS:
                    track.confirmed = True
                # Capped-EMA radius with the merge-suspect gate: a measured
                # radius jumping past the ratio marks a probable merged blob,
                # so the radius update is SKIPPED this frame (the position
                # update above still happened).  The birth frame never reaches
                # here — births happen in step 4 — so the gate structurally
                # cannot apply to the birth seed.
                if detection.radius <= RADIUS_MERGE_SUSPECT_RATIO * track.radius:
                    blended = (
                        RADIUS_EMA_BETA * detection.radius
                        + (1.0 - RADIUS_EMA_BETA) * track.radius
                    )
                    track.radius = min(max(blended, MIN_TRACK_RADIUS), RADIUS_MAX)
                survivors.append(track)
            elif track.confirmed:
                # Confirmed miss: COAST on the prediction (already advanced in
                # step 1).  The track stays emitted through misses
                # 1..COAST_MISSES (its radius untouched), then deletes on the
                # next consecutive miss — <= 0.3 s of coasting at PREDICT_DT.
                track.misses += 1
                if track.misses <= COAST_MISSES:
                    survivors.append(track)
            # else: a TENTATIVE track dies on its first miss (dropped here).

        # 4. Births: every unassociated detection starts a TENTATIVE track,
        #    withheld from output until confirmed.  Iterated in rank order so
        #    birth ids assign deterministically; appended after the survivors,
        #    which keeps the track list id-ascending (new ids are the largest).
        for rank, detection in enumerate(detections):
            if rank in used_ranks:
                continue
            survivors.append(
                _KTrack(
                    id=self._next_track_id,
                    x=detection.centroid_x,
                    y=detection.centroid_y,
                    vx=0.0,
                    vy=0.0,
                    p00=KF_INITIAL_POSITION_VARIANCE,
                    p01=0.0,
                    p11=KF_INITIAL_VELOCITY_VARIANCE,
                    # The birth seed is the floored measured radius — the
                    # merge-suspect gate does not apply (no prior filtered
                    # radius), but the hard cap does: AC6 bounds every reported
                    # radius, including a first-seen merged blob's.
                    radius=min(max(detection.radius, MIN_TRACK_RADIUS), RADIUS_MAX),
                    hits=1,
                    misses=0,
                    confirmed=False,
                )
            )
            self._next_track_id += 1

        self._tracks = survivors

        # 5. Emit CONFIRMED tracks only.  The track list is id-ascending by
        #    construction; the sort restates the output contract.
        confirmed_tracks = [
            Track(
                id=track.id,
                x=track.x,
                y=track.y,
                vx=track.vx,
                vy=track.vy,
                radius=track.radius,
            )
            for track in self._tracks
            if track.confirmed
        ]
        confirmed_tracks.sort(key=lambda track: track.id)
        return confirmed_tracks


def predict_blocked_cells(
    tracks: list[Track],
    planned_path: list[np.ndarray],     # ordered (2,) world-frame waypoints (the robot's current committed path)
    robot_xy: np.ndarray,               # (2,) current robot position
    grid: OccupancyGrid,
    inflation: float,                   # = robot_radius + SAFETY_MARGIN (body-aware band, same as the static grid)
    horizon_steps: int,
    dt: float,
    *,
    geometry: str,                      # "capsule" | "cone"
    exclusion_radius: float,
    corridor_half_width: float,
) -> list[tuple[ThreatKey, list[tuple[int, int]]]]:
    """Predict the grid cells each tracked obstacle will threaten over the horizon.

    For every track, the future footprint is the union over lookahead steps
    ``k = 1..horizon_steps`` of the disk centered at
    ``(x + vx*k*dt, y + vy*k*dt)`` with radius ``r_k``:

    - ``geometry == "capsule"``: ``r_k = track.radius + inflation`` (constant — a
      straight disk train along the velocity vector).
    - ``geometry == "cone"``: ``r_k = track.radius + inflation +
      CONE_GROWTH_PER_STEP * k`` (radius grows linearly with the lookahead step,
      widening to represent estimator uncertainty).

    A track is kept only if its predicted disk geometrically intersects a
    corridor of half-width ``corridor_half_width`` around ``planned_path`` for
    SOME ``k`` (the predicted-conflict gate); its time-to-conflict is the
    smallest such ``k``.  Cells whose center lies within ``exclusion_radius`` of
    ``robot_xy`` are removed from every track's set (the robot exclusion zone).

    Returns one ``(ThreatKey(ttc_steps, track.id), sorted_unique_cells)`` group
    per gated track, the list sorted by ``ThreatKey`` ascending (soonest
    conflict first, then by id), each cell list sorted row-major ascending.

    The function is PURE and deterministic: plain floats/ints + numpy in,
    deterministically-sorted cells out.  No RNG, no irsim calls, no set-iteration
    leaking into the output order.  Two calls on identical inputs return
    byte-identical output.
    """
    if geometry not in {"capsule", "cone"}:
        raise ValueError(
            f"geometry must be 'capsule' or 'cone', received {geometry!r}."
        )

    # h0 / empty guards.  horizon_steps == 0 yields no centers and is the true
    # no-op baseline (AC2/TC57); an empty track list or an empty path likewise
    # produces nothing to stamp.
    if horizon_steps <= 0:
        return []
    if not tracks:
        return []
    if len(planned_path) < 1:
        return []

    # The conflict gate's polyline distance comes from manual_astar's canonical
    # helper, lazy-imported here (not at module top) so this module's top-level
    # imports stay free of the manual_astar coupling.
    from manual_astar import point_to_polyline_distance

    robot_x = float(robot_xy[0])
    robot_y = float(robot_xy[1])
    exclusion_radius_sq = exclusion_radius * exclusion_radius

    groups: list[tuple[ThreatKey, list[tuple[int, int]]]] = []

    for track in tracks:
        base_radius = track.radius + inflation
        ttc_steps: int | None = None
        cells: set[tuple[int, int]] = set()

        for k in range(1, horizon_steps + 1):
            center_x = track.x + track.vx * k * dt
            center_y = track.y + track.vy * k * dt

            if geometry == "capsule":
                r_k = base_radius
            else:  # "cone": radius grows linearly with the lookahead step.
                r_k = base_radius + CONE_GROWTH_PER_STEP * k

            # Predicted-conflict gate: does this disk reach the path corridor?
            # Reuses manual_astar's canonical point_to_polyline_distance (open
            # polyline) so the gate measures distance the SAME way the rest of the
            # harness does, with no drifting private copy.
            distance = point_to_polyline_distance(
                np.array([center_x, center_y], dtype=float), planned_path, closed=False
            )
            if distance <= r_k + corridor_half_width:
                if ttc_steps is None:
                    ttc_steps = k
                # Collect the full disk footprint via the shared _geometry scan —
                # the SAME bounding-box row-major scan _grid._mark_disk fills the
                # lidar fold from — so the predicted stamp can never drift from the
                # fold's geometry.
                cells.update(iter_disk_cells(grid, center_x, center_y, r_k))

        # Drop tracks whose footprint never reaches the corridor.
        if ttc_steps is None:
            continue

        # Robot exclusion zone: never stamp a cell whose center is within
        # exclusion_radius of the robot (AC5).
        if exclusion_radius > 0.0 and cells:
            resolution = grid.resolution
            offset_x = float(grid.offset[0])
            offset_y = float(grid.offset[1])
            kept: set[tuple[int, int]] = set()
            for (row, col) in cells:
                cell_center_x = offset_x + (col + 0.5) * resolution
                cell_center_y = offset_y + (row + 0.5) * resolution
                dx = cell_center_x - robot_x
                dy = cell_center_y - robot_y
                if dx * dx + dy * dy > exclusion_radius_sq:
                    kept.add((row, col))
            cells = kept

        # A track whose every cell fell inside the exclusion zone contributes
        # nothing; skip the empty group so the output carries only real stamps.
        if not cells:
            continue

        sorted_cells = sorted(cells)  # row-major ascending (row, then col)
        groups.append((ThreatKey(ttc_steps, track.id), sorted_cells))

    groups.sort(key=lambda group: group[0])
    return groups


@dataclass(frozen=True)
class TrajectoryConflict:
    """Result of a space-time robot-trajectory-vs-tracks conflict check.

    ``collides`` is True when the robot body overlaps some track's body at a
    MATCHED time step within the horizon; ``ttc_step`` is the earliest such step
    (1-based, None when no collision); ``min_gap`` is the minimum matched-time
    body gap (``center_dist - robot_radius - track_radius``) over every checked
    (step, track) pair (``+inf`` when there are no tracks / no steps to check).
    A surviving (non-colliding) candidate has ``min_gap > margin``, so the caller
    can turn ``min_gap`` into a "predicted clearance" score term.
    """

    collides: bool
    ttc_step: int | None
    min_gap: float


def trajectory_conflict(
    robot_positions: np.ndarray,   # (S, 2) robot world positions at steps k = 1..S (step k is dt*k ahead)
    tracks: list[Track],
    robot_radius: float,
    horizon_steps: int,
    dt: float,
    margin: float,
) -> TrajectoryConflict:
    """Space-time collision check: robot(k) vs each track's constant-velocity pose(k).

    For each lookahead step ``k = 1..min(horizon_steps, S)`` the robot is at
    ``robot_positions[k-1]`` and each track is at
    ``(track.x + track.vx*k*dt, track.y + track.vy*k*dt)`` — the SAME sim time, so
    this is genuine ``(x, y, t)`` reasoning, not a 2-D footprint stamp. A collision
    is registered when the matched-time body gap
    ``dist - robot_radius - track.radius`` drops to within ``margin``.
    ``ttc_step`` is the earliest colliding step (steps are scanned ascending);
    ``min_gap`` is the minimum gap over every checked pair (kept even past a
    collision so a rejected candidate still has a meaningful negative gap and a
    surviving one a positive clearance).

    PURE and deterministic: plain floats + numpy in, a plain dataclass out; the
    track list is iterated in caller order (no set-iteration), no RNG. Two calls on
    identical inputs return byte-identical results (AC4).
    """
    no_conflict = TrajectoryConflict(collides=False, ttc_step=None, min_gap=float("inf"))
    if horizon_steps <= 0 or not tracks or robot_positions.shape[0] == 0:
        return no_conflict

    steps = min(int(horizon_steps), int(robot_positions.shape[0]))
    min_gap = float("inf")
    ttc_step: int | None = None

    for k in range(1, steps + 1):
        robot_x = float(robot_positions[k - 1, 0])
        robot_y = float(robot_positions[k - 1, 1])
        for track in tracks:
            obstacle_x = track.x + track.vx * k * dt
            obstacle_y = track.y + track.vy * k * dt
            delta_x = robot_x - obstacle_x
            delta_y = robot_y - obstacle_y
            gap = math.sqrt(delta_x * delta_x + delta_y * delta_y) - robot_radius - track.radius
            if gap < min_gap:
                min_gap = gap
            if ttc_step is None and gap <= margin:
                ttc_step = k

    return TrajectoryConflict(
        collides=ttc_step is not None, ttc_step=ttc_step, min_gap=min_gap
    )
