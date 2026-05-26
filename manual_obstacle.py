import irsim
import numpy as np


WORLD_FILE = 'obstacle_harder.yaml'
SIM_STEPS = 1000
RENDER_INTERVAL = 0.05

# Controller tuning knobs.
LIDAR_INFLUENCE_MARGIN = 2.5
FORWARD_DETECTION_HALF_ANGLE = np.pi / 2.1
OBSTACLE_TURN_GAIN = 3.5
ESCAPE_DISTANCE = 0.4
ESCAPE_TURN_RATE = 1.3
SIDE_BIAS_DISTANCE = 0.7
SIDE_BIAS_MIN_TURN = 0.05
SIDE_BIAS_TURN = 0.7
SLOW_DISTANCE = 0.5
CAUTION_DISTANCE = 0.8
SLOW_SPEED = 0.5
CAUTION_SPEED = 0.7
STRAIGHT_TURN_THRESHOLD = 0.5
CRUISE_SPEED = 1.0
TURNING_SPEED = 0.7
RANGE_EPSILON = 1e-6


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_lidar_arrays(robot):
    scan = robot.get_lidar_scan()

    if scan is None:
        return np.array([]), np.array([]), {}

    ranges = np.asarray(scan.get('ranges', scan.get('range_data', [])), dtype=float).reshape(-1)

    if ranges.size == 0:
        return ranges, np.array([]), scan

    angles = scan.get('angles', scan.get('angle_list'))

    if angles is None:
        angle_min = float(scan.get('angle_min', -np.pi / 2))
        angle_max = float(scan.get('angle_max', np.pi / 2))
        angles = np.linspace(angle_min, angle_max, ranges.size)
    else:
        angles = np.asarray(angles, dtype=float).reshape(-1)

        if angles.size != ranges.size:
            angles = np.linspace(angles[0], angles[-1], ranges.size)

    return ranges, angles, scan

def target_to_goal(robot):
    x, y, theta = robot.state[:, 0]
    gx, gy, _ = robot.goal[:, 0]

    dx = gx - x
    dy = gy - y

    target_theta = np.arctan2(dy, dx)
    return wrap_to_pi(target_theta - theta)

# This returns a turn command, not an absolute heading.
def push_away(robot):
    ranges, angles, scan = get_lidar_arrays(robot)

    if ranges.size == 0:
        return 0.0, np.inf, 0.0

    range_min = float(scan.get('range_min', 0.0))
    range_max = float(scan.get('range_max', np.max(ranges)))
    valid_mask = np.isfinite(ranges) & (ranges > range_min + RANGE_EPSILON) & (ranges < range_max - RANGE_EPSILON)

    if not np.any(valid_mask):
        return 0.0, np.inf, 0.0

    influence_margin = min(LIDAR_INFLUENCE_MARGIN, range_max)
    close_mask = valid_mask & (ranges < influence_margin)
    front_mask = valid_mask & (np.abs(angles) < FORWARD_DETECTION_HALF_ANGLE)
    left_mask = front_mask & (angles > 0.0)
    right_mask = front_mask & (angles < 0.0)

    total_turn = 0.0
    if np.any(close_mask):
        close_ranges = ranges[close_mask]
        close_angles = angles[close_mask]
        push_strength = ((influence_margin - close_ranges) / influence_margin) ** 2
        turn_terms = -OBSTACLE_TURN_GAIN * push_strength * np.sin(close_angles)
        total_turn = float(np.mean(turn_terms))

    closest_forward_distance = float(np.min(ranges[front_mask])) if np.any(front_mask) else np.inf

    left_clearance = float(np.mean(ranges[left_mask])) if np.any(left_mask) else range_max
    right_clearance = float(np.mean(ranges[right_mask])) if np.any(right_mask) else range_max
    side_bias = np.clip((left_clearance - right_clearance) / influence_margin, -1.0, 1.0)

    return total_turn, closest_forward_distance, float(side_bias)

def action(robot):
    goal_turn = target_to_goal(robot)
    obstacle_turn, closest_forward_distance, side_bias = push_away(robot)

    if closest_forward_distance < ESCAPE_DISTANCE:
        escape_direction = np.sign(side_bias)

        if escape_direction == 0.0:
            escape_direction = np.sign(obstacle_turn)

        if escape_direction == 0.0:
            escape_direction = -np.sign(goal_turn)

        if escape_direction == 0.0:
            escape_direction = 1.0

        return np.array([[0.0], [ESCAPE_TURN_RATE * escape_direction]], dtype=float)

    if closest_forward_distance < SIDE_BIAS_DISTANCE and abs(obstacle_turn) < SIDE_BIAS_MIN_TURN:
        preferred_direction = np.sign(side_bias)

        if preferred_direction == 0.0:
            preferred_direction = np.sign(goal_turn)

        if preferred_direction == 0.0:
            preferred_direction = 1.0

        obstacle_turn += SIDE_BIAS_TURN * preferred_direction

    w = goal_turn + obstacle_turn
    w = np.clip(w, -1.0, 1.0)

    if closest_forward_distance < SLOW_DISTANCE:
        v = SLOW_SPEED
    elif closest_forward_distance < CAUTION_DISTANCE:
        v = CAUTION_SPEED
    elif abs(w) < STRAIGHT_TURN_THRESHOLD:
        v = CRUISE_SPEED
    else:
        v = TURNING_SPEED

    return np.array([[v], [w]], dtype=float)

if __name__ == '__main__':
    env = irsim.make(WORLD_FILE)
    robot = env.robot_list[0]
    robot.sensor_step()

    # Number of steps
    for _ in range(SIM_STEPS):

        # computes the action for the next step
        next_action = action(robot)

        # performs the action
        env.step([next_action])

        # shows the result
        env.render(RENDER_INTERVAL)

        if env.done():
            break

    env.end()

