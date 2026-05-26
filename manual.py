import irsim
import numpy as np

# This function takes as input the current state of the robot and the 
# location of the goal and it outputs the velocity and the angle 
# for the next step.
def next_step(robot):

    # Get the current x, y coordinates and the angle theta of the robot
    x, y, theta = robot.state[:, 0]

    # Get the current x, y coordinates of the goal
    gx, gy, _ = robot.goal[:, 0]

    # Adjust the angle of the robot based on its position and 
    # the position of the goal
    dx = gx - x
    dy = gy - y

    target_theta = np.arctan2(dy, dx)
    angle_error = target_theta - theta
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

    # We are setting velocity to be 0.5
    v = [0.5]
    w = [angle_error]

    action = np.array([v, w], dtype=float)
    return action


env = irsim.make('obstacle.yaml')
robot = env.robot_list[0]

# Number of steps
for _ in range(1000):

    # computes the action for the next step
    action = next_step(robot)

    # performs the action
    env.step([action])

    # shows the result
    env.render(0.05)

    if env.done():
        break

env.end()

