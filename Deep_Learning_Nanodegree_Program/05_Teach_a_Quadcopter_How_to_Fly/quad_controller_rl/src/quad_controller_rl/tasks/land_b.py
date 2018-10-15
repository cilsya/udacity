"""Land task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class LandB(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        # max_force = 25.0
        # max_torque = 25.0
        max_force = 19.5
        max_torque = 19.5
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        #self.max_duration = 20.0  # secs
        #self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        # Task-specific parameters
        self.max_duration = 40.0  # secs
        self.max_error_position = 9.8  # distance units
        self.max_error_velocity = 2.5  # velocity units
        self.initial_position = np.array([0.0, 0.0, 10.0]) 
        self.target_position = np.array([0.0, 0.0, 0.0])  # target position to hover at
        self.weight_position = 0.3
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        self.weight_orientation = 0.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 0.7
        self.entered_landing_zone = 0
        self.in_landing_zone_duration = 3.0
        self.in_landing_zone_time_start = 0.0

        self.current_condition = Pose(
                                        position=Point(0.0, 0.0, 0.0),
                                        orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                                    ), Twist(
                                        linear=Vector3(0.0, 0.0, 0.0),
                                        angular=Vector3(0.0, 0.0, 0.0)
                                    )


    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        self.entered_landing_zone = 0

        # # Return initial condition
        # p = self.initial_position + np.random.normal(0.5, 0.1, size=3)  # slight random position around the target
        # return Pose(
        #         position=Point(*p),
        #         orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        #     ), Twist(
        #         linear=Vector3(0.0, 0.0, 0.0),
        #         angular=Vector3(0.0, 0.0, 0.0)
        #     )

        return self.current_condition

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        # state = np.array([
        #         pose.position.x, pose.position.y, pose.position.z,
        #         pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Prepare state vector (pose, orientation, velocity only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
        state = np.concatenate([position, orientation, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        error_position = np.linalg.norm(self.target_position - state[0:3])  # Euclidean distance from target position vector
        #error_position_exponent = np.abs(error_position) **3
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7])  # Euclidean distance from target orientation quaternion (a better comparison may be needed)
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10])  # Euclidean distance from target velocity vector

        # Compute reward / penalty and check if this episode is complete
        done = False
        #reward = -min(abs(self.target_z - pose.position.z), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        #reward = -(self.weight_position * error_position + self.weight_orientation * error_orientation + self.weight_velocity * error_velocity)

        # Get continuous reward as you get closer to the goal
        reward = (1.0 - error_position/self.initial_position[2])**2

        # Get continuous reward for keeping velocity low
        reward += (1.0 - error_velocity/self.max_error_velocity)**2

        goal_position = np.linalg.norm(self.initial_position - state[0:3]) 
        if goal_position > self.max_error_position:
            #reward += 50.0  # extra reward, agent entered landing zone
            #done = True

            # If this is the first time entering the landing zone, mark it.
            if self.entered_landing_zone == 0:

                self.entered_landing_zone = 1
                self.in_landing_zone_time_start = timestamp

                # DEBUG
                #
                print("\n\nDEBUG - Entered landing zone. self.entered_landing_zone: {}\n\n".format(self.entered_landing_zone))

            # # DEBUG
            # #
            # print("\n\nDEBUG - Stopping - in the goal position. goal_position ({}) > self.max_error_position ({}).\n\n".format(goal_position, self.max_error_position))

        elif timestamp > self.max_duration:
            reward -= 100.0  # extra reward, agent made it to the end
            done = True

            # DEBUG
            #
            print("DEBUG - Stopping - time lapsed passed the max duration. timestamp ({})> self.max_duration ({}).\n\n".format(timestamp, self.max_duration))

        elif error_velocity > self.max_error_velocity:
            reward -= 100.0  # extra penalty, agent too high velocity
            done = True

            # DEBUG
            #
            print("\n\nDEBUG - Stopping - velocity is too high. error_velocity ({}) > self.max_error_velocity ({})\n\n".format(error_velocity, self.max_error_velocity))



        # If we landed, and in the landing zone for good amount of time, bonus.
        if self.entered_landing_zone == 1:

            zone_duration = timestamp - self.in_landing_zone_time_start 
            if zone_duration > self.in_landing_zone_duration:

                reward += 500
                done = True

                # DEBUG
                #
                print("\n\nDEBUG - Stopping - Reached our goal. Stayed in landing zone for duration. self.in_landing_zone_duration: {}\n\n".format(self.in_landing_zone_duration))
            

        # elif error_position > self.max_error_position:

        #     if self.entered_landing_zone == 0:
        #         self.in_landing_zone_time_start = timestamp

        #     elif (timestamp - self.in_landing_zone_time_start) > self.in_landing_zone_duration:           

        #         reward += 50.0
        #         done = True


        # if pose.position.z >= self.target_z:  # agent has crossed the target height
        #     reward += 10.0  # bonus reward
        #     done = True
        # elif timestamp > self.max_duration:  # agent has run out of time
        #     reward -= 10.0  # extra penalty
        #     done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
