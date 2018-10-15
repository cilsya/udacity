"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

from quad_controller_rl.tasks import takeoff_b
from quad_controller_rl.tasks import hover_b
from quad_controller_rl.tasks import land_b

class Combined(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        self.task_takeoff = takeoff_b.TakeoffB()
        self.task_hover = hover_b.HoverB()
        self.task_land = land_b.LandB()

        # Set the current task
        self.o_current_task = self.task_takeoff

        # Current mode.
        # 0 - Takeoff
        # 1 - Hover
        # 2 - Land
        self.mode = 0

        self.current_condition = Pose(
                                        position=Point(0.0, 0.0, 0.0),
                                        orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                                    ), Twist(
                                        linear=Vector3(0.0, 0.0, 0.0),
                                        angular=Vector3(0.0, 0.0, 0.0)
                                    )

    def set_agent(self, agent):
        """Set an agent to carry out this task; to be called from update."""
        #self.agent = agent
        self.task_takeoff.set_agent(agent)
        self.task_hover.set_agent(agent)
        self.task_land.set_agent(agent)

    def reset(self):
        return self.o_current_task.reset()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):

        # Save current condition
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.current_condition = Pose(
                    position=Point(*position),
                    orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                ), Twist(
                    linear=linear_acceleration,
                    angular=angular_velocity
                )

        # Update the tasks current condition.
        self.o_current_task.current_condition = self.current_condition

        # Update the current task
        result = self.o_current_task.update(timestamp, pose, angular_velocity, linear_acceleration)

        # Get the results, we really care about "done"
        _, done = result

        if done:
            self.mode += 1
            self.mode %= 3

            if self.mode == 0:
                self.o_current_task = self.task_takeoff

            if self.mode == 1:
                self.o_current_task = self.task_hover

            if self.mode == 2:
                self.o_current_task = self.task_land

            done = False
            
        # Update the tasks current condition.
        self.o_current_task.current_condition = self.current_condition

        return result
