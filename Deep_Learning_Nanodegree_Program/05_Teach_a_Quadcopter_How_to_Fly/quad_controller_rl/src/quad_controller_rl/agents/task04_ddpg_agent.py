import numpy as np
from copy import deepcopy as deepcopy
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.ornstein_uhlenbeck_noise import OUNoise
from quad_controller_rl.agents.replay_buffer import ReplayBuffer

from quad_controller_rl.tasks import takeoff_b
from quad_controller_rl.tasks import hover_b
from quad_controller_rl.tasks import land_b

from quad_controller_rl.agents import task01_ddpg_agent_b
from quad_controller_rl.agents import task02_ddpg_agent_b
from quad_controller_rl.agents import task03_ddpg_agent_b

import os
import pandas as pd
from quad_controller_rl import util
from keras import models

class Task04_DDPG(BaseAgent):

    def __init__(self, task):

        #---------------------------------------
        # Saving data

        self.stats_filename = os.path.join(
            util.get_param('out') +'/task04/',
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))

        # task_takeoff = deepcopy(task)
        # task_hover = deepcopy(task)
        # task_land = deepcopy(task)
        self.task = task
        self.task_takeoff = takeoff_b.TakeoffB()
        self.task_hover = hover_b.HoverB()
        self.task_land = land_b.LandB()
        self.o_task01_agent = task01_ddpg_agent_b.Task01_DDPG(self.task_takeoff)
        self.o_task02_agent = task02_ddpg_agent_b.Task02_DDPG(self.task_hover)
        self.o_task03_agent = task03_ddpg_agent_b.Task03_DDPG(self.task_land)

        # Current agent
        self.o_current_agent = self.o_task01_agent

        self.mode = 0
        self.episode_num = 0

        self.total_reward = 0.0
  

    def reset_episode_vars(self):
        #return self.o_current_agent.reset_episode_vars()
        self.total_reward = 0.0

    def write_stats(self, stats):
            """Write single episode stats to CSV file."""
            df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
            df_stats.to_csv(self.stats_filename, mode='a', index=False,
                header=not os.path.isfile(self.stats_filename))  # write header first time only

    def step(self, state, reward, done):

        self.total_reward += reward

        if done:


            

            # DEBUG
            #
            print("\n\nDEBUG - done have been called. self.mode: {}\n\n".format(self.mode))
            
            # Go to the mode
            self.mode += 1
            #
            # Cycle mode back to the beginning
            self.mode %= 3

            if self.mode == 0:
                self.o_current_agent = self.o_task01_agent
                self.task  = self.task_takeoff

            if self.mode == 1:
                self.o_current_agent = self.o_task02_agent
                self.task =self.task_hover

            if self.mode == 2:
                self.o_current_agent = self.o_task03_agent
                self.task = self.task_land

            done = False

            # If we cycle back to take off, we count it as an episode
            # and reset reward.
            if self.mode == 0:
                # Write episode stats
                self.write_stats([self.episode_num, self.total_reward])

                self.reset_episode_vars()
                self.episode_num += 1

        return self.o_current_agent.step(state, reward, done)


    def act(self, states):
        return self.o_current_agent.act(states)

    def learn(self, experiences):
        return self.o_current_agent.learn(experiences)

    def soft_update(self, local_model, target_model):
        return self.o_current_agent.soft_update(local_model, target_model)

    def preprocess_state(self, state):
        return self.o_current_agent.preprocess_state(state)

    def postprocess_action(self, action):
        return self.o_current_agent.postprocess_action(action)

    def pass_member_variables_values(in_agent):
        pass