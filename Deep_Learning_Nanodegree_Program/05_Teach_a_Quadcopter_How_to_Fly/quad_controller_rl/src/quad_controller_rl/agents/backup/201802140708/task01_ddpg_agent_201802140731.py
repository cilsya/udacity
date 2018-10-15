import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.task01_ddpg_actor import Actor
from quad_controller_rl.agents.task01_ddpg_critic import Critic
from quad_controller_rl.agents.ornstein_uhlenbeck_noise import OUNoise
from quad_controller_rl.agents.replay_buffer import ReplayBuffer

import os
import pandas as pd
from quad_controller_rl import util


class Task01_DDPG(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        #self.state_size = np.prod(self.task.observation_space.shape)
        self.task.observation_space.high = self.task.observation_space.high[0:3]
        self.task.observation_space.low =  self.task.observation_space.low[0:3]
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        #self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low


        #self.state_range = self.state_range[0:3]
        #self.action_range = self.action_range[0:3]


        # Constrain state and action spaces
        self.state_size = 3  # position only
        self.action_size = 3  # force only
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))



        # # Policy parameters
        # self.w = np.random.normal(
        #     size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
        #     scale=(self.action_range / (2 * self.state_size)).reshape(1, -1))  # start producing actions in a decent range


        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        #---------------------------------------
        # Saving data

        self.stats_filename = os.path.join(
            util.get_param('out') +'/task01/',
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))


        #---------------------------------------

        # Actor (Policy) Model
        #self.action_low = self.task.action_space.low
        #self.action_high = self.task.action_space.high
        self.state_range = self.state_range[0:3]
        self.action_range = self.action_range[0:3]
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        
        # Episode variables
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def write_stats(self, stats):
            """Write single episode stats to CSV file."""
            df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
            df_stats.to_csv(self.stats_filename, mode='a', index=False,
                header=not os.path.isfile(self.stats_filename))  # write header first time only

    def step(self, state, reward, done):


        # Reduce state vector
        state = self.preprocess_state(state)

        # Transform state vector
        state = (state - self.task.observation_space.low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector
        
        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)

            self.total_reward += reward
            self.count += 1
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        
        #----------------------
        
        # Learn, if at end of episode
        if done:

            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1

            self.learn(experiences)
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        #return action

        # Return complete action vector
        return self.postprocess_action(action)


    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample()  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def preprocess_state(self, state):
            """Reduce state vector to relevant dimensions."""
            return state[0:3]  # position only

    def postprocess_action(self, action):
            """Return complete action vector."""
            complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
            complete_action[0:3] = action  # linear force only
            return complete_action