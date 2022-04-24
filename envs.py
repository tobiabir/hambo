import gym
import numpy as np
import torch

import utils

class EnvPoint(gym.core.Env):

    def __init__(self):
        super().__init__()
        self._episode_steps = 0
        self.max_steps_episode = 100

    @property
    def action_space(self):
        return gym.spaces.Box(low=-0.1, high=0.1, shape=(2,))

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment
        Returns:
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        prev_state = self._state
        self._state = prev_state + np.clip(action, -0.1, 0.1)
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = np.copy(self._state)
        self._episode_steps += 1
        return next_observation, reward, done, {}

    def reset(self, seed=None):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        super().reset(seed=seed)
        self._state = np.random.uniform(-2, 2, size=(2,))
        observation = np.copy(self._state)
        self._episode_steps = 0
        return observation

    def done(self, obs):
        if self.max_steps_episode <= self._episode_steps:
            return True
        elif abs(obs[0]) < 0.01 and abs(obs[1]) < 0.01:
            return True
        else:
            return False

    def reward(self, obs, act, obs_next):
        return - np.sqrt(obs_next[0]**2 + obs_next[1]**2)

class WrapperEnvMountainCar(gym.core.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.max_steps_episode = 999

    def reward(self, state, action, state_next):
        position = state_next[0]
        velocity = state_next[1]
        done = bool(position >= self.unwrapped.goal_position and velocity >= self.unwrapped.goal_velocity)
        reward = 0
        if done:
            reward = 100.0
        reward -= 0.1 * action[0]**2
        return reward

class WrapperEnvPendulum(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_steps_episode = 200

    def reward(self, state, action, state_next):
        th, thdot = self.env.state
        action = np.clip(action, -self.env.max_torque, self.env.max_torque)[0]
        th_normalized = ((th + np.pi) % (2 * np.pi)) - np.pi
        costs = th_normalized**2 + 0.1 * thdot**2 + 0.001 * action**2
        return -costs

class WrapperEnvHalfCheetah(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_steps_episode = 1000

    def reward(self, state, action, state_next):
        pos_x = state[0]
        pos_x_next = state_next[0]
        velocity_x = (pos_x_next - pos_x) / self.env.dt
        reward_forward = self.unwrapped._forward_reward_weight * velocity_x
        cost_ctrl = self.unwrapped._ctrl_cost_weight * np.sum(np.square(action))
        reward = reward_forward - cost_ctrl
        return reward

class EnvModel(gym.core.Env):

    def __init__(self, space_observation, space_action, dataset_states_initial, model_reward, model_transition):
        self.space_observation = space_observation
        self.space_action = space_action
        self.dataset_states_initial = dataset_states_initial
        self.model_reward = model_reward
        self.model_transition = model_transition
        self.max_steps_episode = 4
        self.steps = 0

    def step(self, action):
        with torch.no_grad():
            state_action = np.concatenate((self.state, action), axis=-1)
            state_action = torch.tensor(
                state_action, dtype=torch.float32).unsqueeze(dim=0)
            state_next_mean, state_next_std = self.model_transition.get_distr(state_action)
            state_next = torch.distributions.Normal(state_next_mean, state_next_std).sample().numpy()
            state_next = np.clip(state_next, self.space_observation.low, self.space_observation.high)
            reward = self.model_reward(self.state, action, state_next)
            self.state = state_next
            self.steps += 1
            done = self.done(self.state)
            return self.state.copy(), reward, done, {}

    def done(self, obs):
        return self.max_steps_episode <= self.steps

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.dataset_states_initial.sample(1)[0].numpy()
        self.steps = 0
        return self.state.copy()

    @property
    def observation_space(self):
        return self.space_observation

    @property
    def action_space(self):
        return self.space_action


class EnvModelHallucinated(EnvModel):

    def __init__(self, space_observation, space_action, dataset_states_initial, model_reward, model_transition, beta=1.):
        super().__init__(space_observation, space_action, dataset_states_initial, model_reward, model_transition)
        self.space_action_hallucinated = gym.spaces.Box(
            low=-1, high=1, shape=space_observation.shape, dtype=np.float32)
        self.beta = beta

    def step(self, action):
        with torch.no_grad():
            dim_action = self.space_action.shape[0]
            action_hallucinated = action[dim_action:]
            action = action[:dim_action]
            state_action = np.concatenate((self.state, action), axis=-1)
            state_action = torch.tensor(
                state_action, dtype=torch.float32).unsqueeze(dim=0)
            state_next_mean, state_next_std = self.model_transition.get_distr(state_action)
            state_next_mean = state_next_mean.numpy()[0]
            state_next_var = (state_next_std**2).numpy()[0]
            state_next = state_next_mean + self.beta * state_next_var * action_hallucinated
            reward = self.model_reward(self.state, action, state_next)
            self.state = state_next
            self.steps += 1
            done = self.done(self.state)
            return self.state.copy(), reward, done, {}

    @property
    def action_space(self):
        low = np.concatenate((self.space_action.low, self.space_action_hallucinated.low))
        high = np.concatenate((self.space_action.high, self.space_action_hallucinated.high))
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
