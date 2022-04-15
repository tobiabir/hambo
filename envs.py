import gym
import numpy as np
import torch


class EnvPoint(gym.core.Env):

    def __init__(self):
        super().__init__()
        self._episode_steps = 0
        self._max_episode_steps = 100

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

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def done(self, obs):
        if self._max_episode_steps <= self._episode_steps:
            return True
        elif abs(obs[0]) < 0.01 and abs(obs[1]) < 0.01:
            return True
        else:
            return False

    def reward(self, obs, act, obs_next):
        return - np.sqrt(obs_next[0]**2 + obs_next[1]**2)


class EnvModel(gym.core.Env):

    def __init__(self, space_observation, space_action, dataset_states_initial, model_reward, model_transition):
        self.space_observation = space_observation
        self.space_action = space_action
        self.dataset_states_initial = dataset_states_initial
        self.model_reward = model_reward
        self.model_transition = model_transition
        self._max_episode_steps = 100
        self.steps = 0

    def step(self, action):
        with torch.no_grad():
            state_action = np.concatenate((self.state, action), axis=-1)
            state_action = torch.tensor(
                state_action, dtype=torch.float32).unsqueeze(dim=0)
            state_next = self.model_transition(state_action)[0].numpy()[0]
            reward = self.model_reward(self.state, action, state_next)
            self.state = state_next
            self.steps += 1
            done = self.done(self.state)
            return self.state, reward, done, {}

    def done(self, obs):
        return self._max_episode_steps <= self.steps

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.dataset_states_initial.sample(1)[0].numpy()
        self.steps = 0
        return self.state  # TODO: attention! maybe this should be copied

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
            _, mean, var = self.model_transition(state_action)
            mean = mean.numpy()[0]
            var = var.numpy()[0]
            state_next = mean + self.beta * var * action_hallucinated
            reward = self.model_reward(self.state, action, state_next)
            self.state = state_next
            self.steps += 1
            done = self.done(self.state)
            return self.state, reward, done, {}

    @property
    def observation_space(self):
        return self.space_observation

    @property
    def action_space(self):
        low = np.concatenate((self.space_action.low, self.space_action_hallucinated.low))
        high = np.concatenate((self.space_action.high, self.space_action_hallucinated.high))
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
