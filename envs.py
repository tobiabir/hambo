import gym
import numpy as np
import torch

import utils


class EnvPoint(gym.core.Env):

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
        return observation

    def done(self, obs):
        return np.logical_and(np.abs(obs[..., 0]) < 0.01, np.abs(obs[..., 1]) < 0.01)

    def reward(self, obs, act, obs_next):
        return - np.sqrt(obs_next[..., 0]**2 + obs_next[..., 1]**2)


class WrapperEnv(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.state = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if "TimeLimit.truncated" not in info:
            info["TimeLimit.truncated"] = False
        self.state = state
        return state, reward, done, info

    def reset(self, seed=None):
        state = self.env.reset(seed=seed)
        self.state = state
        return state


class WrapperEnvMountainCar(WrapperEnv):

    def done(self, state):
        position = state[..., 0]
        velocity = state[..., 1]
        done = np.logical_and(position >= self.unwrapped.goal_position,
                              velocity >= self.unwrapped.goal_velocity)
        return done


class WrapperEnvPendulum(WrapperEnv):

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False


class WrapperEnvInvertedPendulum(WrapperEnv):

    def done(self, state):
        return np.abs(state[..., 1]) > 0.2


class WrapperEnvSwimmer(WrapperEnv):

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False


class WrapperEnvHopper(WrapperEnv):

    def done(self, state):
        height = state[..., 0]
        angle = state[..., 1]
        not_done = np.isfinite(state).all(axis=-1) * np.abs(state[..., 1:] < 100).all(axis=-1) * (height > 0.7) * (np.abs(angle) < 0.2)
        done = ~not_done
        return done


class WrapperEnvHalfCheetah(WrapperEnv):

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False


class EnvModel(gym.core.Env):

    def __init__(self, space_observation, space_action, dataset_states_initial, model_transition, model_termination, args):
        self.space_observation = space_observation
        self.bound_state_low = torch.tensor(
            space_observation.low, dtype=torch.float32, device=args.device)
        self.bound_state_high = torch.tensor(
            space_observation.high, dtype=torch.float32, device=args.device)
        self.space_action = space_action
        self.dataset_states_initial = dataset_states_initial
        self.model_transition = model_transition
        self.model_termination = model_termination
        self.use_gauss_approx = args.use_gauss_approx
        self.use_aleatoric = args.use_aleatoric
        self.weight_penalty_reward = args.weight_penalty_reward
        self.device = args.device

    def _step(self, state, action):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            x = torch.cat((state, action), dim=-1)
            y_means, y_stds = self.model_transition(x)
            if self.use_gauss_approx:
                idxs_elites = self.model_transition.idxs_elites
                y_means, y_stds = y_means[idxs_elites], y_stds[idxs_elites]
                y_mean, y_std, y_std_epistemic = utils.get_mean_std_of_mixture(y_means, y_stds, epistemic=True)
                if self.use_aleatoric:
                    y_mean, y_std = self.model_transition.scaler_y.inverse_transform(y_mean, y_std)
                    y = torch.distributions.Normal(y_mean, y_std).sample()
                else:
                    y_mean, y_std_epistemic = self.model_transition.scaler_y.inverse_transform(y_mean, y_std_epistemic)
                    y = torch.distributions.Normal(y_mean, y_std_epistemic).sample()
            else:
                size_batch = x.shape[0]
                idxs_idxs_elites = torch.randint(0, self.model_transition.num_elites, (size_batch,), device=self.device)
                idxs_model = self.model_transition.idxs_elites[idxs_idxs_elites]
                idxs_batch = torch.arange(0, size_batch, device=self.device)
                y_mean = y_means[idxs_model, idxs_batch]
                y_std = y_stds[idxs_model, idxs_batch]
                y_mean, y_std = self.model_transition.scaler_y.inverse_transform(y_mean, y_std)
                if self.use_aleatoric:
                    y = torch.distributions.Normal(y_mean, y_std).sample()
                else:
                    y = y_mean
            reward = y[:, :1]
            penalty_reward = torch.amax(torch.linalg.norm(y_stds, dim=2), dim=0).unsqueeze(dim=1)
            reward -= self.weight_penalty_reward * penalty_reward 
            state_next = state + y[:, 1:]
            state_next = torch.clamp(
                state_next, self.bound_state_low, self.bound_state_high)
            reward = reward.squeeze().cpu().numpy()
            state_next = state_next.cpu().numpy()
            done = self.model_termination(state_next)
            return state_next, reward, done, {}

    def step(self, action):
        state = np.expand_dims(self.state, axis=0)
        action = np.expand_dims(action, axis=0)
        state_next, reward, done, info = self._step(state, action)
        state_next = state_next.squeeze()
        reward = reward.squeeze()
        done = done.squeeze()
        return state_next, reward, done, info

    def rollout(self, agent, dataset, num_states_initial, max_length_rollout):
        state = self.dataset_states_initial.sample(num_states_initial)
        for idx_step in range(max_length_rollout):
            action = agent.get_action(state)
            state_next, reward, done, info = self._step(state, action)
            # note: termination model has no time limit (which is added via wrapper)
            mask = done.astype(np.float32)
            batch = list(zip(state, action, reward, state_next, mask))
            dataset.push_batch(batch)
            state = state_next[np.logical_not(done.squeeze())]
            if state.size == 0:
                break

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.dataset_states_initial.sample(1)[0]
        return self.state.copy()

    @property
    def observation_space(self):
        return self.space_observation

    @property
    def action_space(self):
        return self.space_action


class EnvModelHallucinated(EnvModel):

    def __init__(self, space_observation, space_action, dataset_states_initial, model_transition, model_termination, args):
        super().__init__(space_observation, space_action, dataset_states_initial,
                         model_transition, model_termination, args)
        self.space_action_hallucinated = gym.spaces.Box(
            low=-1, high=1, shape=space_observation.shape, dtype=np.float32)
        self.beta = args.beta

    def _step(self, state, action):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            dim_action = self.space_action.shape[0]
            action_hallucinated = action[:, dim_action:]
            action = action[:, :dim_action]
            x = torch.cat((state, action), dim=-1)
            y_mean, y_std, y_std_epistemic = self.model_transition.get_distr(x, epistemic=True)
            reward_mean = y_mean[:, :1]
            reward_std = y_std[:, :1]
            reward_std_epistemic = y_std_epistemic[:, :1]
            state_next_mean = y_mean[:, 1:]
            state_next_std = y_std[:, 1:]
            state_next_std_epistemic = y_std_epistemic[:, 1:]
            state_next = state_next_mean + self.beta * state_next_std_epistemic * action_hallucinated
            if self.use_aleatoric:
                reward = torch.distributions.Normal(reward_mean, reward_std).sample()
                state_next_var = state_next_std**2
                state_next_var_epistemic = state_next_std_epistemic**2
                state_next_var_aleatoric = state_next_var - state_next_var_epistemic
                state_next_std_aleatoric = torch.sqrt(state_next_var_aleatoric)
                state_next = torch.distributions.Normal(state_next, state_next_std_aleatoric).sample()
            else:
                reward = torch.distributions.Normal(reward_mean, reward_std_epistemic).sample()
            state_next += state
            state_next = torch.clamp(
                state_next, self.bound_state_low, self.bound_state_high)
            reward = reward.squeeze().cpu().numpy()
            state_next = state_next.cpu().numpy()
            done = self.model_termination(state_next)
            return state_next, reward, done, {}

    @property
    def action_space(self):
        low = np.concatenate(
            (self.space_action.low, self.space_action_hallucinated.low))
        high = np.concatenate(
            (self.space_action.high, self.space_action_hallucinated.high))
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
