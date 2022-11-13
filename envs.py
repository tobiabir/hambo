import gym
import numpy as np
import torch

import utils


class EnvPoint(gym.core.Env):

    def __init__(self, dim_state):
        super().__init__()
        self.dim_state = dim_state

    @property
    def action_space(self):
        return gym.spaces.Box(low=-0.1, high=0.1, shape=(self.dim_state,))

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim_state,))

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment

        Returns:
            state : agent's state of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        prev_state = self._state
        self._state = prev_state + np.clip(action, -0.1, 0.1)
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_state = np.copy(self._state)
        return next_state, reward, done, {}

    def reset(self, seed=None):
        """Resets the state of the environment, returning an initial state.

        Args:
            seed:   rng seed to use

        Returns:
            state:  the initial state
        """
        super().reset(seed=seed)
        sample = np.zeros(self.dim_state, dtype=np.float32)
        while not sample.any():
            sample = np.random.normal(0, 1, self.dim_state).astype(np.float32)
        self._state = sample / np.linalg.norm(sample)
        state = np.copy(self._state)
        return state

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False

    def reward(self, state, act, state_next):
        return - state_next @ state_next


class EnvPointEscape(EnvPoint):

    def reset(self, seed):
        self._state = np.zeros(self.dim_state)
        return np.zeros(self.dim_state)

    def reward(self, state, action, state_next):
        return state_next @ state_next


class WrapperEnv(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.state = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = np.float32(state)
        reward = np.float32(reward)
        if "TimeLimit.truncated" not in info:
            info["TimeLimit.truncated"] = False
        self.state = state
        return state, reward, done, info

    def reset(self, seed=None):
        state = self.env.reset(seed=seed)
        state = np.float32(state)
        self.state = state
        return state


class WrapperEnvMaze(WrapperEnv):

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False


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
        not_done = np.isfinite(state).all(axis=-1) * (np.abs(state[..., 1:]) < 100).all(axis=-1) * (height > 0.7) * (np.abs(angle) < 0.2)
        done = ~not_done
        return done


class WrapperEnvHalfCheetah(WrapperEnv):

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False


class WrapperEnvReacher(WrapperEnv):

    def done(self, state):
        if len(state.shape) == 2:
            return np.zeros(state.shape[0], dtype=np.bool_)
        return False


class WrapperEnvWalker(WrapperEnv):

    def done(self, state):
        z = state[..., 0]
        angle = state[..., 1]
        is_healthy_z = np.logical_and(0.8 < z, z < 2.0)
        is_healthy_angle = np.logical_and(-1.0 < angle, angle < 1.0)
        is_healthy = np.logical_and(is_healthy_z, is_healthy_angle)
        return is_healthy


class WrapperEnvProtagonist(gym.core.Wrapper):

    def __init__(self, env, agent):
        super().__init__(env)
        self.agent = agent
        self.state = None

    def step(self, action):
        action = self.agent.get_action(self.state), action
        state_next, reward, done, info = self.env.step(action)
        self.state = state_next
        return state_next, -reward, done, info

    def reset(self, seed=None):
        state = self.env.reset(seed=seed)
        self.state = state
        return state

    @property
    def action_space(self):
        return self.env.space_action_hallucinated


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
        self.method_sampling = args.method_sampling
        self.use_aleatoric = args.use_aleatoric
        self.weight_penalty_reward = args.weight_penalty_reward
        self.device = args.device

    def _step(self, state, action):
        # make the inputs to torch tensors
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)

        # create model input and get predictions from model
        x = torch.cat((state, action), dim=-1)
        with torch.no_grad():
            y_means, y_stds = self.model_transition(x)

        # sample according to the sampling method
        if self.method_sampling == "DS":
            y_mean, y_std, y_std_epistemic = self.model_transition._aggregate_distrs(y_means, y_stds, epistemic=True)
            if self.use_aleatoric:
                y = torch.distributions.Normal(y_mean, y_std).sample()
            else:
                y = torch.distributions.Normal(y_mean, y_std_epistemic).sample()
        elif self.method_sampling == "TS1":
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
        else:
            y_mean, y_std = y_means[self.model_transition.idxs_elites], y_stds[self.model_transition.idxs_elites]
            y_mean, y_std = self.model_transition.scaler_y.inverse_transform(y_mean, y_std)
            if self.use_aleatoric:
                y = torch.distributions.Normal(y_mean, y_std).sample()
            else:
                y = y_mean

        # get reward and apply reward penalty
        reward = y[..., :1]
        penalty_reward = torch.amax(torch.linalg.norm(y_stds, dim=2), dim=0).unsqueeze(dim=1)
        reward -= self.weight_penalty_reward * penalty_reward

        # get next state and add old state (note: we predict state difference)
        state_next = state + y[..., 1:]

        # clamp to get valid next state
        state_next = torch.clamp(state_next, self.bound_state_low, self.bound_state_high)

        # make the predictions to numpy arrays
        reward = reward.squeeze(dim=-1).cpu().numpy()
        state_next = state_next.cpu().numpy()

        # get terminals from termination model
        done = self.model_termination(state_next)

        return state_next, reward, done, {}

    def step(self, action):
        state = np.expand_dims(self.state, axis=0)
        if isinstance(action, tuple):
            action = tuple(np.expand_dims(action, axis=0) for action in action)
        else:
            action = np.expand_dims(action, axis=0)
        state_next, reward, done, info = self._step(state, action)
        state_next = state_next.squeeze(axis=0)
        reward = reward.squeeze(axis=0)
        done = done.squeeze()
        self.state = state_next
        return state_next, reward, done, info

    def rollout(self, agent, dataset, num_states_initial, max_length_rollout):
        state = self.dataset_states_initial.sample(num_states_initial)
        returns = np.zeros(num_states_initial)
        active = np.array([True] * num_states_initial)
        for idx_step in range(max_length_rollout):
            action = agent.get_action(state)
            state_next, reward, done, info = self._step(state, action)
            returns[active] += reward
            if dataset is not None:
                # note: termination model has no time limit (which is added via wrapper)
                mask = done.astype(np.float32)
                if isinstance(action, tuple):
                    action = zip(*action)
                batch = list(zip(state, action, reward, state_next, mask))
                dataset.push_batch(batch)
            active_next = np.logical_not(done)
            state = state_next[active_next]
            active[active] = active_next
            if state.size == 0:
                break
        return returns

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.dataset_states_initial.sample(1)[0]
        return self.state

    @property
    def observation_space(self):
        return self.space_observation

    @property
    def action_space(self):
        return self.space_action


class EnvModelHallucinated(EnvModel):

    def __init__(self, space_observation, space_action, dataset_states_initial, model_transition, model_termination, args):
        super().__init__(space_observation, space_action, dataset_states_initial, model_transition, model_termination, args)
        if self.method_sampling == "DS":
            self.space_action_hallucinated = gym.spaces.Box(low=-1, high=1, shape=space_observation.shape, dtype=np.float32)
        else:
            self.space_action_hallucinated = gym.spaces.Discrete(model_transition.num_elites)
        self.beta = args.beta

    def _step(self, state, action):
        # make the inputs to torch tensors
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action, action_hallucinated = action
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if self.method_sampling == "DS":
            action_hallucinated = torch.tensor(action_hallucinated, dtype=torch.float32, device=self.device)
        else:
            action_hallucinated = torch.tensor(action_hallucinated, dtype=torch.int64, device=self.device)

        # create model input and get predictions from model
        x = torch.cat((state, action), dim=-1)
        with torch.no_grad():
            y_means, y_stds = self.model_transition(x)

        if self.method_sampling == "DS":
            # aggregate
            y_mean, y_std, y_std_epistemic = self.model_transition._aggregate_distrs(y_means, y_stds, epistemic=True)
            # extract reward predictions
            reward_mean = y_mean[:, :1]
            reward_std = y_std[:, :1]
            reward_std_epistemic = y_std_epistemic[:, :1]

            # extract next state predictions
            state_next_mean = y_mean[:, 1:]
            state_next_std = y_std[:, 1:]
            state_next_std_epistemic = y_std_epistemic[:, 1:]
            state_next_var = state_next_std**2
            state_next_var_epistemic = state_next_std_epistemic**2
            state_next_var_aleatoric = state_next_var - state_next_var_epistemic
            state_next_std_aleatoric = torch.sqrt(state_next_var_aleatoric)

            # apply hallucinated control
            state_next_mean = state_next_mean + self.beta * state_next_std_epistemic * action_hallucinated

            # sample
            if self.use_aleatoric:
                reward = torch.distributions.Normal(reward_mean, reward_std).sample()
                state_next = torch.distributions.Normal(state_next_mean, state_next_std_aleatoric).sample()
            else:
                reward = torch.distributions.Normal(reward_mean, reward_std_epistemic).sample()
                state_next = state_next_mean

        else:
            # extract predictions
            y_means = y_means[self.model_transition.idxs_elites]
            y_stds = y_stds[self.model_transition.idxs_elites]
            size_batch = x.shape[0]
            idxs_model = torch.randint(0, self.model_transition.num_elites, (size_batch,), device=self.device)
            idxs_batch = torch.arange(0, x.shape[0], device=self.device)
            reward_mean = y_means[idxs_model, idxs_batch, :1]
            reward_std = y_stds[idxs_model, idxs_batch, :1]
            state_next_mean = y_means[action_hallucinated, idxs_batch, 1:]
            state_next_std = y_stds[action_hallucinated, idxs_batch, 1:]
            y_mean = torch.cat((reward_mean, state_next_mean), dim=-1)
            y_std = torch.cat((reward_std, state_next_std), dim=-1)
            y_mean, y_std = self.model_transition.scaler_y.inverse_transform(y_mean, y_std)

            # sample
            if self.use_aleatoric:
                y = torch.distributions.Normal(y_mean, y_std).sample()
            else:
                y = y_mean

            # split into reward and next state
            reward = y[:, :1]
            state_next = y[:, 1:]

        # we predict state_next - state -> add state to next state
        state_next += state

        # clamp to get valid next state
        state_next = torch.clamp(state_next, self.bound_state_low, self.bound_state_high)

        # make the predictions to numpy arrays
        reward = reward.squeeze(dim=-1).cpu().numpy()
        state_next = state_next.cpu().numpy()

        # get terminals from termination model
        done = self.model_termination(state_next)

        return state_next, reward, done, {}

    @property
    def action_space(self):
        return gym.spaces.Tuple((self.space_action, self.space_action_hallucinated))
