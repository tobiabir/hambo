import numpy as np
import torch

from abc import ABC, abstractmethod

import models
import utils


class Agent(ABC):
    """Base class for agents.

    Methods:
        eval:        set agent in eval mode
        get_action:  given a state get an action
        step:        do a training step
        train:       set agent in train mode
    """

    def __init__(self):
        pass

    def eval(self):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    def step(self, data):
        return (None,) * 3

    def train(self):
        pass


class AgentRandom(Agent):
    """Agent taking random actions.

    Attributes:
       space_action:    gym action space
    """

    def __init__(self, space_action):
        self.space_action = space_action
        self.dim_action = space_action.shape[0]

    def get_action(self, state):
        if len(state.shape) == 2:
            actions = [self.space_action.sample() for _ in range(state.shape[0])]
            action = np.stack(actions, axis=0)
            return action
        return self.space_action.sample()


class AgentPolynomial(Agent):

    def __init__(self, space_action, coef):
        super().__init__()
        bound_action_low = space_action.low[0]
        bound_action_high = space_action.high[0]
        self.bound_action_scale = (bound_action_high - bound_action_low) / 2
        self.bound_action_bias = (bound_action_high + bound_action_low) / 2
        self.p = np.polynomial.Polynomial(coef)

    def get_action(self, state):
        action = self.p(state)
        action = self.bound_action_scale * np.tanh(action) + self.bound_action_bias
        return action


class AgentPointRandom(Agent):
    """Agent taking random actions for the Point environment.
    """
    
    def __init__(self, alpha, beta):
        super().__init__()
        self.distr = torch.distributions.Beta(alpha, beta)
        self.bound_action_low = -0.1
        self.bound_action_high = 0.1

    def get_action(self, state):
        sample = self.distr.sample(state.shape).numpy()
        action = self.bound_action_low + (self.bound_action_high - self.bound_action_low) * sample
        sign = np.sign(state)
        sign[sign == 0] = 1
        action = sign * action
        return action


class AgentPointOptimal(Agent):
    """Agent taking the optimal actions for the Point environment.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        action_max_abs = np.ones(state.shape) * 0.1
        state_abs = np.abs(state)
        action_abs = np.minimum(state_abs, action_max_abs)
        action = - np.sign(state) * action_abs
        return action


class AgentPointEscapeOptimal(Agent):
    """Agent taking the optimal actions for the PointEscape environment.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        return np.ones(state.shape) * 0.1


class AgentPointOptimalAntagonist(Agent):
    """Agent taking the optimal adversarial actions for the Point environment.
    """
    
    def get_action(self, state):
        action_abs = np.ones(state.shape)
        action = np.sign(state) * action_abs
        return action


class AgentPointEscapeOptimalAntagonist(Agent):
    """Agent taking the optimal adversarial actions for the PointEscape environment.
    """
    
    def get_action(self, state):
        action_abs = np.ones(state.shape)
        action = - np.sign(state) * action_abs
        return action
        

class AgentSAC(Agent):
    """General agent learning via [SAC](https://arxiv.org/abs/1801.01290).
    """

    def __init__(self, space_state, space_action, args):
        super().__init__()
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device
        dim_state = space_state.shape[0]
        dim_action = space_action.shape[0]
        self.dim_action = dim_action
        num_h = 2
        dim_h = 256
        self.critic = models.NetDoubleQ(dim_state, dim_action, num_h, dim_h).to(self.device)
        self.critic_target = models.NetDoubleQ(dim_state, dim_action, num_h, dim_h).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.optim_critic = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr_agent)
        bound_action_low = space_action.low[0]
        bound_action_high = space_action.high[0]
        self.policy = models.PolicyGauss(
            dim_state, dim_action, num_h, dim_h, bound_action_low, bound_action_high).to(self.device)
        self.optim_policy = torch.optim.Adam(
            self.policy.parameters(), lr=args.lr_agent)
        self.entropy_target = -np.prod(space_action.shape).astype(np.float32)
        self.learn_alpha = args.learn_alpha
        if self.learn_alpha:
            self.alpha_log = torch.tensor(0., requires_grad=True).to(self.device)
            self.alpha = self.alpha_log.detach().exp()
            self.optim_alpha = torch.optim.Adam([self.alpha_log], lr=args.lr_agent)
        else:
            self.alpha = args.alpha
        self.conservative = args.conservative
        self.weight_conservative = 5.0
        self.size_batch_env = int(args.ratio_env_model * args.size_batch)
        self.training = True

    def _preprocess_data(iself, data):
        return data

    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.training = False

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            action, mean, _ = self.policy(state)
        if self.training:
            action = action.cpu().numpy()
        else:
            action = mean.cpu().numpy()
        if len(state.shape) == 1:
            action = action.squeeze(axis=0)
        return action

    def step(self, data):
        state, action, reward, state_next, done = self._preprocess_data(data)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        state_next = state_next.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            action_next, _, prob_log_next = self.policy(state_next)
            q_target_next = torch.min(*self.critic_target(state_next, action_next))
            q_soft_target_next = q_target_next - self.alpha * prob_log_next
            q_target = reward + (1 - done) * self.gamma * q_soft_target_next
        q1, q2 = self.critic(state, action)
        loss_q1 = torch.nn.functional.mse_loss(q1, q_target)
        loss_q2 = torch.nn.functional.mse_loss(q2, q_target)

        if self.conservative:
            # Note: COMBO would actually only use the model states here.
            # However I argue the action distribution shift alone already demands conservatism.
            # Therefore we use all the states.
            num_samples = 10
            state_tmp = state.unsqueeze(dim=1).repeat(1, num_samples, 1).reshape(-1, state.shape[1])
            action_random = torch.distributions.Uniform(-1, 1).sample((state_tmp.shape[0], action.shape[1])).to(self.device)
            prob_log_random = np.log(0.5 ** action_random.shape[-1])
            action_tmp, _, prob_log_tmp = self.policy(state_tmp)
            state_next_tmp = state_next.unsqueeze(dim=1).repeat(1, num_samples, 1).reshape(-1, state.shape[-1])
            action_next_tmp, _, prob_log_next_tmp = self.policy(state_next_tmp)
            q1_random, q2_random = self.critic(state_tmp, action_random)
            q1_curr, q2_curr = self.critic(state_tmp, action_tmp)
            q1_next, q2_next = self.critic(state_tmp, action_next_tmp)

            cat_q1 = torch.cat([q1_random - prob_log_random, q1_next - prob_log_next_tmp.detach(), q1_curr - prob_log_tmp.detach()], dim=1)
            cat_q2 = torch.cat([q2_random - prob_log_random, q2_next - prob_log_next_tmp.detach(), q2_curr - prob_log_tmp.detach()], dim=1)
                
            # Note: As model data has action distribution shift even for rollout length 1
            # it always makes sense to exclude model data here.
            q1_env, q2_env = self.critic(state[:self.size_batch_env], action[:self.size_batch_env])

            loss_q1_conservative = torch.logsumexp(cat_q1, dim=1,).mean() - q1_env.mean()
            loss_q2_conservative = torch.logsumexp(cat_q2, dim=1,).mean() - q2_env.mean()
            
            loss_q1 += self.weight_conservative * loss_q1_conservative
            loss_q2 += self.weight_conservative * loss_q2_conservative

        loss_q = loss_q1 + loss_q2
        self.optim_critic.zero_grad()
        loss_q.backward()
        self.optim_critic.step()

        action, _, prob_log = self.policy(state)
        q = torch.min(*self.critic(state, action))
        loss_pi = (self.alpha * prob_log - q).mean()
        self.optim_policy.zero_grad()
        loss_pi.backward()
        self.optim_policy.step()

        if self.learn_alpha:
            loss_alpha = -(self.alpha_log * (prob_log +
                                         self.entropy_target).detach()).mean()
            self.optim_alpha.zero_grad()
            loss_alpha.backward()
            self.optim_alpha.step()
            self.alpha = self.alpha_log.detach().exp()
        else:
            loss_alpha = 0

        utils.soft_update(self.critic_target, self.critic, self.tau)

        return loss_pi, loss_q, loss_alpha

    def train(self):
        self.policy.train()
        self.critic.train()
        self.training = True


class AgentSACAntagonist(AgentSAC):
    """General agent learning to get minimal reward via [SAC](https://arxiv.org/abs/1801.01290).
    """

    def __init__(self, space_state, space_action, args):
        super().__init__(space_state, space_action, args)

    def _preprocess_data(self, data):
        state, action, reward, state_next, done = data
        return state, action, -reward, state_next, done


class AgentConcat(Agent):
    """Combines agents into one agent giving one action that is the concatenation of the agents actions.
    """
    
    def __init__(self, agents):
        super().__init__()
        self.agents = agents

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def get_action(self, state):
        actions = [agent.get_action(state) for agent in self.agents]
        action = np.concatenate(actions, axis=-1)
        return action

    def step(self, data):
        state, action, reward, state_next, done = data
        dims_action = [agent.dim_action for agent in self.agents]
        idxs_split = np.cumsum(dims_action)[:-1]
        actions = np.split(action, idxs_split, axis=-1)
        losses = []
        for idx_agent in range(len(self.agents)):
            loss = self.agents[idx_agent].step((state, actions[idx_agent], reward, state_next, done))
            losses.append(loss)
        return list(zip(*losses))
            
    def train(self):
        for agent in self.agents:
            agent.train()


class AgentDOPE(Agent):
    """DOPE agent.
    """

    def __init__(self, weights):
        self.fc0_w = weights["fc0/weight"]
        self.fc0_b = weights["fc0/bias"]
        self.fc1_w = weights["fc1/weight"]
        self.fc1_b = weights["fc1/bias"]
        self.fclast_w = weights["last_fc/weight"]
        self.fclast_b = weights["last_fc/bias"]
        self.fclast_w_std_log = weights["last_fc_log_std/weight"]
        self.fclast_b_std_log = weights["last_fc_log_std/bias"]
        relu = lambda x: np.maximum(x, 0)
        self.nonlinearity = np.tanh if weights["nonlinearity"] == "tanh" else relu
        identity = lambda x: x
        self.output_transformation = np.tanh if weights["output_distribution"] == "tanh_gaussian" else identity
        self.dim_action = self.fclast_w.shape[0]

    def get_action(self, state):
        x = state @ self.fc0_w.T + self.fc0_b
        x = self.nonlinearity(x)
        x = x @ self.fc1_w.T + self.fc1_b
        x = self.nonlinearity(x)
        mean = x @ self.fclast_w.T + self.fclast_b
        std_log = x @ self.fclast_w_std_log.T + self.fclast_b_std_log
        std = np.exp(std_log)
        action = self.output_transformation(np.random.normal(mean, std))
        return action

