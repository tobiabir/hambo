import numpy as np
import torch

from abc import ABC, abstractmethod

import nets
import utils


class Agent(ABC):

    def __init__(self):
        pass

    def eval(self):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    def step(self, data):
        pass

    def train(self):
        pass

class AgentRandom(Agent):

    def __init__(self, space_action):
        self.space_action = space_action

    def get_action(self, state):
        return self.space_action.sample()

class AgentZero(Agent):

    def __init__(self, space_action):
        self.dim_action = space_action.shape[0]

    def get_action(self, state):
        return np.zeros(self.dim_action, dtype=np.float32)

class AgentPointOptimal(Agent):

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        action_max_abs = np.ones(2) * 0.1
        state_abs = np.abs(state)
        action_abs = np.min((state_abs, action_max_abs), axis=-1)
        action = - np.sign(state) * action_abs
        return action

class AgentSAC(Agent):

    def __init__(self, space_state, space_action, args):
        super().__init__()
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device
        dim_state = space_state.shape[0]
        dim_action = space_action.shape[0]
        num_h = 2
        dim_h = 256
        self.critic = nets.NetDoubleQ(dim_state, dim_action, num_h, dim_h).to(self.device)
        self.critic_target = nets.NetDoubleQ(dim_state, dim_action, num_h, dim_h).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.optim_critic = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr)
        bound_action_low = space_action.low[0]
        bound_action_high = space_action.high[0]
        self.policy = nets.PolicyGauss(
            dim_state, dim_action, num_h, dim_h, bound_action_low, bound_action_high).to(self.device)
        self.optim_policy = torch.optim.Adam(
            self.policy.parameters(), lr=args.lr)
        self.entropy_target = -np.prod(space_action.shape).astype(np.float32)
        self.learn_alpha = args.learn_alpha
        if self.learn_alpha:
            self.alpha_log = torch.tensor(0., requires_grad=True).to(self.device)
            self.alpha = self.alpha_log.detach().exp()
            self.optim_alpha = torch.optim.Adam([self.alpha_log], lr=args.lr)
        else:
            self.alpha = args.alpha
        self.training = True

    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.training = False

    def get_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state).to(self.device)
            action, mean, _ = self.policy(state)
            if self.training:
                action = action.cpu().numpy()
            else:
                action = mean.cpu().numpy()
            return action

    def step(self, data):
        state, action, reward, state_next, done = data
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

        return loss_q, loss_pi, loss_alpha

    def train(self):
        self.policy.train()
        self.critic.train()
        self.training = True

