import gpytorch
import gym
import numpy as np
import time
import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP 

N_STEPS_TRAIN = 8

class MyGP(gpytorch.models.ExactGP):

    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.LinearMean(4)
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
        

if __name__ == "__main__":
    
    env = gym.make("Pendulum-v1", g=9.81)
    dim_state = env.observation_space.shape[0]
    print(env.observation_space.shape)
    print(env.action_space)

    state = env.reset()
    state = torch.tensor(state)
    done = False

    data = []
    num_episodes = 64
    for idx_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state)
        done = False
        while not done:
            action = env.action_space.sample()
            action = torch.tensor(action)
            state_next, reward, done, info = env.step(action)
            state_next = torch.tensor(state_next)
            data.append((state, action, reward, state_next)) 
            state = state_next

    print(len(data))
    x_train = [torch.cat((state, action), dim=-1) for state, action, _, _ in data]
    x_train = torch.stack(x_train, dim=0)
    y_train = [state_next for _, _, _, state_next in data]
    y_train = torch.stack(y_train, dim=0)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    models = []
    for idx_dim_state in range(dim_state):
        models.append(MyGP(x_train, y_train[:, idx_dim_state], likelihood))
        print(models[idx_dim_state].train_inputs)
    
    likelihood.train()
    for idx_dim_state in range(dim_state):
        models[idx_dim_state].train()

    optimizers = []
    for idx_dim_state in range(dim_state):
        optimizers.append(torch.optim.Adam(models[idx_dim_state].parameters(), lr=0.1))

    mlls = []
    fns_loss = []
    for idx_agent in range(dim_state):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, models[idx_agent])
        fns_loss.append(lambda y_pred, y_train: -mll(y_pred, y_train))

    for idx_step_train in range(N_STEPS_TRAIN):
        for idx_dim_state in range(dim_state):
            optimizers[idx_dim_state].zero_grad()
            y_pred = models[idx_dim_state](x_train)
            loss = fns_loss[idx_agent](y_pred, y_train[:, idx_agent])
            print(loss)
            loss.backward()
            optimizers[idx_dim_state].step()

    likelihood.eval()
    for idx_dim_state in range(dim_state):
        models[idx_dim_state].eval()

    state = env.reset()
    state = torch.tensor(state)
    x_test = torch.cat((state, torch.tensor([0.])))
    x_test = x_test.unsqueeze(dim=0)
    print(x_test)
    for idx_dim_state in range(dim_state):
        mean = models[idx_dim_state].mean_module(x_test)
        print(mean)
    state_next, reward, done, info = env.step(np.array([0]))
    print(state_next)

