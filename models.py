import gpytorch
import math
import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP 

STD_MIN = math.exp(-20)
STD_MAX = math.exp(2)

class GPExact(gpytorch.models.ExactGP):

    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.LinearMean(4)
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class ModelGP(torch.nn.Module):
    
    def __init__(self, x_train, y_train):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.models = torch.nn.ModuleList()
        self.optimizers = []
        self.fns_loss = []
        for idx_model in range(y_train.shape[1]):
            self.models.append(GPExact(x_train, y_train[:,idx_model], self.likelihood))
            self.optimizers.append(torch.optim.Adam(self.models[idx_model].parameters(), lr=0.1))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.models[idx_model])
            self.fns_loss.append(lambda y_pred, y_train: -mll(y_pred, y_train))

    def forward(self, x):
        means = []
        variances = []
        for idx_model in range(len(self.models)):
            distr = self.models[idx_model](x)
            mean = distr.mean
            covar = distr.covariance_matrix
            covar = covar.clamp(min=STD_MIN, max=STD_MAX)
            distr = gpytorch.distributions.MultivariateNormal(mean, covar)
            means.append(mean)
            variances.append(covar[0])
        mean = torch.stack(means, dim=-1)
        var = torch.stack(variances, dim=-1)
        std = torch.sqrt(var)
        return mean, std

    def get_distr(self, x):
        return self(x)

    def step(self):
        losses = []
        for idx_model in range(len(self.models)): 
            x_train = self.models[idx_model].train_inputs[0]
            y_train = self.models[idx_model].train_targets
            self.optimizers[idx_model].zero_grad()
            y_pred = self.models[idx_model](x_train)
            loss = self.fns_loss[idx_model](y_pred, y_train)
            loss.backward()
            self.optimizers[idx_model].step()
            losses.append(loss.detach())
        return losses

