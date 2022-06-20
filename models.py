import gpytorch
import math
import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP 

import utils

STD_MIN = math.exp(-20)
STD_MAX = math.exp(2)
STD_LOG_MIN = -10
STD_LOG_MAX = 1
EPS = 1e-6

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


class LayerLinear(torch.nn.Module):
    
    def __init__(self, dim_x, dim_y, size_ensemble=1):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.zeros((size_ensemble, dim_x, dim_y)))
        self.bias = torch.nn.parameter.Parameter(torch.zeros((size_ensemble, 1, dim_y)))
        self.reset_parameters()

    def forward(self, x):
        y = torch.bmm(x, self.weight)
        y += self.bias
        return y

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for idx_model in range(self.weight.shape[0]):
            torch.nn.init.kaiming_uniform_(self.weight[idx_model], a=math.sqrt(5))
        if self.bias is not None:
            _, fan_in = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)


class NetDense(torch.nn.Module):

    def __init__(self, dim_x, dim_y, num_h, dim_h, size_ensemble=1, num_elites=1, use_scalers=False):
        super().__init__()
        self.dim_x = dim_x
        self.layers = torch.nn.Sequential()
        self.layers.append(LayerLinear(dim_x, dim_h, size_ensemble))
        self.layers.append(torch.nn.ReLU())
        for idx_h in range(num_h - 1):
            self.layers.append(LayerLinear(dim_h, dim_h, size_ensemble))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(LayerLinear(dim_h, dim_y, size_ensemble))
        self.scaler_x = utils.ScalerStandard()
        self.scaler_y = utils.ScalerStandard()
        if not use_scalers:
            self.scaler_x.deactivate()
            self.scaler_y.deactivate()
        self.size_ensemble = size_ensemble
        self.num_elites = num_elites
        self.idxs_elites = torch.arange(0, num_elites)

    def _apply_layers(self, x):
        x = self.scaler_x.transform(x)
        x = x.repeat(self.size_ensemble, 1, 1)
        y = self.layers(x)
        return y 

    def _extract_distrs(self, y):
        means = y
        stds = torch.ones(y.shape, dtype=torch.float32, device=y.device)
        return means, stds

    def forward(self, x):
        y = self._apply_layers(x)
        means, stds = self._extract_distrs(y)
        return means, stds

    def get_distr(self, x, epistemic=False):
        means, stds = self(x)
        means, stds = means[self.idxs_elites], stds[self.idxs_elites]
        mean = torch.mean(means, dim=0)
        var_aleatoric = torch.mean(stds**2, dim=0)
        var_epistemic = torch.var(means, dim=0, unbiased=False)
        var = var_aleatoric + var_epistemic
        std = torch.sqrt(var)
        mean, std = self.scaler_y.inverse_transform(mean, std)
        if epistemic:
            std_epistemic = torch.sqrt(var_epistemic)
            _, std_epistemic = self.scaler_y.inverse_transform(mean, std_epistemic)
            return mean, std, std_epistemic
        return mean, std

    def load_state_dict_single(self, state_dict_new, idx_model):
        state_dict = self.layers.state_dict()
        for key in state_dict:
            state_dict[key][idx_model] = state_dict_new[key][idx_model]
        self.layers.load_state_dict(state_dict)


class NetGaussHomo(NetDense):

    def __init__(self, dim_x, dim_y, num_h, dim_h, size_ensemble=1, num_elites=1, use_scalers=False):
        super().__init__(dim_x, dim_y, num_h, dim_h, size_ensemble, num_elites, use_scalers)
        stds_log = torch.zeros((size_ensemble, 1, dim_y))
        torch.nn.init.kaiming_uniform_(stds_log, a=math.sqrt(5))
        self.stds_log = torch.nn.parameter.Parameter(stds_log)

    def _extract_distrs(self, y):
        means = y
        stds_log = STD_LOG_MAX - torch.nn.functional.softplus(STD_LOG_MAX - self.stds_log)
        stds_log = STD_LOG_MIN + torch.nn.functional.softplus(self.stds_log - STD_LOG_MIN)
        stds_log = stds_log.repeat(1, means.shape[1], 1)
        if len(means.shape) == 2:
            stds_log = stds_log.squeeze(dim=1)
        stds = stds_log.exp()
        return means, stds


class NetGaussHetero(NetDense):

    def __init__(self, dim_x, dim_y, num_h, dim_h, size_ensemble=1, num_elites=1, use_scalers=False):
        super().__init__(dim_x, 2 * dim_y, num_h, dim_h, size_ensemble, num_elites, use_scalers)
        self.dim_y = dim_y
    
    def _extract_distrs(self, y):
        means = y[..., :self.dim_y]
        stds_log = y[..., self.dim_y:]
        stds_log = STD_LOG_MAX - torch.nn.functional.softplus(STD_LOG_MAX - stds_log)
        stds_log = STD_LOG_MIN + torch.nn.functional.softplus(stds_log - STD_LOG_MIN)
        stds = stds_log.exp()
        return means, stds


class PolicyGauss(torch.nn.Module):

    def __init__(self, dim_state, dim_action, num_h, dim_h, bound_action_low, bound_action_high):
        super().__init__()
        self.net = NetGaussHetero(dim_state, dim_action, num_h, dim_h)
        bound_action_scale = (bound_action_high - bound_action_low) / 2
        self.bound_action_scale = torch.tensor(
            bound_action_scale, dtype=torch.float32)
        bound_action_bias = (bound_action_high + bound_action_low) / 2
        self.bound_action_bias = torch.tensor(
            bound_action_bias, dtype=torch.float32)

    def _project(self, y, mean, prob_log):
        tmp = torch.tanh(y)
        y = tmp * self.bound_action_scale + self.bound_action_bias
        mean = torch.tanh(mean) * self.bound_action_scale + self.bound_action_bias
        prob_log -= torch.log(self.bound_action_scale * (1 - tmp.pow(2)) + EPS)
        return y, mean, prob_log

    def forward(self, state):
        mean, std = self.net.get_distr(state)
        mean, std = mean.squeeze(dim=0), std.squeeze(dim=0)
        distr = torch.distributions.Normal(mean, std)
        action = distr.rsample()
        prob_log = distr.log_prob(action)
        action, mean, prob_log = self._project(action, mean, prob_log)
        prob_log = prob_log.sum(-1, keepdim=True)
        return action, mean, prob_log


class NetDoubleQ(torch.nn.Module):

    def __init__(self, dim_state, dim_action, num_h, dim_h):
        super().__init__()
        self.net1 = NetDense(dim_state + dim_action, 1, num_h, dim_h)
        self.net2 = NetDense(dim_state + dim_action, 1, num_h, dim_h)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        mean1, std1 = self.net1.get_distr(x)
        mean2, std2 = self.net2.get_distr(x)
        return mean1, mean2
