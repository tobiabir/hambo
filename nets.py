import math
import torch

STD_LOG_MIN = -20
STD_LOG_MAX = 2
EPS = 1e-6

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
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

class NetDense(torch.nn.Module):

    def __init__(self, dim_x, dim_y, num_h, dim_h, size_ensemble=1):
        super().__init__()
        self.size_ensemble = size_ensemble
        self.layers = torch.nn.Sequential()
        self.layers.append(LayerLinear(dim_x, dim_h, size_ensemble))
        self.layers.append(torch.nn.ReLU())
        for idx_h in range(num_h - 1):
            self.layers.append(LayerLinear(dim_h, dim_h, size_ensemble))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(LayerLinear(dim_h, dim_y, size_ensemble))

    def forward(self, x):
        x = x.repeat(self.size_ensemble, 1, 1)
        y = self.layers(x).squeeze(dim=1)
        mean = torch.mean(y, dim=0)
        std = torch.std(y, dim=0)
        return y, mean, std

class NetGaussHomo(torch.nn.Module):

    def __init__(self, dim_x, dim_y, num_h, dim_h, size_ensemble=1):
        super().__init__()
        self.net = NetDense(dim_x, dim_y, num_h, dim_h, size_ensemble)
        self.stds_log = torch.nn.parameter.Parameter(torch.zeros((size_ensemble, 1,dim_y)))
        torch.nn.init.kaiming_uniform_(self.stds_log, a=math.sqrt(5))

    def forward(self, x):
        means, _, _ = self.net(x)
        stds_log = torch.clamp(self.stds_log, min=STD_LOG_MIN, max=STD_LOG_MAX)
        stds = stds_log.exp()
        mean = torch.mean(means, dim=0)
        var_aleatoric = torch.mean(stds**2, dim=0)
        var_epistemic = torch.var(means, dim=0)
        var = var_aleatoric + var_epistemic
        std = torch.sqrt(var)
        distr = torch.distributions.Normal(mean, std)
        y = distr.rsample()
        return y, mean, std

class NetGauss(torch.nn.Module):

    def __init__(self, dim_x, dim_y, num_h, dim_h):
        super().__init__()
        self.base = NetDense(dim_x, dim_h, num_h - 1, dim_h)
        self.activation = torch.nn.ReLU()
        self.head_mean = torch.nn.Linear(dim_h, dim_y)
        self.head_std_log = torch.nn.Linear(dim_h, dim_y)

    def forward(self, x):
        _, h, _ = self.base(x)
        h = self.activation(h)
        mean = self.head_mean(h)
        std_log = self.head_std_log(h)
        std_log = torch.clamp(std_log, min=STD_LOG_MIN, max=STD_LOG_MAX)
        std = std_log.exp()
        distr = torch.distributions.Normal(mean, std)
        y = distr.rsample()
        prob_log = distr.log_prob(y)
        return y, mean, prob_log


class PolicyGauss(torch.nn.Module):

    def __init__(self, dim_state, dim_action, num_h, dim_h, bound_action_low, bound_action_high):
        super().__init__()
        self.net = NetGauss(dim_state, dim_action, num_h, dim_h)
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
        action, mean, prob_log = self.net(state)
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
        y1, mean1, std1 = self.net1(x)
        y2, mean2, std2 = self.net2(x)
        return mean1, mean2
