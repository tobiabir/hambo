import math
import numpy as np
import random
import torch

import rollout


def soft_update(target, source, tau):
    """Perform a soft update of the targets parameters with the sources parameters.
    That is target = (1 - tau) * target + tau * source.

    Args:
        target: the target model
        source: the source model
        tau:    the tau parameter (should be in [0,1])

    Returns:
        None
    """
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def set_seeds(seed):
    """Set all the seeds.

    Args:
        seed:   the seed to use

    Returns:
        None
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_mean_std_mm(means, stds, epistemic=False):
    """Get mean and std of a Gaussian mixture.
    Note that we assume independent components.

    Args:
        means:      the means of the Gaussians
        stds:       the stds of the Gaussians
        epistemic:  set to return epistemic uncertainty as well

    Returns:
        mean:               the mean of the mixture
        std:                the std of the mixture (aleatoric and epistemic)
        (std_epistemic):    the epistemic std of the mixture
    """
    mean = torch.mean(means, dim=0)
    var_aleatoric = torch.mean(stds**2, dim=0)
    var_epistemic = torch.var(means, dim=0, unbiased=False)
    var = var_aleatoric + var_epistemic
    std = torch.sqrt(var)
    if epistemic:
        std_epistemic = torch.sqrt(var_epistemic)
        return mean, std, std_epistemic
    return mean, std


def golden_section_search(f, a, b, eps=1e-5):
    """Golden Section Search. Finds argmin of unimodal function f in interval [a, b].

    Args:
        f:      unimodal function
        a:      lower limit of the interval to search in
        b:      upper limit of the interval to search in
        eps:    accuracy until which the search should run
    """
    # define needed constant g
    g = (np.sqrt(5) - 1) / 2

    # get initial values
    # note: invariant: a < c < d < b and c = a + (1 - g) * (b - a) and d = a + g * (b - a)
    fa = f(a)
    fb = f(b)
    diff = b - a
    c = a + (1 - g) * diff
    d = a + g * diff
    fc = f(c)
    fd = f(d)

    # main loop
    while diff > eps:
        if fc < fd:
            b, fb = d, fd
            diff = b - a
            d, fd = c, fc
            c = a + (1 - g) * diff
            fc = f(c)
        else:
            a, fa = c, fc
            diff = b - a
            c, fc = d, fd
            d = a + g * diff
            fd = f(d)

    # return (everything in [a, b] would be reasonable)
    return c


class KernelRBF(torch.nn.Module):

    def __init__(self, sigma=10.0):
        super().__init__()
        self.gamma = 1 / (2 * sigma**2)

    def forward(self, x1, x2):
        norm_sq = torch.diag(x1 @ x1.T).unsqueeze(dim=1) - 2 * x1 @ x2.T + torch.diag(x2 @ x2.T).unsqueeze(dim=0)
        y = torch.exp(-self.gamma * norm_sq)
        return y
