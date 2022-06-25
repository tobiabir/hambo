import math
import numpy as np
import random
import torch

import agents
import rollout


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_scores_calibration(y_pred_means, y_pred_stds, y_train):
    cdfs = torch.distributions.Normal(y_pred_means, y_pred_stds).cdf(y_train)
    levels_confidence = torch.linspace(0, 1, 11, device=cdfs.device) 
    num_preds = cdfs.shape[1] * cdfs.shape[2]
    levels_confidence_empirical = [(cdfs < p).sum(dim=(1, 2)) / num_preds for p in levels_confidence]
    levels_confidence_empirical = torch.stack(levels_confidence_empirical, dim=1)
    scores = ((levels_confidence_empirical - levels_confidence)**2).sum(dim=1)
    return scores


def get_mean_std_of_mixture(means, stds, epistemic=False):
    mean = torch.mean(means, dim=0)
    var_aleatoric = torch.mean(stds**2, dim=0)
    var_epistemic = torch.var(means, dim=0, unbiased=False)
    var = var_aleatoric + var_epistemic
    std = torch.sqrt(var)
    if epistemic:
        std_epistemic = torch.sqrt(var_epistemic)
        return mean, std, std_epistemic
    return mean, std


