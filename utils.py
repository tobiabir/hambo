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


def get_scores_calibration(y_pred_means, y_pred_stds, y_train, *args):
    """Get the calibration score. See [Kuleshov et al.](https://proceedings.mlr.press/v80/kuleshov18a) eq. 9.

    Args:
        y_pred_means:   mean predictions
        y_pred_std:     std predictions
        y_train:        ground truth
        *args:          to discard additional arguments which allows common interface with other evaluation scores

    Returns:
        scores: the calibration scores (one for every model in the ensemble)
    """
    cdfs = torch.distributions.Normal(y_pred_means, y_pred_stds).cdf(y_train)
    levels_confidence = torch.linspace(0, 1, 11, device=cdfs.device) 
    num_preds = cdfs.shape[1] * cdfs.shape[2]
    levels_confidence_empirical = [(cdfs < p).sum(dim=(1, 2)) / num_preds for p in levels_confidence]
    levels_confidence_empirical = torch.stack(levels_confidence_empirical, dim=1)
    scores = ((levels_confidence_empirical - levels_confidence)**2).sum(dim=1)
    return scores


def get_scores_calibration_symmetric(y_pred_means, y_pred_stds, y_train, *args):
    """Get the calibration score based on symmetric confidence intervals. Adapted from [Kuleshov et al.](https://proceedings.mlr.press/v80/kuleshov18a) eq. 9.

    Args:
        y_pred_means:   mean predictions
        y_pred_std:     std predictions
        y_train:        ground truth
        *args:          to discard additional arguments which allows common interface with other evaluation scores

    Returns:
        scores: the calibration scores (one for every model in the ensemble)
    """
    distr = torch.distributions.Normal(0, 1)
    y_train = (y_train - y_pred_means) / y_pred_stds
    y_train_abs = torch.abs(y_train)
    levels_confidence = torch.linspace(0, 1, 11, device=y_train.device) 
    percentiles = distr.icdf((1 + levels_confidence) / 2) # == - icdf((1-levels_confidence) / 2)
    num_preds = y_pred_means.shape[1] * y_pred_means.shape[2]
    levels_confidence_empirical = [(y_train_abs < p).sum(dim=(1, 2)) / num_preds for p in percentiles]
    levels_confidence_empirical = torch.stack(levels_confidence_empirical, dim=1)
    scores = ((levels_confidence_empirical - levels_confidence)**2).sum(dim=1)
    return scores


def get_mean_std_of_mixture(means, stds, epistemic=False):
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


