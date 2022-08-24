import torch

import utils

def calibrate(model, dataset, device="cpu"):
    """Calibrate model using temperature scaling.

    Args:
        model:      the model to calibrate
        dataset:    the dataset to use

    Returns:
        None
    """
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
    state, action, reward, state_next, _ = next(iter(dataloader)) 
    reward = reward.unsqueeze(dim=1)
    x = torch.cat((state, action[0]), dim=-1).to(device)
    y = torch.cat((reward, state_next - state), dim=-1).to(device)
    y = model.scaler_y.transform(y)
    with torch.no_grad():
        y_pred_means, y_pred_stds = model(x)

    def f(x):
        score = get_score_calibration_mm(y_pred_means, y_pred_stds, y, temperature=x)
        return score

    # get initial score and temperature
    temperature = model.temperature
    score = f(temperature)

    # exponential search up
    temperature_upper = temperature * 2
    score_upper_old = score
    score_upper = f(temperature_upper)
    while score_upper < score_upper_old:
        temperature_upper *= 2
        score_upper_old = score_upper
        score_upper = f(temperature_upper)
    temperature_upper = temperature_upper / 2

    # exponential search down
    temperature_lower = temperature / 2
    score_lower_old = score
    score_lower = f(temperature_lower)
    while score_lower < score_lower_old:
        temperature_lower /= 2
        score_lower_old = score_lower
        score_lower = f(temperature_lower)
    temperature_lower = temperature_lower * 2

    # extract 
    if score_lower_old < score_upper_old:
        temperature = temperature_lower
    else:
        temperature = temperature_upper
    temperature_min = temperature / 2
    temperature_max = temperature * 2

    temperature = utils.golden_section_search(f, temperature_min, temperature_max, eps=0.01)

    model.temperature = temperature


def get_scores_calibration(y_pred_means, y_pred_stds, y_train, **kwargs):
    """Get the calibration scores based on symmetric confidence intervals. Adapted from [Kuleshov et al.](https://proceedings.mlr.press/v80/kuleshov18a) eq. 9.

    Args:
        y_pred_means:   mean predictions
        y_pred_std:     std predictions
        y_train:        ground truth
        **kwargs:        to discard additional arguments (allows common interface with other evaluation scores)

    Returns:
        scores: the calibration scores (one for every model in the ensemble)
    """
    distr = torch.distributions.Normal(0, 1)
    y_train_abs = torch.abs((y_train - y_pred_means) / y_pred_stds)
    levels_confidence = torch.linspace(0, 1, 11, device=y_train.device) 
    percentiles = distr.icdf((1 + levels_confidence) / 2) # == - icdf((1-levels_confidence) / 2)
    num_preds = y_train.shape[-2] * y_train.shape[-1]
    levels_confidence_empirical = [(y_train_abs <= p).sum(dim=(1, 2)) / num_preds for p in percentiles]
    levels_confidence_empirical = torch.stack(levels_confidence_empirical, dim=1)
    scores = ((levels_confidence_empirical - levels_confidence)**2).sum(dim=1)
    return scores


def get_score_calibration_mm(y_pred_means, y_pred_stds, y_train, temperature=1.0, **kwargs):
    """Get the calibration score based on symmetric confidence intervals after model aggregation via moment matching. Adapted from [Kuleshov et al.](https://proceedings.mlr.press/v80/kuleshov18a) eq. 9.

    Args:
        y_pred_means:   mean predictions
        y_pred_std:     std predictions
        y_train:        ground truth
        **kwargs:        to discard additional arguments (allows common interface with other evaluation scores)

    Returns:
        score: the calibration score after model aggregation (via moment matching) 
    """
    distr = torch.distributions.Normal(0, 1)
    y_pred_mean, y_pred_std = utils.get_mean_std_mm(y_pred_means, y_pred_stds)
    y_pred_std *= temperature
    y_train_abs = torch.abs((y_train - y_pred_mean) / y_pred_std)
    levels_confidence = torch.linspace(0, 1, 11, device=y_train.device) 
    percentiles = distr.icdf((1 + levels_confidence) / 2) # == - icdf((1-levels_confidence) / 2)
    num_preds = y_train.shape[-2] * y_train.shape[-1]
    levels_confidence_empirical = [(y_train_abs <= p).sum() / num_preds for p in percentiles]
    levels_confidence_empirical = torch.stack(levels_confidence_empirical)
    score = ((levels_confidence_empirical - levels_confidence)**2).sum()
    return score

