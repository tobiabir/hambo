import copy
import gpytorch
import numpy as np
import torch
import wandb

import data
import evaluation
import models
import rollout
import utils


def train_gp(dataset):
    # set up training data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    state, action, reward, state_next, done = next(iter(dataloader))
    x = torch.cat((state, action), dim=-1)
    y = torch.cat((reward, state_next - state), dim=-1)

    # create model
    model = models.ModelGP(x, y)

    # train model
    model.train()
    for idx_dim_y in range(y.shape[1]):
        model_curr = model.models[idx_dim_y]
        optimizer = torch.optim.Adam(model_curr.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model_curr)
        for idx_step in range(100):
            x_train = model_curr.train_inputs[0]
            y_train = model_curr.train_targets
            y_pred = model_curr(x_train)
            loss = - mll(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, None, None


def get_fn_loss_map(model, weight_prior):
    """Get MAP loss function using weight_prior as weight on the prior of the posterior
    """
    def fn_loss(y_pred_mean, y_pred_std, y_train, state, action):
        loss_mle = - torch.distributions.Normal(y_pred_mean, y_pred_std).log_prob(y_train).sum(dim=2).mean(dim=1)
        distr_prior = torch.distributions.Normal(0, 1)
        loss_prior = torch.zeros(loss_mle.shape, device=loss_mle.device)
        for name, parameter in model.named_parameters():
            if "weight" in name:
                loss_prior -= weight_prior * distr_prior.log_prob(parameter).sum(dim=(1, 2))
        loss = loss_prior + loss_mle
        return loss
    return fn_loss


def train_ensemble_map(model, dataset, weight_prior, lr, size_batch, device):
    """Train an ensemble using the posterior as loss function.

    Args:
        weight_prior:   the weight on the prior in the posterior

    Rest:
        see train_ensemble
    """
    fn_loss = get_fn_loss_map(model, weight_prior)
    return train_ensemble(model, dataset, fn_loss, lr, size_batch, device)


def train_ensemble_adversarial(model, dataset, agent, model_termination, weight_prior, gamma, weight_adversarial, lr, size_batch, device):
    """Train an ensemble using the adversarial loss (see [RAMBO](https://arxiv.org/abs/2204.12581) eq. 5 and 9).

    Args:
        weight_prior:       the weight on the prior in the posterior
        model_termination:  termination model to use
        gamma:              discount factor to use
        weight_adversarial: weight on the adversarial loss

    Rest:
        see train_ensemble
    """
    fn_loss_map = get_fn_loss_map(model, weight_prior)

    def fn_loss(y_pred_mean, y_pred_std, y_train, state, action):
        loss_map = fn_loss_map(y_pred_mean, y_pred_std, y_train, state, action)
        distr_y_pred = torch.distributions.Normal(y_pred_mean, y_pred_std)
        with torch.no_grad():
            y_pred = distr_y_pred.rsample()
            reward = y_pred[:, :, :1]
            state_diff = y_pred[:, :, 1:]
            state_next = state + state_diff
            # Note: no clamp because gradients would just become zero
            terminal = torch.tensor(model_termination(
                state_next.cpu().numpy()), dtype=torch.float32, device=state_next.device)
            actions_next = [agent.policy(state_next[i])[0] for i in range(model.size_ensemble)]
            action_next = torch.stack(actions_next)
            q = torch.min(*agent.critic(state, action))
            qs_next =[torch.min(*agent.critic(state_next[i], action_next[i])) for i in range(model.size_ensemble)]
            q_next = torch.stack(qs_next)
            q_pred = reward + gamma * q_next * terminal
            advantage = q - q_pred
            advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
            advantage_std = torch.std(advantage, dim=1, keepdim=True)
            advantage = (advantage - advantage_mean) / advantage_std
        y_pred_prob_log = distr_y_pred.log_prob(y_pred)
        loss_adversarial = advantage * y_pred_prob_log.sum(dim=2, keepdim=True)
        return loss_map + weight_adversarial * loss_adversarial.mean(dim=[1,2])
    return train_ensemble(model, dataset, fn_loss, lr, size_batch, device)


def train_ensemble(model, dataset, fn_loss, lr, size_batch, device):
    """Train an ensemble until for five epochs none of the models improves on the evaluation set.

    Args:
        model:      the model to train
        dataset:    the dataset to train on
        fn_loss:    the loss function to use
        lr:         the learning rate to use
        size_batch: the batch size to use
        device:     the device to use

    Returns:
        losses_eval_best:           the evaluation losses of the final models
        scores_calibration_eval:    the calibration scores of the final models
    """
    model.train()
    len_train = int(0.9 * len(dataset))
    len_eval = len(dataset) - len_train
    dataset_train, dataset_eval = torch.utils.data.random_split(
        dataset, [len_train, len_eval])
    data.preprocess(model, dataset_train, device)
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=size_batch,
        shuffle=True,
        num_workers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses_eval_best = evaluation.evaluate_model(model, dataset_eval, [fn_loss], device)[0]
    state_dicts_best = [model.layers.state_dict()] * model.size_ensemble
    idxs_epoch_best = - np.ones(model.size_ensemble)
    idx_epoch_curr = 0
    idx_step = 0
    while (idx_epoch_curr - 5 <= idxs_epoch_best).any():
        model.train()
        for state, action, reward, state_next, done in dataloader_train:
            x = torch.cat((state, action), dim=-1)[:, :model.dim_x].to(device)
            y_pred_means, y_pred_stds = model(x)
            y = torch.cat((reward, state_next - state), dim=-1).to(device)
            y = model.scaler_y.transform(y)
            loss = fn_loss(y_pred_means, y_pred_stds, y, state, action).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx_step += 1
        losses_eval_curr = evaluation.evaluate_model(model, dataset_eval, [fn_loss], device)[0]
        state_dict = model.layers.state_dict()
        for idx_model in range(model.size_ensemble):
            if losses_eval_curr[idx_model] < losses_eval_best[idx_model]:
                losses_eval_best[idx_model] = losses_eval_curr[idx_model]
                state_dicts_best[idx_model] = state_dict
                idxs_epoch_best[idx_model] = idx_epoch_curr
        idx_epoch_curr += 1

    for idx_model in range(model.size_ensemble):
        model.load_state_dict_single(state_dicts_best[idx_model], idx_model)

    model.idxs_elites = torch.argsort(losses_eval_best)[:model.num_elites]

    scores_calibration_eval = evaluation.evaluate_model(
        model, dataset_eval, [utils.get_scores_calibration], device)[0]

    return losses_eval_best, scores_calibration_eval
