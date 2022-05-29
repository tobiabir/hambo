import copy
import numpy as np
import torch
import wandb

import data
import evaluation
import models
import rollout
import utils

def train_gp(model, dataset, args):
    state, action, reward, state_next, done = dataset.sample(len(dataset)) 
    state_action = torch.cat((state, action), dim=-1)
    model = models.ModelGP(state_action, state_next)
    model.train()
    for idx_step in range(args.num_steps_model):
        model.step() 
    return model


def train_ensemble_map(model, dataset, args):
    model.train()
    len_train = int(0.9 * len(dataset))
    len_eval = len(dataset) - len_train
    dataset_train, dataset_eval = torch.utils.data.random_split(dataset, [len_train, len_eval])
    utils.preprocess(model, dataset_train, args.device)
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.size_batch,
        shuffle=True,
        num_workers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model)

    def fn_loss(y_pred_mean, y_pred_std, y_train, scale_prior_weight=0.0):
        loss_mle = - torch.distributions.Normal(y_pred_mean, y_pred_std).log_prob(y_train).sum(dim=2).mean(dim=1)
        distr_prior = torch.distributions.Normal(0, 1)
        loss_prior = torch.zeros(loss_mle.shape)
        for name, parameter in model.named_parameters():
            if "weight" in name:
                loss_prior -= scale_prior_weight * distr_prior.log_prob(parameter).sum(dim=(1,2))
        loss = loss_prior + loss_mle
        return loss

    losses_eval_best = evaluation.evaluate_model(model, dataset_eval, [fn_loss], args.device)[0]
    state_dicts_best = [model.layers.state_dict()] * model.size_ensemble
    idxs_epoch_best = - np.ones(model.size_ensemble)
    idx_epoch_curr = 0
    idx_step = 0
    while (idx_epoch_curr - 5 <= idxs_epoch_best).any():
        model.train()
        for state, action, reward, state_next, done in dataloader_train:
            x = torch.cat((state, action), dim=-1).to(args.device)
            y_pred_means, y_pred_stds = model(x)
            y = torch.cat((reward, state_next - state), dim=-1).to(args.device)
            y = model.scaler_y.transform(y)
            loss_map = fn_loss(y_pred_means, y_pred_stds, y, args.weight_regularizer_model).sum()
            loss = loss_map + 0.01 * model.std_log_max.sum() - 0.01 * model.std_log_min.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx_step += 1
        losses_eval_curr = evaluation.evaluate_model(model, dataset_eval, [fn_loss], args.device)[0]
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

    scores_calibration_eval = evaluation.evaluate_model(model, dataset_eval, [utils.get_scores_calibration], args.device)[0]

    return losses_eval_best, scores_calibration_eval
        
def train_sac(agent, env, dataset_env, dataset_model, args, idx_step_start=0):
    if "interval_eval_agent" in args:
        num_samples = np.gcd(args.interval_train_agent_internal, args.interval_eval_agent)
    else:
        num_samples = args.interval_train_agent
    env.reset()
    idx_step = idx_step_start
    while idx_step < args.num_steps_agent:
        agent.train()
        if hasattr(env, "rollout"):
            env.rollout(agent, dataset_model, num_samples, args.num_steps_rollout_model)
        else:
            rollout.rollout_steps(env, agent, dataset_model, None, num_samples)
        idx_step += num_samples
        args.idx_step_agent_global += num_samples
        if idx_step % args.interval_train_agent_internal == 0:
            dataset = torch.utils.data.ConcatDataset((dataset_env, dataset_model))
            dataloader = utils.get_dataloader(dataset, args.num_steps_train_agent, args.size_batch)
            for batch in dataloader:
                loss_pi, loss_q, loss_alpha = agent.step(batch)
            wandb.log({"loss_critic": loss_q, "loss_actor": loss_pi, "alpha": args.alpha, "loss_alpha": loss_alpha, "idx_step": args.idx_step_agent_global - 1})
            print(f"idx_step_agent_global: {args.idx_step_agent_global}, idx_step_agent: {idx_step}, loss_q: {loss_q}, loss_pi: {loss_pi}, alpha: {agent.alpha}, loss_alpha: {loss_alpha}")
        if "interval_eval_agent" in args and idx_step % args.interval_eval_agent == 0:
            env_eval = copy.deepcopy(env)
            reward_avg = evaluation.evaluate(agent, env_eval, args.num_episodes_eval_agent)
            wandb.log({"reward": reward_avg, "idx_step": args.idx_step_agent_global - 1})
            print(f"idx_step_agent: {idx_step}, reward: {reward_avg}")

    return agent

