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

def train_ensemble(model, dataset, args):
    model.train()
    dataloader = utils.get_dataloader(dataset, args.num_steps_train_model, args.size_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def fn_loss(y_pred, y_train):
        y_train = y_train.repeat(model.size_ensemble, 1, 1)
        losses = torch.nn.functional.mse_loss(y_pred, y_train, reduction="none")
        losses = torch.mean(losses, dim=(1,2))
        loss = torch.sum(losses)
        return loss
    for state, action, reward, state_next, done in dataloader:
        state_action = torch.cat((state, action), dim=-1).to(args.device)
        state_next_pred = model(state_action)
        state_next = state_next.to(args.device)
        loss = fn_loss(state_next_pred, state_next)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

#class TrainerModel(Trainer):
#    
#    def __init__(self, model)
#        self.model = model
#        self.fn_loss = fn_loss_map 
#

    

def train_ensemble_map(model, dataset, args):
    model.train()
    len_train = int(0.8 * len(dataset))
    len_eval = len(dataset) - len_train
    print(len_train, len_eval)
    dataset_train, dataset_eval = torch.utils.data.random_split(dataset, [len_train, len_eval])
    utils.preprocess(model, dataset_train, args.device)
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.size_batch,
        shuffle=True,
        num_workers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model)

    def fn_loss(y_pred_mean, y_pred_std, y_train, weight_regularizer=0.0):
        loss = - torch.distributions.Normal(y_pred_mean, y_pred_std).log_prob(y_train).sum(dim=2).mean(dim=1)
        if weight_regularizer > 0:
            distr_standard_normal = torch.distributions.Normal(0,1)
            for parameter in model.parameters():
                loss -= weight_regularizer * distr_standard_normal.log_prob(parameter).sum(dim=(1,2))
        return loss

    losses_eval_best = evaluation.evaluate_model(model, dataset_eval, fn_loss, args.device)
    state_dicts_best = [model.state_dict()] * model.size_ensemble
    idxs_epoch_best = - np.ones(model.size_ensemble)
    idx_epoch_curr = 0
    idx_step = 0
    while (idx_epoch_curr - 5 <= idxs_epoch_best).any():
        model.train()
        for state, action, reward, state_next, done in dataloader_train:
            x = torch.cat((state, action), dim=-1).to(args.device)
            y_pred_means, y_pred_stds = model(x)
            y = torch.cat((reward, state_next), dim=-1).to(args.device)
            y = model.scaler_y.transform(y)
            loss = fn_loss(y_pred_means, y_pred_stds, y, args.weight_regularizer_model).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx_step += 1
        losses_eval_curr = evaluation.evaluate_model(model, dataset_eval, fn_loss, args.device)
        improvement = (losses_eval_best - losses_eval_curr) / losses_eval_best
        state_dict = model.state_dict()
        for idx_model in range(model.size_ensemble):
            if 0.001 < improvement[idx_model]:
                losses_eval_best[idx_model] = losses_eval_curr[idx_model]
                state_dicts_best[idx_model] = state_dict
                idxs_epoch_best[idx_model] = idx_epoch_curr
        idx_epoch_curr += 1
    
    for idx_model in range(model.size_ensemble):
        model.load_state_dict_single(state_dicts_best[idx_model], idx_model)
    loss_model = losses_eval_best.mean()
    wandb.log({"loss_model": loss_model, "idx_step": args.idx_step})
    print(f"idx_step: {args.idx_step}, loss: {loss_model}")

    return model
        
def train_sac(agent, env, dataset, args, idx_step_start=0):
    if "interval_eval_agent" in args:
        num_samples = np.gcd(args.interval_train_agent_internal, args.interval_eval_agent)
    else:
        num_samples = args.interval_train_agent
    env.reset()
    idx_step = idx_step_start
    while idx_step < args.num_steps_agent:
        agent.train()
        if hasattr(env, "rollout"):
            env.rollout(agent, dataset, num_samples, args.num_steps_rollout_model)
        else:
            rollout.rollout_steps(env, agent, dataset, None, num_samples)
        idx_step += num_samples
        args.idx_step_agent_global += num_samples
        if idx_step % args.interval_train_agent_internal == 0:
            dataloader = utils.get_dataloader(dataset, args.num_steps_train_agent, args.size_batch)
            for batch in dataloader:
                loss_q, loss_pi, loss_alpha = agent.step(batch)
            wandb.log({"loss_critic": loss_q, "loss_actor": loss_pi, "alpha": args.alpha, "loss_alpha": loss_alpha, "idx_step": args.idx_step_agent_global - 1})
            print(f"idx_step_agent_global: {args.idx_step_agent_global}, idx_step_agent: {idx_step}, loss_q: {loss_q}, loss_pi: {loss_pi}, alpha: {agent.alpha}, loss_alpha: {loss_alpha}")
        if "interval_eval_agent" in args and idx_step % args.interval_eval_agent == 0:
            env_eval = copy.deepcopy(env)
            reward_avg = evaluation.evaluate(agent, env_eval, args.num_episodes_eval_agent)
            wandb.log({"reward": reward_avg, "idx_step": args.idx_step_agent_global - 1})
            print(f"idx_step_agent: {idx_step}, reward: {reward_avg}")

    return agent

