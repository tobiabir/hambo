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

def train_ensemble_map(model, dataset, args):
    model.train()
    utils.preprocess(model, dataset, args.device)
    dataloader = utils.get_dataloader(dataset, args.num_steps_train_model, args.size_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def fn_loss(y_pred_mean, y_pred_std, y_train):
        loss = - torch.distributions.Normal(y_pred_mean, y_pred_std).log_prob(y_train).sum()
        if args.weight_regularizer_model > 0:
            distr_standard_normal = torch.distributions.Normal(0,1)
            for parameter in model.parameters():
                loss -= args.weight_regularizer_model * distr_standard_normal.log_prob(parameter).sum()
        return loss
    for state, action, reward, state_next, done in dataloader:
        x = torch.cat((state, action), dim=-1).to(args.device)
        y_pred_means, y_pred_stds = model(x)
        y = torch.cat((reward, state_next), dim=-1).to(args.device)
        y = model.scaler_y.transform(y)
        loss = fn_loss(y_pred_means, y_pred_stds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
        
def train_sac(agent, env, dataset, args):
    if "interval_eval_agent" in args:
        num_samples = np.gcd(args.interval_train_agent, args.interval_eval_agent)
    else:
        num_samples = args.interval_train_agent
    env.reset()
    idx_step = 0
    while idx_step < args.num_steps_agent:
        agent.train()
        if hasattr(env, "rollout"):
            env.rollout(agent, dataset, num_samples, args.num_steps_rollout_model)
        else:
            rollout.rollout_steps(env, agent, dataset, num_samples)
        idx_step += num_samples
        args.idx_step_agent_global += num_samples
        if idx_step % args.interval_train_agent == 0:
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

