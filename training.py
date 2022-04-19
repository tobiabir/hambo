import torch

import data
import models
import rollout

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def fn_loss(y_pred, y_train):
        y_train = y_train.repeat(model.size_ensemble, 1, 1)
        losses = torch.nn.functional.mse_loss(y_pred, y_train, reduction="none")
        losses = torch.mean(losses, dim=(1,2))
        loss = torch.sum(losses)
        return loss
    for idx_step in range(args.num_steps_model):
        state, action, reward, state_next, done = dataset.sample(args.size_batch)
        state_action = torch.cat((state, action), dim=-1)
        state_next_pred = model(state_action)
        loss = fn_loss(state_next_pred, state_next)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_ensemble_map(model, dataset, args):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def fn_loss(y_pred_mean, y_pred_std, y_true):
        loss = - torch.distributions.Normal(y_pred_mean, y_pred_std).log_prob(y_true).sum()
        for parameter in model.parameters():
            loss -= torch.sum(parameter**2)
        return loss
    for idx_step in range(args.num_steps_model):
        state, action, reward, state_next, done = dataset.sample(args.size_batch)
        state_action = torch.cat((state, action), dim=-1)
        state_next_means, state_next_stds = model(state_action)
        loss = fn_loss(state_next_means, state_next_stds, state_next)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
        
def train_sac(agent, env, args):

    mem = data.DatasetSARS(capacity=args.replay_size)

    for idx_episode in range(args.num_episodes_agent):
        mem_episode, _, reward_episode = rollout.rollout_episode(env, agent)
        mem.concat(mem_episode)
        if len(mem) > args.size_batch:
            for idx_step in range(128):
                batch = mem.sample(args.size_batch)
                loss_q, loss_pi = agent.step(batch)
    
    return agent

