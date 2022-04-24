import copy
import torch

import data
import evaluation
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
    for idx_step in range(args.num_steps_train_model):
        state, action, reward, state_next, done = dataset.sample(args.size_batch)
        state_action = torch.cat((state, action), dim=-1)
        state_next_pred = model(state_action)
        loss = fn_loss(state_next_pred, state_next)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)
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
        
def train_sac(agent, env, dataset, args):

    state = env.reset()

    idx_step_episode = 0
    for idx_step in range(args.num_steps_agent):
        agent.train()
        action = agent.get_action(state)
        state_next, reward, done, _ = env.step(action)
        mask = 0. if idx_step_episode + 1 == env.max_steps_episode else float(done) 
        dataset.append(state, action, reward, state_next, mask)
        state = state_next
        idx_step_episode += 1
        if done:
            idx_step_episode = 0
            state = env.reset()
        if (idx_step + 1) % args.interval_train_agent == 0:
            for idx_step_train in range(args.num_steps_train_agent):
                batch = dataset.sample(args.size_batch)
                loss_q, loss_pi, loss_alpha = agent.step(batch)
                args.idx_step_train_agent_global += 1
            print(f"idx_step_train_agent_global: {args.idx_step_train_agent_global}, idx_step_agent: {idx_step}, loss_q: {loss_q}, loss_pi: {loss_pi}, alpha: {agent.alpha}, loss_alpha: {loss_alpha}")
        if "interval_eval_agent" in args and (idx_step + 1) % args.interval_eval_agent == 0:
            env_eval = copy.deepcopy(env)
            reward_avg = evaluation.evaluate(agent, env_eval, args.num_episodes_eval_agent)
            if args.id_experiment is not None:
                args.writer.add_scalar("reward", reward_avg, args.idx_step_train_agent_global + 1) 
            print(f"idx_step_agent: {idx_step}, reward: {reward_avg}")

    return agent

