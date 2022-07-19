import argparse
import gym
import numpy as np
import random
import torch

import agents
import envs
import rollout


def evaluate_agent(agent, env, num_episodes):
    """Evaluate an agent

    Args:
        agent:          the agent to evaluate
        env:            the environment to evaluate on
        num_episodes:   the number of episodes to evaluate for

    Returns:
        reward_avg:     average cumulative reward seen in the episodes
    """
    agent.eval()
    reward = 0
    for idx_episode in range(num_episodes):
        _, _, reward_episode = rollout.rollout_episode(env, agent)
        reward += reward_episode
    reward_avg = reward / num_episodes
    return reward_avg


def evaluate_model(model, dataset, fns_eval, device):
    """Evaluate a model

    Args:
        model:      the model to evaluate
        dataset:    the dataset to evaluate on
        fns_eval:   list of evaluation functions
        device:     the device to use

    Returns:
        scores:     list of the evaluation scores (one for every evaluation function)
    """
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
    model.eval()
    with torch.no_grad():
        state, action, reward, state_next, done = next(iter(dataloader))
        x = torch.cat((state, action), dim=-1)[:, :model.dim_x].to(device)
        y_pred_means, y_pred_stds = model(x)
        y = torch.cat((reward, state_next - state), dim=-1).to(device)
        y = model.scaler_y.transform(y)
        scores = [fn_eval(y_pred_means, y_pred_stds, y, state, action) for fn_eval in fns_eval]
    return scores

