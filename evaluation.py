import argparse
import gym
import numpy as np
import random
import torch

import agents
import envs
import rollout

def evaluate_agent(agent, env, num_episodes):
    agent.eval()
    reward = 0
    for idx_episode in range(num_episodes):
        _, _, reward_episode = rollout.rollout_episode(env, agent)
        reward += reward_episode
    reward_avg = reward / num_episodes
    return reward_avg

def evaluate_model(model, dataset, fns_eval, device):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
    model.eval()
    with torch.no_grad():
        state, action, reward, state_next, done = next(iter(dataloader))
        x = torch.cat((state, action), dim=-1)[:, :model.dim_x].to(device)
        y_pred_means, y_pred_stds = model(x)
        y = torch.cat((reward, state_next - state), dim=-1).to(device)
        y = model.scaler_y.transform(y)
        scores = [fn_eval(y_pred_means, y_pred_stds, y) for fn_eval in fns_eval]
    return scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HUCRL")
    parser.add_argument("--num_episodes_eval", type=int, default=32, metavar="N",
                        help="number of episodes (default: 32)")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="random seed (default: 42)")
    args = parser.parse_args()

    env = envs.EnvPoint()

    # setting rng seeds
    random.seed(args.seed)    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    agent = agents.AgentPointOptimal()

    evaluate(agent, env, args)
