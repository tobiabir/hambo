import argparse
import gym
import numpy as np
import random
import torch

import agents
import data
import envs
import nets
import rollout
import training

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ensemble Model")
    parser.add_argument("--lr", type=float, default=0.003, metavar="G",
                        help="learning rate (default: 0.003)")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="random seed (default: 42)")
    parser.add_argument("--size_batch", type=int, default=256, metavar="N",
                        help="batch size (default: 256)")
    parser.add_argument("--num_episodes", type=int, default=32, metavar="N",
                        help="number of episodes (default: 32)")
    parser.add_argument("--num_steps_model", type=int, default=1024, metavar="N",
                        help="number of steps to train model (default: 1024)")
    parser.add_argument("--device", default="cpu",
                        help="device (default: cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()

    #env = gym.make("Pendulum-v1", g=9.81)
    #env = gym.make("MountainCarContinuous-v0")
    env = envs.EnvPoint()

    # setting rng seeds
    random.seed(args.seed)    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    
    agent = agents.AgentRandom(env.action_space)

    mem = data.DatasetSARS(capacity=args.replay_size)

    for idx_episode in range(args.num_episodes):
        mem_episode, _, reward_episode = rollout.rollout_episode(env, agent)
        mem.concat(mem_episode)
    
    dim_x = env.observation_space.shape[0] + env.action_space.shape[0]
    dim_y = env.observation_space.shape[0]
    dim_h = 16
    num_h = 2
    size_ensemble = 8
    model = nets.NetGaussHomo(dim_x, dim_y, num_h, dim_h, size_ensemble)

    model = training.train_ensemble_map(model, mem, args)

    state, action, reward, state_next, done = mem.sample(1)
    state_action = torch.cat((state, action), dim=-1)
    mean, std = model(state_action)
    print(state_action)
    print(mean)
    print(std)

