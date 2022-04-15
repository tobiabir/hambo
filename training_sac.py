import argparse
import gym
import numpy as np
import random
import time
import torch

import agents
import data
import envs
import rollout

if __name__ == "__main__":

    torch.set_printoptions(precision=16)

    parser = argparse.ArgumentParser(description="Soft Actor-Critic")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G",
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005, metavar="G",
                        help="target smoothing coefficient(τ) (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.05, metavar="G",
                        help="regularizer weight alpha (default: 0.05)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr", type=float, default=0.0003, metavar="G",
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="random seed (default: 42)")
    parser.add_argument("--batch_size", type=int, default=256, metavar="N",
                        help="batch size (default: 256)")
    parser.add_argument("--num_episodes", type=int, default=32, metavar="N",
                        help="number of episodes (default: 32)")
    parser.add_argument("--device", default="cpu",
                        help="device (default: cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()

    #env = gym.make("Pendulum-v1", g=9.81)
    #env = gym.make("CartPole-v1")
    #env = gym.make("MountainCarContinuous-v0")
    env = envs.EnvPoint()

    # setting rng seeds
    random.seed(args.seed)    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    
    agent = agents.AgentSAC(env.observation_space, env.action_space, args)

    mem = data.DatasetSARS(capacity=args.replay_size)

    for idx_episode in range(args.num_episodes):
        mem_episode, _, reward_episode = rollout.rollout_episode(env, agent)
        mem.concat(mem_episode)
        if len(mem) > args.batch_size:
            for idx_step in range(512):
                batch = mem.sample(args.batch_size)
                loss_q, loss_pi = agent.step(batch)
        print("idx_episode: %i, reward_episode: %f" % (idx_episode, reward_episode))

    print("----------")
    print("Evaluation")
    num_episodes_eval = 10
    agent.eval()
    for idx_episode in range(num_episodes_eval):
        state = env.reset()
        done = False

        reward_episode = 0
        idx_step = 0
        while not done:
            action = agent.get_action(state)
            state_next, reward, done, info = env.step(action)
            state = state_next
            reward_episode += reward
            idx_step += 1
        
        print("idx_episode: %i, reward_episode: %f" % (idx_episode, reward_episode))
        
