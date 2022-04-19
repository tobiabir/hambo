import argparse
import copy
import gym
import numpy as np
import os
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter

import agents
import data
import envs
import evaluation

if __name__ == "__main__":

    torch.set_printoptions(precision=16)

    parser = argparse.ArgumentParser(description="Soft Actor-Critic")
    parser.add_argument("--id_experiment", type=str,
                        help="id of the experiment")
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
    parser.add_argument("--size_batch", type=int, default=256, metavar="N",
                        help="batch size (default: 256)")
    parser.add_argument("--num_steps", type=int, default=4096, metavar="N",
                        help="number of steps (default: 4096)")
    parser.add_argument("--interval_train", type=int, default=128, metavar="N",
                        help="training round interval in steps (default: 128)")
    parser.add_argument("--num_steps_train", type=int, default=512, metavar="N",
                        help="number of steps (default: 512)")
    parser.add_argument("--interval_eval", type=int, default=128, metavar="N",
                        help="evaluation round interval in steps (default: 128)")
    parser.add_argument("--num_episodes_eval", type=int, default=16, metavar="N",
                        help="number of episodes to evaluate (default: 16)")
    parser.add_argument("--device", default="cpu",
                        help="device (default: cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()

    if args.id_experiment is not None:
        dir_log = os.path.join("Logs", "Training")
        dir_log = os.path.join(dir_log, args.id_experiment)
        writer = SummaryWriter(log_dir=dir_log)

    env = gym.make("Pendulum-v1", g=9.81)
    #env = gym.make("CartPole-v1")
    #env = gym.make("MountainCarContinuous-v0")
    #env = envs.EnvPoint()

    # setting rng seeds
    random.seed(args.seed)    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state = env.reset(seed=args.seed)
    
    agent = agents.AgentSAC(env.observation_space, env.action_space, args)

    mem = data.DatasetSARS(capacity=args.replay_size)

    idx_step_episode = 0
    for idx_step in range(args.num_steps):
        agent.train()
        action = agent.get_action(state)
        state_next, reward, done, _ = env.step(action)
        mask = 0. if idx_step_episode == env._max_episode_steps else float(done) 
        mem.append(state, action, reward, state_next, done)
        state = state_next
        idx_step_episode += 1
        if done:
            idx_step_episode = 0
        if (idx_step + 1) % args.interval_train == 0 and len(mem) >= args.size_batch:
            for idx_step_train in range(args.num_steps_train):
                batch = mem.sample(args.size_batch)
                loss_q, loss_pi = agent.step(batch)
        if (idx_step + 1) % args.interval_eval == 0:
            env_eval = copy.deepcopy(env)
            reward_avg = evaluation.evaluate(agent, env_eval, args)
            if args.id_experiment is not None:
                writer.add_scalar("reward", reward_avg, idx_step + 1) 
            print("idx_step: %i, reward: %f" % (idx_step, reward_avg))

