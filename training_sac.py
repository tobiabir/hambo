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
import training

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Soft Actor-Critic")
    parser.add_argument("--name_env", type=str, choices=["Point", "MountainCar", "Pendulum", "HalfCheetah"], required=True,
                        help="name of the environment")
    parser.add_argument("--id_experiment", type=str,
                        help="id of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="regularizer weight alpha (default: 0.05)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--num_steps_agent", type=int, default=4096,
                        help="number of steps (default: 4096)")
    parser.add_argument("--interval_train_agent", type=int, default=128,
                        help="training round interval in steps (default: 128)")
    parser.add_argument("--num_steps_train_agent", type=int, default=128,
                        help="number of steps (default: 128)")
    parser.add_argument("--interval_eval_agent", type=int, default=128,
                        help="evaluation round interval in steps (default: 128)")
    parser.add_argument("--num_episodes_eval_agent", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--device", default="cpu",
                        help="device (default: cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()

    if args.name_env == "Point":
        env = envs.EnvPoint()
    elif args.name_env == "MountainCar":
        env = gym.make("MountainCarContinuous-v0")
    elif args.name_env == "Pendulum":
        env = gym.make("Pendulum-v1", g=9.81)
        env = envs.WrapperEnvPendulum(env)
    elif args.name_env == "HalfCheetah":
        env = gym.make("HalfCheetah-v3", exclude_current_positions_from_observation=False)
        env = envs.WrapperEnvHalfCheetah(env)

    if args.id_experiment is not None:
        dir_log = os.path.join("Logs", "Training")
        dir_log = os.path.join(dir_log, args.name_env)
        dir_log = os.path.join(dir_log, args.id_experiment)
        writer = SummaryWriter(log_dir=dir_log)
        args.writer = writer
    args.idx_step_agent_global = 0

    # setting rng seeds
    random.seed(args.seed)    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    
    agent = agents.AgentSAC(env.observation_space, env.action_space, args)

    dataset = data.DatasetSARS(capacity=args.replay_size)

    training.train_sac(agent, env, dataset, args)

