import argparse
import copy
import gym
import numpy as np
import os
import random
import time
import torch
import wandb

import agents
import data
import envs
import evaluation
import training
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Soft Actor-Critic")
    parser.add_argument("--name_env", type=str, choices=["Point", "MountainCar", "Pendulum", "InvertedPendulum", "Swimmer", "Hopper", "HalfCheetah"], required=True,
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
    parser.add_argument("--num_steps_startup", type=int, default=0,
                        help="number of steps of rollout to do during startup (default: 0)")
    parser.add_argument("--num_steps_agent", type=int, default=4096,
                        help="number of steps (default: 4096)")
    parser.add_argument("--interval_train_agent_internal", type=int, default=128,
                        help="training round interval in steps (default: 128)")
    parser.add_argument("--num_steps_train_agent", type=int, default=128,
                        help="number of steps (default: 128)")
    parser.add_argument("--interval_eval_agent", type=int, default=128,
                        help="evaluation round interval in steps (default: 128)")
    parser.add_argument("--num_episodes_eval_agent", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--device", default=None,
                        help="device (default: gpu if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()
    args.algorithm = "SAC"
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

    wandb.init(project="Master Thesis", entity="tobiabir", config=args)

    if args.name_env == "Point":
        env = envs.EnvPoint()
        env = gym.wrappers.TimeLimit(env, 100)
        env = envs.WrapperEnv(env)
    elif args.name_env == "MountainCar":
        env = gym.make("MountainCarContinuous-v0")
        env = envs.WrapperEnvMountainCar(env)
    elif args.name_env == "Pendulum":
        env = gym.make("Pendulum-v1")
        env = envs.WrapperEnvPendulum(env)
    elif args.name_env == "InvertedPendulum":
        env = gym.make("InvertedPendulum-v2")
        env = envs.WrapperEnvInvertedPendulum(env)
    elif args.name_env == "Swimmer":
        env = gym.make("Swimmer-v3")
        env = envs.WrapperEnvSwimmer(env)
    elif args.name_env == "Hopper":
        env = gym.make("Hopper-v3")
        env = envs.WrapperEnvHopper(env)
    elif args.name_env == "HalfCheetah":
        env = gym.make("HalfCheetah-v3")
        env = envs.WrapperEnvHalfCheetah(env)

    args.idx_step_agent_global = 0

    # setting rng seeds
    utils.set_seeds(args.seed)
    
    agent = agents.AgentSAC(env.observation_space, env.action_space, args)

    dataset = data.DatasetSARS(capacity=args.replay_size)

    env.reset()

    utils.startup(env, agent, dataset, None, args.num_steps_startup)
    args.idx_step_agent_global += args.num_steps_rollout_startup

    training.train_sac(agent, env, dataset, args, args.num_steps_rollout_startup)

