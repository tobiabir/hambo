import argparse
import copy
import gym
import numpy as np
import os
import random
import torch
import wandb

import agents
import data
import envs
import evaluation
import nets
import training
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HUCRL")
    parser.add_argument("--name_env", type=str, choices=["Point", "MountainCar", "Pendulum", "InvertedPendulum", "Swimmer", "Hopper", "HalfCheetah"], required=True,
                        help="name of the environment")
    parser.add_argument("--id_experiment", type=str,
                        help="id of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--model", type=str, choices=["GP", "EnsembleDeterministic", "EnsembleProbabilistic"], default="EnsembleProbabilistic",
                        help="what type of model to use to learn dyamics of the environment (default: EnsembleProbabilistic)")
    parser.add_argument("--num_h_model", type=int, default=2,
                        help="number of hidden layers in model (only for ensembles) (default: 2)")
    parser.add_argument("--dim_h_model", type=int, default=256,
                        help="dimension of hidden layers in model (only for ensembles) (default: 256)")
    parser.add_argument("--size_ensemble_model", type=int, default=7,
                        help="number of networks in model (only for ensembles) (default: 7)")
    parser.add_argument("--weight_regularizer_model", type=float, default=0.0,
                        help="regularizer weight lambda of the prior in the map estimate for model training (default: 0.0)")
    parser.add_argument("--use_true_reward", default=False, action="store_true",
                        help="set to use true reward instead of learned by model (default: False)")
    parser.add_argument("--use_aleatoric", default=False, action="store_true",
                        help="set to use aleatoric noise from transition model (default: False)")
    parser.add_argument("--hallucinate", default=False, action="store_true",
                        help="set to add hallucinated control (default: False)")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="regularizer weight alpha of the entropy regularization term for sac training (default: 0.05)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr_model", type=float, default=0.0001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--num_steps", type=int, default=4096,
                        help="number of steps (default: 4096)")
    parser.add_argument("--num_steps_startup", type=int, default=0,
                        help="number of steps of rollout to do during startup (default: 0)")
    parser.add_argument("--interval_train_model", type=int, default=128,
                        help="interval of steps after which a round of training is done (default: 128)")
    parser.add_argument("--num_steps_rollout_model", type=int, default=1,
                        help="number of steps to rollout model from initial state (a.k.a. episode length) (default: 1)")
    parser.add_argument("--num_steps_agent", type=int, default=1024,
                        help="number of steps to train agent per iteration (default: 1024)")
    parser.add_argument("--interval_train_agent", type=int, default=128,
                        help="interval of steps after which a round of training is done for agent (default: 128)")
    parser.add_argument("--num_steps_train_agent", type=int, default=128,
                        help="number of steps to train agent per iteration (default: 128)")
    parser.add_argument("--interval_train_agent_internal", type=int, default=128,
                        help="interval of steps (from model env) after which a round of training is done for agent (default: 128)")
    parser.add_argument("--interval_eval", type=int, default=128,
                        help="interval of steps after which a round of evaluation is done (default: 128)")
    parser.add_argument("--num_episodes_eval", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--device", default=None,
                        help="device (default: gpu if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()
    if args.hallucinate:
        args.algorithm = "HMBSAC"
    else:
        args.algorithm = "MBSAC"
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

    dim_action = env.action_space.shape[0]
    
    args.idx_step = 0
    args.idx_step_agent_global = 0

    # setting rng seeds
    utils.set_seeds(args.seed)
    
    if args.model == "GP":
        Model = None # TODO
    elif args.model == "EnsembleDeterministic":
        Model = nets.NetDense
    elif args.model == "EnsembleProbabilistic":
        Model = nets.NetGaussHomo
    model = Model(
        dim_x=env.observation_space.shape[0] + env.action_space.shape[0],
        dim_y=1 + env.observation_space.shape[0],
        num_h=args.num_h_model,
        dim_h=args.dim_h_model,
        size_ensemble=args.size_ensemble_model
    ).to(args.device)
    if args.hallucinate:
        EnvModel = envs.EnvModelHallucinated
    else:
        EnvModel = envs.EnvModel
    env_model = EnvModel(env.observation_space, env.action_space, None, env.reward, model, env.done, args)
    env_model = gym.wrappers.TimeLimit(env_model, args.num_steps_rollout_model)
    env_model = envs.WrapperEnv(env_model)
    is_trained_model = False

    agent_random = agents.AgentRandom(env.action_space)
    agent = agents.AgentSAC(env_model.observation_space, env_model.action_space, args)

    dataset = data.DatasetSARS(capacity=args.replay_size)
    dataset_agent = data.DatasetSARS(capacity=args.replay_size)
    dataset_states_initial = data.DatasetNumpy()

    state = env.reset()
    dataset_states_initial.append(state)

    utils.startup(env, agent, dataset, dataset_states_initial, args.num_steps_startup) 
    args.idx_step += args.num_steps_startup

    for idx_step in range(args.num_steps_startup, args.num_steps):
        agent.train()
        action = agent.get_action(state)[:dim_action]
        state_next, reward, done, info = env.step(action)
        mask = float(done and not info["TimeLimit.truncated"]) 
        dataset.push(state, action, reward, state_next, mask)
        state = state_next
        if done:
            state = env.reset()
        dataset_states_initial.append(state)
        if (idx_step + 1) % args.interval_train_model == 0:
            training.train_ensemble_map(model, dataset, args)
            is_trained_model = True
        if is_trained_model and (idx_step + 1) % args.interval_train_agent == 0:
            model.eval()
            env_model = EnvModel(env.observation_space, env.action_space, dataset_states_initial, env.reward, model, env.done, args)
            env_model = gym.wrappers.TimeLimit(env_model, args.num_steps_rollout_model)
            env_model = envs.WrapperEnv(env_model)
            training.train_sac(agent, env_model, dataset_agent, args)
        if (idx_step + 1) % args.interval_eval == 0:
            env_eval = copy.deepcopy(env)
            reward_avg = evaluation.evaluate(agent, env_eval, args.num_episodes_eval)
            wandb.log({"reward": reward_avg, "idx_step": idx_step})
            print(f"idx_step: {idx_step}, reward: {reward_avg}")
        args.idx_step = idx_step

