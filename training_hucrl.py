import argparse
import copy
import gym
import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import agents
import data
import envs
import evaluation
import nets
import training

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HUCRL")
    parser.add_argument("--name_env", type=str, choices=["Point", "MountainCar", "Pendulum", "InvertedPendulum", "HalfCheetah"], required=True,
                        help="name of the environment")
    parser.add_argument("--id_experiment", type=str,
                        help="id of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--model", type=str, choices=["GP", "EnsembleDeterministic", "EnsembleProbabilistic"], default="EnsembleProbabilistic",
                        help="what type of model to use to learn dyamics of the environment")
    parser.add_argument("--weight_regularizer_model", type=float, default=0.0,
                        help="regularizer weight lambda of the prior in the map estimate for model training (default: 0.0)")
    parser.add_argument("--hallucinate", default=False, action="store_true",
                        help="set to add hallucinated control (default: False)")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="regularizer weight alpha of the entropy regularization term for sac training (default: 0.05)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--num_steps", type=int, default=4096,
                        help="number of steps (default: 4096)")
    parser.add_argument("--interval_train", type=int, default=128,
                        help="interval of steps after which a round of training is done (default: 128)")
    parser.add_argument("--num_steps_train_model", type=int, default=512,
                        help="number of steps to train model per iteration (default: 512)")
    parser.add_argument("--num_steps_rollout_model", type=int, default=1,
                        help="number of steps to rollout model from initial state (a.k.a. episode length) (default: 1)")
    parser.add_argument("--num_steps_agent", type=int, default=1024,
                        help="number of steps to train agent per iteration (default: 1024)")
    parser.add_argument("--interval_train_agent", type=int, default=128,
                        help="interval of steps after which a round of training is done for agent (default: 128)")
    parser.add_argument("--num_steps_train_agent", type=int, default=128,
                        help="number of steps to train agent per iteration (default: 128)")
    parser.add_argument("--interval_eval", type=int, default=128,
                        help="interval of steps after which a round of evaluation is done (default: 128)")
    parser.add_argument("--num_episodes_eval", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--device", default=None,
                        help="device (default: gpu if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

    if args.name_env == "Point":
        env = envs.EnvPoint()
        env = gym.wrappers.TimeLimit(env, 100)
        env = envs.WrapperEnv(env)
    elif args.name_env == "MountainCar":
        env = gym.make("MountainCarContinuous-v0")
        env = envs.WrapperEnvMountainCar(env)
    elif args.name_env == "Pendulum":
        env = gym.make("Pendulum-v1", g=9.81)
        env = envs.WrapperEnvPendulum(env)
    elif args.name_env == "InvertedPendulum":
        env = gym.make("InvertedPendulum-v2")
        env = envs.WrapperEnvInvertedPendulum(env)
    elif args.name_env == "HalfCheetah":
        env = gym.make("HalfCheetah-v3", exclude_current_positions_from_observation=False)
        env = envs.WrapperEnvHalfCheetah(env)

    dim_action = env.action_space.shape[0]
    
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
    state = env.reset(seed=args.seed)
    
    if args.model == "GP":
        Model = None # TODO
    elif args.model == "EnsembleDeterministic":
        Model = nets.NetDense
    elif args.model == "EnsembleProbabilistic":
        Model = nets.NetGaussHomo
    model = Model(
        dim_x=env.observation_space.shape[0] + env.action_space.shape[0],
        dim_y=env.observation_space.shape[0],
        num_h=2,
        dim_h=256,
        size_ensemble=7
    ).to(args.device)
    if args.hallucinate:
        EnvModel = envs.EnvModelHallucinated
    else:
        EnvModel = envs.EnvModel
    env_model = EnvModel(env.observation_space, env.action_space, None, env.reward, model, env.done, args)
    env_model = gym.wrappers.TimeLimit(env_model, args.num_steps_rollout_model)
    env_model = envs.WrapperEnv(env_model)

    agent = agents.AgentSAC(env_model.observation_space, env_model.action_space, args)

    dataset = data.DatasetSARS(capacity=args.replay_size)
    dataset_agent = data.DatasetSARS(capacity=args.replay_size)
    dataset_states_initial = data.DatasetNumpy()
    dataset_states_initial.append(state)

    for idx_step in range(args.num_steps):
        agent.train()
        action = agent.get_action(state)[:dim_action]
        state_next, reward, done, info = env.step(action)
        mask = float(done and not info["TimeLimit.truncated"]) 
        dataset.push(state, action, reward, state_next, mask)
        state = state_next
        if done:
            state = env.reset()
        dataset_states_initial.append(state)
        if (idx_step + 1) % args.interval_train == 0:
            training.train_ensemble_map(model, dataset, args)
            model.eval()
            env_model = EnvModel(env.observation_space, env.action_space, dataset_states_initial, env.reward, model, env.done, args)
            env_model = gym.wrappers.TimeLimit(env_model, args.num_steps_rollout_model)
            env_model = envs.WrapperEnv(env_model)
            training.train_sac(agent, env_model, dataset_agent, args)
        if (idx_step + 1) % args.interval_eval == 0:
            env_eval = copy.deepcopy(env)
            reward_avg = evaluation.evaluate(agent, env_eval, args.num_episodes_eval)
            if args.id_experiment is not None:
                writer.add_scalar("reward", reward_avg, idx_step + 1) 
            print(f"idx_step: {idx_step}, reward: {reward_avg}")

