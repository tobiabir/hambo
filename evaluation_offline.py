import argparse
import d4rl
import gym
import json
import numpy as np
import os
import pickle
import tensorflow as tf
import torch
import wandb

import agents
import data
import envs
import evaluation
import models
import training
import utils

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Offline Policy Evaluation")

    # dataset arguments
    parser.add_argument("--names_dataset", type=str, nargs="+", required=True,
                        help="names of the dataset to use")

    # model arguments
    parser.add_argument("--path_checkpoint_model", type=str, default=None,
                        help="path to model checkpoint (default: None (no checkpointing))")

    # model environment arguments
    parser.add_argument("--use_gauss_approx", default=False, action="store_true",
                        help="set to use gauss approximation instead of true mixture to sample from transition model (default: False)")
    parser.add_argument("--use_aleatoric", default=False, action="store_true",
                        help="set to use aleatoric noise from transition model (default: False)")
    parser.add_argument("--weight_penalty_reward", type=float, default=0.0,
                        help="weight on the reward penalty (see MOPO) (default: 0.0)")
    parser.add_argument("--hallucinate", default=False, action="store_true",
                        help="set to add hallucinated control (default: False)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="parameter for the amount of hallucinated control (only has effect if hallucinate is set) (default: 1.0)")

    # SAC arguments
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="regularizer weight alpha of the entropy regularization term for sac training (default: 0.1)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr_agent", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--conservative", default=False, action="store_true",
                        help="set to add conservative Q learning (default: False)")

    # procedure arguments
    parser.add_argument("--ratio_env_model", type=float, default=0.05,
                        help="ratio of env data to model data in agent batches (default: 0.05)")
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=128,
                        help="number of episodes to train (default: 128)")
    parser.add_argument("--interval_rollout_model", type=int, default=128,
                        help="interval of steps after which a round of training is done (default: 128)")
    parser.add_argument("--num_steps_rollout_model", type=int, default=128,
                        help="number of steps to rollout model in a round of rollout (default: 128)")
    parser.add_argument("--max_length_rollout_model", type=int, default=1,
                        help="number of steps to rollout model from initial state (a.k.a. episode length) (default: 1)")
    parser.add_argument("--num_steps_train_agent", type=int, default=128,
                        help="number of steps to train agent per iteration (default: 128)")
    parser.add_argument("--interval_eval_agent", type=int, default=128,
                        help="interval of steps after which a round of evaluation is done (default: 128)")
    parser.add_argument("--num_episodes_eval", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--device", default=None,
                        help="device (default: cuda if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=1000000,
                        help="capacity of replay buffer (default: 1000000)")
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

    # setting rng seeds
    utils.set_seeds(args.seed)

    # check if all datasets use the same environment
    names_env = [name_dataset.split("-")[0] for name_dataset in args.names_dataset]
    name_env = names_env[0]
    assert all(name_env == name_env_curr for name_env_curr in names_env), "some datasets were generated from different environments"

    # set up environment
    env = gym.make(args.names_dataset[0])
    if "halfcheetah" in name_env:
        env = envs.WrapperEnvHalfCheetah(env)
    elif "hopper" in name_env:
        env = envs.WrapperEnvHopper(env)
    elif "maze" in name_env:
        env = envs.WrapperEnvMaze(env)
    elif "walker" in name_env:
        env = envs.WrapperEnvWalker(env)

    # get offline data and set up environment dataset
    datasets = [d4rl.qlearning_dataset(gym.make(name_dataset)) for name_dataset in args.names_dataset]
    state = np.concatenate(tuple(dataset["observations"] for dataset in datasets))
    action = np.concatenate(tuple(dataset["actions"] for dataset in datasets))
    reward = np.concatenate(tuple(dataset["rewards"] for dataset in datasets))
    state_next = np.concatenate(tuple(dataset["next_observations"] for dataset in datasets))
    terminal = np.concatenate(tuple(dataset["terminals"] for dataset in datasets))
    state = torch.tensor(state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.float32)
    reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(dim=1)
    state_next = torch.tensor(state_next, dtype=torch.float32)
    terminal = torch.tensor(terminal, dtype=torch.float32).unsqueeze(dim=1)
    if args.hallucinate:
        action_hallucinated = torch.zeros(state.shape, dtype=torch.float32)
        action = torch.cat((action, action_hallucinated), dim=-1)
    dataset_env = torch.utils.data.TensorDataset(state, action, reward, state_next, terminal)

    # set up initial state dataset
    dataset_states_initial = data.DatasetNumpy()
    for idx_state_initial in range(100):
        state_initial = env.reset()
        dataset_states_initial.append(state_initial)

    # set up model dataset
    dataset_model = data.DatasetSARS(capacity=args.replay_size)

    # get model checkpoint
    checkpoint_model = torch.load(args.path_checkpoint_model, map_location=args.device)
    model = checkpoint_model["model"]

    # set up model environment
    model.eval()
    if args.hallucinate:
        EnvModel = envs.EnvModelHallucinated
    else:
        EnvModel = envs.EnvModel
    env_model = EnvModel(env.observation_space, env.action_space, dataset_states_initial, model, env.done, args)
    env_model = gym.wrappers.TimeLimit(env_model, 100)
    env_model = envs.WrapperEnv(env_model)
    env.env._max_episode_steps = 100

    # set up agents
    with open("policies_metadata.json", "r") as f:
        policies_metadata = json.load(f)
    idx_policy = 55
    policy_metadata = policies_metadata[idx_policy]
    print(policy_metadata)
    path_policy = policy_metadata["policy_path"]
    with tf.io.gfile.GFile(os.path.join("gs://gresearch/deep-ope/d4rl/", path_policy), "rb") as f:
        weights = pickle.load(f)
    agent = agents.AgentDOPE(weights)
    if args.hallucinate:
        agent_antagonist = agents.AgentSACAntagonist(env.observation_space, env_model.space_action_hallucinated, args)
        agent = agents.AgentConcat([agent, agent_antagonist])

    # train antagonist
    if args.hallucinate:
        for idx_epoch in range(args.num_epochs):
            agent.train()
            if (idx_epoch + 1) % args.interval_rollout_model == 0:
                env_model.rollout(agent, dataset_model, args.num_steps_rollout_model, args.max_length_rollout_model)
            dataloader = data.get_dataloader(dataset_env, dataset_model, args.num_steps_train_agent, args.size_batch, args.ratio_env_model)
            for batch in dataloader:
                loss_actor, loss_critic, loss_alpha = agent.step(batch)
            print(f"idx_epoch: {idx_epoch}, loss_actor: {loss_actor}, loss_critic: {loss_critic}, loss_alpha: {loss_alpha}")
            if (idx_epoch + 1) % args.interval_eval_agent == 0:
                return_eval = evaluation.evaluate_agent(agent, env_model, args.num_episodes_eval)
                print(f"idx_epoch: {idx_epoch}, return_eval: {return_eval}")

    # save antagonist checkpoint
    torch.save(agent_antagonist, "checkpoint_antagonist_tmp")

    # evaluate
    return_eval_env = evaluation.evaluate_agent(agent, env, args.num_episodes_eval)    
    print(f"true: {return_eval_env}")
    betas = [0.0, 0.2533, 0.5244, 0.8416, 1.2816, 2.0, 4.0]
    for idx_beta, beta in enumerate(betas):
        env_model.unwrapped.beta = beta
        return_eval_model = evaluation.evaluate_agent(agent, env_model, args.num_episodes_eval)
        print(f"beta = {beta}: {return_eval_model}")

