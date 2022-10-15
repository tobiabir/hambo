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
import utils

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Offline Policy Evaluation")

    # model arguments
    parser.add_argument("--path_checkpoint_model", type=str, default=None,
                        help="path to model checkpoint (default: None (no checkpointing))")

    # model environment arguments
    parser.add_argument("--max_length_rollout_model", type=int, default=1000,
                        help="number of steps to rollout model from initial state (a.k.a. episode length) (default: 1000)")
    parser.add_argument("--method_sampling", default="DS", choices=["DS", "TS1", "TSInf"],
                        help="sampling method to use in model environment (see [Chua et al.](https://arxiv.org/abs/1805.12114) for explanation) (default: DS)")
    parser.add_argument("--use_aleatoric", default=False, action="store_true",
                        help="set to use aleatoric noise from transition model (default: False)")
    parser.add_argument("--weight_penalty_reward", type=float, default=0.0,
                        help="weight on the reward penalty (see MOPO) (default: 0.0)")
    parser.add_argument("--hallucinate", default=False, action="store_true",
                        help="set to add hallucinated control (default: False)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="parameter for the amount of hallucinated control (only has effect if hallucinate is set) (default: 1.0)")

    # policy arguments
    parser.add_argument("--path_agent", type=str,
                        help="path/index of the agent/policy to evaluate (default: None)")

    # agent learning arguments
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")

    # SAC arguments
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="regularizer weight alpha of the entropy regularization term for sac training (default: 0.1)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr_agent", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--conservative", default=False, action="store_true",
                        help="set to add conservative Q learning (default: False)")

    # procedure arguments
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--num_rounds", type=int, default=0,
                        help="number of episodes to train (default: 0)")
    parser.add_argument("--interval_rollout_model", type=int, default=1,
                        help="interval of steps after which a round of training is done (default: 1)")
    parser.add_argument("--num_episodes_rollout_model", type=int, default=100,
                        help="number of episodes to rollout model in a round of rollout (default: 100)")
    parser.add_argument("--num_steps_train_agent", type=int, default=1000,
                        help="number of steps to train agent per iteration (default: 1000)")
    parser.add_argument("--interval_eval_agent", type=int, default=128,
                        help="interval of steps after which a round of evaluation is done (default: 1)")
    parser.add_argument("--num_episodes_eval", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--device", default=None,
                        help="device (default: cuda if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=1000000,
                        help="capacity of replay buffer (default: 1000000)")

    # output arguments
    parser.add_argument("--path_results", default=os.path.join("Results", "results_tmp"),
                        help="path to write the results to (default: Results/results_tmp")

    args = parser.parse_args()
    args.ratio_env_model = 0.0 # we train the adversary only on model data
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

    # assert correct usage
    if args.hallucinate:
        assert args.method_sampling == "TSInf" or 0 < args.num_rounds, "An adversary has to be trained for DS and TS1 -> num_rounds should be greater than 0"
        assert args.method_sampling != "TSInf" or args.num_rounds == 0, "No training required for TSInf -> num_rounds should be 0!"
    else:
        assert args.num_rounds == 0, "No training required without hallucination -> hallucinate should not be set!"

    # setting rng seeds
    utils.set_seeds(args.seed)

    # get model checkpoint
    checkpoint_model = torch.load(args.path_checkpoint_model, map_location=args.device)
    model = checkpoint_model["model"]
    id_env = checkpoint_model["id_env"]

    # set up environment
    env = gym.make(id_env)
    if "HalfCheetah" in id_env:
        env = envs.WrapperEnvHalfCheetah(env)
    elif "Hopper" in id_env:
        env = envs.WrapperEnvHopper(env)
    elif "Pendulum" in id_env:
        env = envs.WrapperEnvPendulum(env)
    elif "Walker" in id_env:
        env = envs.WrapperEnvWalker(env)
    env.reset(args.seed)

    # set up initial state dataset
    dataset_states_initial = data.DatasetNumpy()
    num_states_initial = 100000
    for idx_state_initial in range(num_states_initial):
        state_initial = env.reset().astype(np.float32)
        dataset_states_initial.append(state_initial)

    # set up model dataset
    dataset_model = data.DatasetSARS(capacity=args.replay_size)

    # set up model environment
    model.eval()
    if args.hallucinate or args.method_sampling == "TSInf":
        EnvModel = envs.EnvModelHallucinated
    else:
        EnvModel = envs.EnvModel
    env_model = EnvModel(env.observation_space, env.action_space, dataset_states_initial, model, env.done, args)
    env_model = gym.wrappers.TimeLimit(env_model, args.max_length_rollout_model)
    env_model = envs.WrapperEnv(env_model)
    env.env._max_episode_steps = args.max_length_rollout_model

    # set up protagonist
    if os.path.isfile(args.path_agent):
        checkpoint_agent = torch.load(args.path_agent)
        agent_protagonist = checkpoint_agent["agent"].agents[0]
        agent_protagonist = agents.AgentFixed(agent_protagonist)
    else:
        with open("policies_metadata.json", "r") as f:
            policies_metadata = json.load(f)
        idx_policy = int(args.path_agent)
        policy_metadata = policies_metadata[idx_policy]
        print(policy_metadata)
        path_policy = policy_metadata["policy_path"]
        with tf.io.gfile.GFile(os.path.join("gs://gresearch/deep-ope/d4rl/", path_policy), "rb") as f:
            weights = pickle.load(f)
        agent_protagonist = agents.AgentDOPE(weights)

    # train antagonist
    if 0 < args.num_rounds:
        # set up antagonist
        agent_antagonist_random = agents.AgentRandom(env_model.space_action_hallucinated)
        agent_random = agents.AgentTuple([agent_protagonist, agent_antagonist_random])
        if args.method_sampling == "DS":
            agent_antagonist = agents.AgentSACAntagonist(env.observation_space, env_model.space_action_hallucinated, args)
        else:
            agent_antagonist = agents.AgentDQNAntagonist(env.observation_space, env_model.space_action_hallucinated, args)
        agent = agents.AgentTuple([agent_protagonist, agent_antagonist])

        # rollout using the random antagonist to ensure good initial exploration
        env_model.rollout(agent_random, dataset_model, args.num_episodes_rollout_model, args.max_length_rollout_model)

        # alternating between antagonist training and rollout
        for idx_round in range(args.num_rounds):
            agent.train()
            if (idx_round + 1) % args.interval_rollout_model == 0 and idx_round > 0:
                if args.method_sampling == "TS1":
                    agent_antagonist.epsilon -= args.interval_rollout_model / args.num_rounds
                env_model.rollout(agent, dataset_model, args.num_episodes_rollout_model, args.max_length_rollout_model)
            dataloader = data.get_dataloader(dataset_model, None, args.num_steps_train_agent, args.size_batch)
            for batch in dataloader:
                losses = agent.step(batch)
            print(f"idx_round: {idx_round}, losses: {losses}")
            if (idx_round + 1) % args.interval_eval_agent == 0:
                return_eval = evaluation.evaluate_agent(agent, env_model, args.num_episodes_eval)
                print(f"idx_round: {idx_round}, return_eval: {return_eval}")

        # save antagonist checkpoint
        torch.save(agent_antagonist, "checkpoint_antagonist_tmp")
    
    else:
        agent = agent_protagonist

    # evaluate
    results = {}
    return_eval_env = evaluation.evaluate_agent(agents.AgentTuple([agent_protagonist]), env, args.num_episodes_eval)    
    results["return_eval_env"] = return_eval_env
    print(f"true: {return_eval_env}")
    
    if args.method_sampling == "TSInf":
        idxs_elites = model.idxs_elites
        returns_eval_model = np.zeros(len(idxs_elites))
        for idx_idx_elite in range(len(idxs_elites)):
            agent_antagonist = agents.AgentModelSelectFixed(idx_idx_elite)
            agent = agents.AgentTuple([agent_protagonist, agent_antagonist])
            return_eval_model = evaluation.evaluate_agent(agent, env_model, args.num_episodes_eval) 
            returns_eval_model[idx_idx_elite] = return_eval_model
        return_eval_model_mean = returns_eval_model.mean()
        return_eval_model_std = returns_eval_model.std()
        if args.hallucinate:
            return_eval_model = returns_eval_model.min()
        else:
            return_eval_model = return_eval_model_mean
        results["return_eval_model"] = return_eval_model
        results["return_eval_model_mean"] = return_eval_model_mean
        results["return_eval_model_std"] = return_eval_model_std
        print(f"estimated: mean: {return_eval_model_mean}, std: {return_eval_model_std}")
        print(f"estimated: {return_eval_model}")
    else:
        return_eval_model = evaluation.evaluate_agent(agent, env_model, args.num_episodes_eval)
        results["return_eval_model"] = return_eval_model
        print(f"estimated: {return_eval_model}")

    # save evaluation results
    torch.save(results, args.path_results)

