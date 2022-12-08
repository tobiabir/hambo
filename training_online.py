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
import models
import rollout
import training
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HUCRL")
    parser.add_argument("--name_env", type=str, choices=["Point", "MountainCar", "Pendulum", "InvertedPendulum", "Swimmer", "Hopper", "HalfCheetah", "Reacher"], required=True,
                        help="name of the environment")
    parser.add_argument("--id_experiment", type=str,
                        help="id of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for reward (default: 0.99)")
    parser.add_argument("--model", type=str, choices=["GP", "EnsembleDeterministic", "EnsembleProbabilisticHomoscedastic", "EnsembleProbabilisticHeteroscedastic"], default="EnsembleProbabilisticHeteroscedastic",
                        help="what type of model to use to learn dyamics of the environment (default: EnsembleProbabilisticHeteroscedastic)")
    parser.add_argument("--num_h_model", type=int, default=2,
                        help="number of hidden layers in model (only for ensembles) (default: 2)")
    parser.add_argument("--dim_h_model", type=int, default=256,
                        help="dimension of hidden layers in model (only for ensembles) (default: 256)")
    parser.add_argument("--size_ensemble_model", type=int, default=7,
                        help="number of networks in model (default: 7)")
    parser.add_argument("--num_elites_model", type=int, default=5,
                        help="number of elite networks in model (default: 5)")
    parser.add_argument("--use_scalers", default=False, action="store_true",
                        help="set to use scalers for transition model (default: False)")
    parser.add_argument("--activation_model", default="ReLU", choices=["ReLU", "GELU", "SiLU"],
                        help="activation function to use for the hidden layers of the model (default: ReLU)")
    parser.add_argument("--weight_prior_model", type=float, default=0.0,
                        help="weight on the prior in the map estimate for model training (default: 0.0)")
    parser.add_argument("--method_sampling", default="DS", choices=["DS", "TS1"],
                        help="sampling method to use in model environment (see [Chua et al.](https://arxiv.org/abs/1805.12114) for explanation) (default: DS)")
    parser.add_argument("--use_aleatoric", default=False, action="store_true",
                        help="set to use aleatoric noise from transition model (default: False)")
    parser.add_argument("--hallucinate", default=False, action="store_true",
                        help="set to add hallucinated control (default: False)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="parameter for the amount of hallucinated control (only has effect if hallucinate is set) (default: 1.0)")
    parser.add_argument("--ratio_env_model", type=float, default=0.05,
                        help="ratio of env data to model data in agent batches")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="regularizer weight alpha of the entropy regularization term for sac training (default: 0.05)")
    parser.add_argument("--learn_alpha", default=False, action="store_true",
                        help="set to learn alpha (default: False)")
    parser.add_argument("--lr_model", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_agent", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--num_steps", type=int, default=4096,
                        help="number of steps (default: 4096)")
    parser.add_argument("--num_steps_startup", type=int, default=0,
                        help="number of steps of rollout to do during startup (default: 0)")
    parser.add_argument("--interval_train_model", type=int, default=np.e,
                        help="interval of steps after which a round of training is done (default: never)")
    parser.add_argument("--interval_rollout_model", type=int, default=128,
                        help="interval of steps after which a round of training is done (default: 128)")
    parser.add_argument("--num_steps_rollout_model", type=int, default=128,
                        help="number of steps to rollout model in a round of rollout (default: 128)")
    parser.add_argument("--max_length_rollout_model", type=int, default=1,
                        help="number of steps to rollout model from initial state (a.k.a. episode length) (default: 1)")
    parser.add_argument("--interval_train_agent", type=int, default=128,
                        help="interval of steps after which a round of training is done for agent (default: 128)")
    parser.add_argument("--num_steps_train_agent", type=int, default=128,
                        help="number of steps to train agent per iteration (default: 128)")
    parser.add_argument("--interval_eval_agent", type=int, default=128,
                        help="interval of steps after which a round of evaluation is done (default: 128)")
    parser.add_argument("--num_episodes_eval_agent", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--device", default=None,
                        help="device (default: gpu if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=1000000,
                        help="capacity of replay buffer (default: 1000000)")
    args = parser.parse_args()
    if type(args.interval_train_model) == float:
        args.algorithm = "SAC"
    else:
        if args.hallucinate:
            args.algorithm = "HMBSAC"
        else:
            args.algorithm = "MBSAC"
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"algorithm: {args.algorithm}")
    print(f"device: {args.device}")
    args.weight_penalty_reward = 0.0
    args.conservative = False

    wandb.init(mode="disabled", project="HMBSAC", entity="tobiabir", config=args)

    # setting rng seeds
    utils.set_seeds(args.seed)

    if args.name_env == "Point":
        env = envs.EnvPoint(dim_state=2)
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
    elif args.name_env == "Reacher":
        env = gym.make("Reacher-v2")
        env = envs.WrapperEnvHalfCheetah(env)

    if args.model == "GP":
        Model = None  # TODO
    elif args.model == "EnsembleDeterministic":
        Model = models.NetDense
    elif args.model == "EnsembleProbabilisticHomoscedastic":
        Model = models.NetGaussHomo
    elif args.model == "EnsembleProbabilisticHeteroscedastic":
        Model = models.NetGaussHetero
    model = Model(
        dim_x=env.observation_space.shape[0] + env.action_space.shape[0],
        dim_y=1 + env.observation_space.shape[0],
        num_h=args.num_h_model,
        dim_h=args.dim_h_model,
        size_ensemble=args.size_ensemble_model,
        num_elites=args.num_elites_model,
        use_scalers=args.use_scalers,
        activation=eval("torch.nn." + args.activation_model),
    ).to(args.device)
    if args.hallucinate:
        EnvModel = envs.EnvModelHallucinated
    else:
        EnvModel = envs.EnvModel
    env_model = EnvModel(env.observation_space, env.action_space, None, model, env.done, args)
    env_model = gym.wrappers.TimeLimit(
        env_model, args.max_length_rollout_model)
    env_model = envs.WrapperEnv(env_model)
    env_eval = copy.deepcopy(env)
    has_trained_model = False

    agent_random = agents.AgentRandom(env.action_space)
    agent_protagonist = agents.AgentSAC(env.observation_space, env.action_space, args)
    if args.hallucinate:
        agent_random_hallucinate = agents.AgentRandom(env_model.space_action_hallucinated)
        agent_random = agents.AgentTuple([agent_random, agent_random_hallucinate])
        agent_hallucinate = agents.AgentSAC(env_model.observation_space, env_model.space_action_hallucinated, args)
        agent = agents.AgentTuple([agent_protagonist, agent_hallucinate])
    else:
        agent_random = agents.AgentTuple([agent_random])
        agent = agents.AgentTuple([agent_protagonist])

    dataset_env = data.DatasetSARS(capacity=args.replay_size)
    dataset_model = data.DatasetSARS(capacity=args.replay_size)
    dataset_states_initial = data.DatasetNumpy()

    state = env.reset()
    dataset_states_initial.append(state)

    print("startup...")
    rollout.rollout_steps(env, agent_random, dataset_env, dataset_states_initial, args.num_steps_startup)

    state = env.state

    for idx_step in range(args.num_steps_startup, args.num_steps):
        agent.train()
        action = agent.get_action(state)
        state_next, reward, done, info = env.step(action[0])
        mask = np.float32(done and not info["TimeLimit.truncated"])
        dataset_env.push(state, action, reward, state_next, mask)
        state = state_next
        if done:
            state = env.reset()
        dataset_states_initial.append(state)
        if (idx_step + 1) % args.interval_train_model == 0:
            losses_model, scores_calibration = training.train_ensemble_map(model, dataset_env, args.weight_prior_model, args.lr_model, args.size_batch, args.device)
            loss_model = losses_model.mean()
            score_calibration = scores_calibration.mean()
            wandb.log({"loss_model": loss_model, "score_calibration": score_calibration, "idx_step": idx_step})
            print(f"idx_step: {idx_step}, loss_model: {loss_model}, score_calibration: {score_calibration}")
            has_trained_model = True
        if has_trained_model and (idx_step + 1) % args.interval_rollout_model == 0:
            model.eval()
            env_model = EnvModel(env.observation_space, env.action_space,
                                 dataset_states_initial, model, env.done, args)
            env_model = gym.wrappers.TimeLimit(env_model, args.max_length_rollout_model)
            env_model = envs.WrapperEnv(env_model)
            env_model.rollout(agent, dataset_model, args.num_steps_rollout_model, args.max_length_rollout_model)
        if (idx_step + 1) % args.interval_train_agent == 0:
            dataloader = data.get_dataloader(dataset_env, dataset_model, args.num_steps_train_agent, args.size_batch, args.ratio_env_model)
            for batch in dataloader:
                loss_agent = agent.step(batch)
            #wandb.log({"loss_actor": loss_actor, "loss_critic": loss_critic, "loss_alpha": loss_alpha, "alpha": agent.alpha, "idx_step": idx_step})
            print(f"idx_step: {idx_step}, loss_agent: {loss_agent}")
        if (idx_step + 1) % args.interval_eval_agent == 0:
            return_eval = evaluation.evaluate_agent(agent, env_eval, args.num_episodes_eval_agent)
            wandb.log({"return_eval": return_eval, "idx_step": idx_step})
            print(f"idx_step: {idx_step}, return_eval: {return_eval}")
            torch.save({"agent": agent, "return_eval": return_eval}, f"Checkpoints/Agents/checkpoint_reacher_{idx_step + 1}")
