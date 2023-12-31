import argparse
import copy
import d4rl
import gym
import numpy as np
import torch
import wandb

import agents
import data
import envs
import evaluation
import training
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Offline RL")

    # dataset arguments
    parser.add_argument("--name_env", type=str, choices=["HalfCheetah", "Hopper", "Walker"], required=True,
                        help="name of the environment")
    parser.add_argument("--name_task", type=str, choices=["random", "medium", "expert", "medium-expert"],  required=True,
                        help="name of the task")

    # model arguments
    parser.add_argument("--path_checkpoint_model", type=str, default=None,
                        help="path to model checkpoint (default: None (no checkpointing))")
    parser.add_argument("--weight_prior_model", type=float, default=0.0,
                        help="weight on the prior in the map estimate for model training (default: 0.0)")
    parser.add_argument("--weight_loss_adversarial_model", type=float, default=0.0,
                        help="regularizer weight lambda of the prior in the map estimate for model training (default: 0.0)")
    parser.add_argument("--lr_model", type=float, default=0.001,
                        help="learning rate (default: 0.001)")

    # model environment arguments
    parser.add_argument("--method_sampling", default="DS", choices=["DS", "TS1"],
                        help="sampling method to use in model environment (see [Chua et al.](https://arxiv.org/abs/1805.12114) for explanation)) (default: DS)")
    parser.add_argument("--calibrate", default=False, action="store_true",
                        help="set to use calibration for transition model (default: False)")
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
    parser.add_argument("--alpha", type=float, default=0.1,
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
    parser.add_argument("--ratio_env_model", type=float, default=0.05,
                        help="ratio of env data to model data in agent batches (default: 0.05)")
    parser.add_argument("--num_rounds", type=int, default=128,
                        help="number of rounds to train (default: 128)")
    parser.add_argument("--interval_rollout_model", type=int, default=1,
                        help="interval of rounds after which a round of model rollout is done (default: 1)")
    parser.add_argument("--num_steps_rollout_model", type=int, default=100000,
                        help="number of steps to rollout model in a round of rollout (default: 100000)")
    parser.add_argument("--max_length_rollout_model", type=int, default=1,
                        help="number of steps to rollout model from initial state (a.k.a. episode length) (default: 1)")
    parser.add_argument("--interval_train_model_adversarial", type=int, default=1,
                        help="interval of rounds after which a round of adversarial training is done (default: 1)")
    parser.add_argument("--num_steps_train_agent", type=int, default=250,
                        help="number of steps to train agent per round (default: 250)")
    parser.add_argument("--interval_eval_agent", type=int, default=1,
                        help="interval of rounds after which a round of evaluation is done (default: 1)")
    parser.add_argument("--num_episodes_eval_agent", type=int, default=1,
                        help="number of episodes to evaluate (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--device", default=None,
                        help="device (default: cuda if available else cpu)")
    parser.add_argument("--replay_size", type=int, default=1000000,
                        help="capacity of replay buffer (default: 1000000)")

    args = parser.parse_args()
    use_model = args.ratio_env_model < 1.0
    train_adversarial = args.weight_loss_adversarial_model > 0.0
    if use_model:
        if args.weight_loss_adversarial_model == 0.0:
            if args.hallucinate:
                if args.weight_penalty_reward == 0.0:
                    if args.conservative:
                        args.algorithm = "COMBO + HAMBO"
                    else:
                        args.algorithm = "HAMBO"
                else:
                    if args.conservative:
                        args.algorithm = "MOPO + COMBO + HAMBO"
                    else:
                        args.algorithm = "MOPO + HAMBO"
            else:
                if args.weight_penalty_reward == 0.0:
                    if args.conservative:
                        args.algorithm = "COMBO"
                    else:
                        args.algorithm = "MBPO"
                else:
                    if args.conservative:
                        args.algorithm = "MOPO + COMBO"
                    else:
                        args.algorithm = "MOPO"
        else:
            if args.hallucinate:
                if args.weight_penalty_reward == 0.0:
                    if args.conservative:
                        args.algorithm = "COMBO + RAMBO + HAMBO"
                    else:
                        args.algorithm = "RAMBO + HAMBO"
                else:
                    if args.conservative:
                        args.algorithm = "MOPO + COMBO + RAMBO + HAMBO"
                    else:
                        args.algorithm = "MOPO + RAMBO + HAMBO"
            else:
                if args.weight_penalty_reward == 0.0:
                    if args.conservative:
                        args.algorithm = "COMBO + RAMBO"
                    else:
                        args.algorithm = "RAMBO"
                else:
                    if args.conservative:
                        args.algorithm = "MOPO + COMBO + RAMBO"
                    else:
                        args.algorithm = "MOPO + RAMBO"
    else:
        if args.conservative:
            args.algorithm = "CQL"
        else:
            args.algorithm = "SAC"
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"algorithm: {args.algorithm}")
    print(f"device: {args.device}")

    wandb.init(mode="disabled", project="Offline RL", entity="tobiabir", config=args)

    # setting rng seeds
    utils.set_seeds(args.seed)

    if args.name_env == "HalfCheetah":
        env = gym.make("halfcheetah-" + args.name_task + "-v2")
        env = envs.WrapperEnvHalfCheetah(env)
    elif args.name_env == "Hopper":
        env = gym.make("hopper-" + args.name_task + "-v2")
        env = envs.WrapperEnvHopper(env)
    elif args.name_env == "Walker":
        env = gym.make("walker2d-" + args.name_task + "-v2")
        env = envs.WrapperEnvWalker(env)

    dataset_env = env.get_dataset()
    dataset_states_initial = data.DatasetNumpy()
    dataset_states_initial.append_batch(list(dataset_env["observations"]))
    state = dataset_env["observations"]
    action = dataset_env["actions"]
    if use_model and args.hallucinate:
        if args.method_sampling == "DS":
            action_hallucinated = np.zeros(state.shape, dtype=np.float32)
        else:
            action_hallucinated = np.zeros(state.shape[0], dtype=np.int64)
        action = action, action_hallucinated
    reward = dataset_env["rewards"]
    state_next = dataset_env["next_observations"]
    terminal = dataset_env["terminals"]
    dataset_env = data.DatasetSARS()
    dataset_env.push_batch(list(zip(state, zip(*action), reward, state_next, terminal)))
    dataset_model = data.DatasetSARS(capacity=args.replay_size)

    if use_model:
        checkpoint_model = torch.load(args.path_checkpoint_model, map_location=args.device)
        model = checkpoint_model["model"]
        if not args.calibrate:
            model.temperature = 1.0

        model.eval()
        if args.hallucinate:
            EnvModel = envs.EnvModelHallucinated
        else:
            EnvModel = envs.EnvModel
        env_model = EnvModel(env.observation_space, env.action_space, dataset_states_initial, model, env.done, args)
        env_model = gym.wrappers.TimeLimit(env_model, args.max_length_rollout_model)
        env_model = envs.WrapperEnv(env_model)

    agent = agents.AgentSAC(env.observation_space, env.action_space, args)
    if use_model and args.hallucinate:
        if args.method_sampling == "DS":
            agent_antagonist = agents.AgentSACAntagonist(env.observation_space, env_model.space_action_hallucinated, args)
        else:
            agent_antagonist = agents.AgentDQNAntagonist(env.observation_space, env_model.space_action_hallucinated, args)
        agent = agents.AgentTuple([agent, agent_antagonist])

    for idx_round in range(args.num_rounds):
        agent.train()
        if use_model and (idx_round + 1) % args.interval_rollout_model == 0:
            model.eval()
            env_model.rollout(agent, dataset_model, args.num_steps_rollout_model, args.max_length_rollout_model)
        dataloader = data.get_dataloader(dataset_env, dataset_model, args.num_steps_train_agent, args.size_batch, args.ratio_env_model)
        for batch in dataloader:
            losses = agent.step(batch)
        wandb.log({"agent_protagonist": losses[0], "agent_antagonist": losses[1], "idx_round": idx_round})
        print(f"idx_round: {idx_round}, agent_protagonist: {losses[0]}, agent_antagonist: {losses[1]}")
        if train_adversarial and (idx_round + 1) % args.interval_train_model_adversarial == 0:
            model.train()
            losses_model, scores_calibration = training.train_ensemble_adversarial(model, dataset_env, agent, env_model.model_termination, args.weight_prior_model, args.gamma, args.weight_loss_adversarial_model, args.lr_model, args.size_batch, args.device)
            wandb.log({"loss_model": loss_model, "score_calibration": score_calibration, "idx_round": idx_round})
            print(f"idx_round: {idx_round}, loss_model: {loss_model}, score_calibration: {score_calibration}")
        if (idx_round + 1) % args.interval_eval_agent == 0:
            return_eval = evaluation.evaluate_agent(agent, env, args.num_episodes_eval_agent)
            return_eval_normalized = env.get_normalized_score(return_eval)
            wandb.log({"return_eval": return_eval, "return_eval_normalized": return_eval_normalized, "idx_round": idx_round}) 
            print(f"idx_round: {idx_round}, return_eval: {return_eval}, return_eval_normalized: {return_eval_normalized}")

