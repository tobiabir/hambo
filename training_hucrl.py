import argparse
import gym
import numpy as np
import random
import torch

import agents
import data
import envs
import nets
import rollout
import training

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HUCRL")
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
    parser.add_argument("--num_steps_model", type=int, default=256, metavar="N",
                        help="number of steps to train model per iteration (default: 256)")
    parser.add_argument("--num_episodes_agent", type=int, default=16, metavar="N",
                        help="number of episodes to train agent per iteration (default: 16)")
    parser.add_argument("--device", default="cpu",
                        help="device (default: cpu)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="capacity of replay buffer (default: 100000)")
    args = parser.parse_args()

    #env = gym.make("MountainCarContinuous-v0")
    #env = gym.make("Pendulum-v1", g=9.81)
    env = envs.EnvPoint()

    # setting rng seeds
    random.seed(args.seed)    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    
    model = nets.NetDense(
        dim_x=env.observation_space.shape[0] + env.action_space.shape[0],
        dim_y=env.observation_space.shape[0],
        num_h=1,
        dim_h=64,
        size_ensemble=5
    )
    #EnvModel = envs.EnvModel
    EnvModel = envs.EnvModelHallucinated
    env_model = EnvModel(env.observation_space, env.action_space, None, env.reward, model)

    agent = agents.AgentSAC(env_model.observation_space, env_model.action_space, args)

    dataset = data.DatasetSARS(capacity=args.replay_size)
    dataset_states_initial = data.DatasetNumpy()

    for idx_episode in range(args.num_episodes):
        dataset_episode, state_initial, reward_episode = rollout.rollout_episode(env, agent)
        print("idx_episode: %i, reward_episode: %f" % (idx_episode, reward_episode))
        dataset.concat(dataset_episode)
        dataset_states_initial.append(state_initial)
        if len(dataset) > args.batch_size:
            model = training.train_ensemble(model, dataset, args)
        model.eval()
        env_model = EnvModel(env.observation_space, env.action_space, dataset_states_initial, env.reward, model)
        agent = training.train_sac(agent, env_model, args)
