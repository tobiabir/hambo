import argparse
import gym
import re
import torch

import agents
import data
import envs
import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset Generation")
    parser.add_argument("--id_env", type=str, required=True,
                        help="identifier of the environment")
    parser.add_argument("--path_agent", type=str,
                        help="path to the agent checkpoint (if None the random agent is used) (default: None)")
    parser.add_argument("--num_steps", type=int, default=1000000,
                        help="number of steps to rollout for (default: 1000000)")
    parser.add_argument("--path_dataset", type=str,
                        help="path to store the dataset to (default: None)")
    args = parser.parse_args()

    if "Point" in args.id_env:
        assert re.fullmatch("Point[0-9]+d") is not None
        dim_state = re.search("[0-9]+").group(0)
        env = envs.EnvPointEscape(dim_state)
        env = gym.wrappers.TimeLimit(env, 100)
        env = envs.WrapperEnv(env)
    else:
        env = gym.make(args.id_env)
        if "Pendulum" in args.id_env:
            env = envs.WrapperEnvPendulum(env)
        elif "Hopper" in args.id_env:
            env = envs.WrapperEnvHopper(env)
        elif "HalfCheetah" in args.id_env:
            env = envs.WrapperEnvHalfCheetah(env)
        elif "Reacher" in args.id_env:
            env = envs.WrapperEnvReacher(env)

    if args.path_agent is None:
        agent = agents.AgentRandom(env.action_space)
    else:
        checkpoint = torch.load(args.path_agent)
        agent = checkpoint#["agent"]

    dataset = data.DatasetSARS()

    env.reset()
    rollout.rollout_steps(env, agent, dataset, None, args.num_steps)

    checkpoint = {"dataset": dataset, "id_env": args.id_env}
    torch.save(checkpoint, args.path_dataset)
