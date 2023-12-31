import argparse
import d4rl
import gym
import numpy as np
import os
import pickle
import torch
import wandb

import data
import envs
import evaluation
import models
import training
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transition Model training for Offline RL")

    # dataset arguments
    parser.add_argument("--paths_dataset", type=str, nargs="+", required=True,
                        help="paths or names of the datasets to use")

    # model arguments
    parser.add_argument("--path_checkpoint_model", type=str, default=None,
                        help="path to model checkpoint (default: None (no checkpointing))")
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
    parser.add_argument("--lr_model", type=float, default=0.001,
                        help="learning rate (default: 0.001)")

    # procedure arguments
    parser.add_argument("--size_batch", type=int, default=256,
                        help="batch size (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--device", default=None,
                        help="device (default: cuda if available else cpu)")

    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

    # setting rng seeds
    utils.set_seeds(args.seed)

    # get offline data and set up dataset and env (to get observation- and action space)
    if os.path.isfile(args.paths_dataset[0]):
        datasets = []
        for path_dataset in args.paths_dataset:
            checkpoint = torch.load(path_dataset)
            dataset = checkpoint["dataset"]
            datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(datasets)
        id_env = checkpoint["id_env"]
    else:
        datasets = [d4rl.qlearning_dataset(gym.make(name_dataset)) for name_dataset in args.paths_dataset]
        state = np.concatenate(tuple(dataset["observations"] for dataset in datasets))
        action = np.concatenate(tuple(dataset["actions"] for dataset in datasets)),
        reward = np.concatenate(tuple(dataset["rewards"] for dataset in datasets))
        state_next = np.concatenate(tuple(dataset["next_observations"] for dataset in datasets))
        terminal = np.concatenate(tuple(dataset["terminals"] for dataset in datasets))
        dataset = data.DatasetSARS()
        dataset.push_batch(list(zip(state, zip(*action), reward, state_next, terminal)))
        id_env = args.paths_dataset[0]
    env = gym.make(id_env)

    # train model
    if args.model == "GP":
        # train GP
        model, loss_model, score_calibration = training.train_gp(dataset)

        # create evaluation scores
        losses_model = [loss_model]
        scores_calibration = [score_calibration]

    else:
        # choose model class
        if args.model == "EnsembleDeterministic":
            Model = models.NetDense
        elif args.model == "EnsembleProbabilisticHomoscedastic":
            Model = models.NetGaussHomo
        elif args.model == "EnsembleProbabilisticHeteroscedastic":
            Model = models.NetGaussHetero

        # initialize model
        model = Model(
            dim_x=env.observation_space.shape[0] +
            env.action_space.shape[0],
            dim_y=1 + env.observation_space.shape[0],
            num_h=args.num_h_model,
            dim_h=args.dim_h_model,
            size_ensemble=args.size_ensemble_model,
            num_elites=args.num_elites_model,
            use_scalers=args.use_scalers,
            activation=eval("torch.nn." + args.activation_model),
        ).to(args.device)

        # train ensemble
        losses_model, scores_calibration = training.train_ensemble_map(model, dataset, args.weight_prior_model, args.lr_model, args.size_batch, args.device)

    # print evaluation scores
    print(f"losses_model: {losses_model}, scores_calibration: {scores_calibration}")

    # create checkpoint
    checkpoint = {
        "id_env": id_env,
        "paths_dataset": args.paths_dataset,
        "losses_model": losses_model,
        "scores_calibration": scores_calibration,
        "model": model,
    }

    # save checkpoint
    torch.save(checkpoint, args.path_checkpoint_model)
