import argparse
import d4rl
import gym
import os
import torch
import wandb

import data
import evaluation
import models
import training
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transition Model training for Offline RL")

    # dataset arguments
    parser.add_argument("--name_dataset", type=str, required=True,
                        help="name of the dataset to use")

    # model arguments
    parser.add_argument("--path_checkpoint_model", type=str, default=None,
                        help="path to model checkpoint (default: None (no checkpointing))")
    parser.add_argument("--model", type=str, choices=["GP", "EnsembleDeterministic", "EnsembleProbabilisticHomoscedastic", "EnsembleProbabilisticHeteroscedastic"], default="EnsembleProbabilisticHeteroscedastic",
                        help="what type of model to use to learn dyamics of the environment (default: EnsembleProbabilisticHeteroscedastic)")
    parser.add_argument("--use_scalers", default=False, action="store_true",
                        help="set to use scalers for transition model (default: False)")
    parser.add_argument("--num_h_model", type=int, default=2,
                        help="number of hidden layers in model (only for ensembles) (default: 2)")
    parser.add_argument("--dim_h_model", type=int, default=256,
                        help="dimension of hidden layers in model (only for ensembles) (default: 256)")
    parser.add_argument("--size_ensemble_model", type=int, default=7,
                        help="number of networks in model (default: 7)")
    parser.add_argument("--num_elites_model", type=int, default=5,
                        help="number of elite networks in model (default: 5)")
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

    env = gym.make(args.name_dataset)
    dataset_env = env.get_dataset()

    state = torch.tensor(dataset_env["observations"], dtype=torch.float32)
    action = torch.tensor(dataset_env["actions"], dtype=torch.float32)
    reward = torch.tensor(dataset_env["rewards"], dtype=torch.float32).unsqueeze(dim=1)
    state_next = torch.tensor(dataset_env["next_observations"], dtype=torch.float32)
    terminal = torch.tensor(dataset_env["terminals"], dtype=torch.float32).unsqueeze(dim=1)
    dataset_env = torch.utils.data.TensorDataset(state, action, reward, state_next, terminal)

    if args.model == "GP":
        Model = None  # TODO
    elif args.model == "EnsembleDeterministic":
        Model = models.NetDense
    elif args.model == "EnsembleProbabilisticHomoscedastic":
        Model = models.NetGaussHomo
    elif args.model == "EnsembleProbabilisticHeteroscedastic":
        Model = models.NetGaussHetero
    model = Model(
        dim_x=env.observation_space.shape[0] +
        env.action_space.shape[0],
        dim_y=1 + env.observation_space.shape[0],
        num_h=args.num_h_model,
        dim_h=args.dim_h_model,
        size_ensemble=args.size_ensemble_model,
        num_elites=args.num_elites_model,
        use_scalers=args.use_scalers,
    ).to(args.device)

    losses_model, scores_calibration = training.train_ensemble_map(model, dataset_env, args.weight_prior_model, args.lr_model, args.size_batch, args.device)
    loss_model = losses_model.mean()
    score_calibration = scores_calibration.mean()
    print(f"loss_model: {loss_model}, score_calibration: {score_calibration}")

    checkpoint = {
        "name_dataset": name_dataset,
        "losses_model": losses_model,
        "scores_calibration": scores_calibration,
        "model": model,
    }
    torch.save(checkpoint, args.path_checkpoint_model)

