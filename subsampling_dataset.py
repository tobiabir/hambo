import argparse
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset Generation")
    parser.add_argument("--path_dataset_in", type=str,
                        help="path to load the dataset from (if None the random agent is used) (default: None)")
    parser.add_argument("--num_steps", type=int, default=1000000,
                        help="number of steps to have in the dataset (default: 1000000)")
    parser.add_argument("--path_dataset_out", type=str,
                        help="path to store the dataset to (default: None)")
    args = parser.parse_args()

    checkpoint = torch.load(args.path_dataset_in)
    checkpoint["dataset"].data = checkpoint["dataset"].data[:args.num_steps]
    torch.save(checkpoint, args.path_dataset_out)
