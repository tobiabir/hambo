import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visualisation of Offline Policy Evaluation Results")

    # dataset arguments
    parser.add_argument("--paths_results", type=str, nargs="+", required=True,
                        help="paths of the results to use")

    # model environment arguments
    parser.add_argument("--method_sampling", default="DS", choices=["DS", "TS1", "TSInf"],
                        help="sampling method to use in model environment (see [Chua et al.](https://arxiv.org/abs/1805.12114) for explanation) (default: DS)")
    

    args = parser.parse_args()

    width_group = 0.8
    if args.method_sampling == "DS":
        size_group = 8
    else:
        size_group = 2
    width_bar = width_group / size_group 
    x_group = np.linspace(-0.5 * width_group + width_bar / 2, 0.5 * width_group - width_bar / 2, size_group)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:cyan"]

    for idx_path_result, path_result in enumerate(args.paths_results):
        result = torch.load(path_result)
        plt.bar(idx_path_result + x_group[0], result["return_eval_env"], width=width_bar, color=colors[0], label="true" if idx_path_result == 0 else "")
        if args.method_sampling == "DS":
            for idx_beta, beta in enumerate(result["return_eval_model"]):
                plt.bar(idx_path_result + x_group[idx_beta + 1], result["return_eval_model"][beta], width=width_bar, color=colors[idx_beta + 1], label=f"beta = {beta}" if idx_path_result == 0 else "")
        else:
            plt.bar(idx_path_result + x_group[1], result["return_eval_model"], width=width_bar, color=colors[1], label=f"pessimistic" if idx_path_result == 0 else "")
                
    
    plt.xticks(range(len(args.paths_results)))
    bottom, top = plt.ylim()
    plt.title("Offline Policy Evaluation " + args.method_sampling)
    plt.xlabel("Policy")
    plt.ylabel("Return")
    plt.legend()
    plt.show()
