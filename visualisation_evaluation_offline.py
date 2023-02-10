import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot(title, paths_results, ticks_x, label_x, show_legend=False, path_visual=None):
    width_group = 0.8
    size_group = 7
    width_bar = width_group / size_group 
    x_group = np.linspace(-0.5 * width_group + width_bar, 0.5 * width_group, size_group)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:cyan"]

    plt.rcParams.update({'font.size': 12})
    plt.subplots_adjust(left=0.12, bottom=0.13, right=0.93)

    for idx_path_result, path_result in enumerate(paths_results):
        result = torch.load(path_result)

        # true
        plt.axhline(result["tsinf"]["return_eval_env_mean"], color="black", linestyle="--", label="true" if idx_path_result == 0 else "")
        plt.axhline(np.NaN, color="none", label=" " if idx_path_result == 0 else "")
        plt.axhline(np.NaN, color="none", label=" " if idx_path_result == 0 else "")

        # tsinf
        errorbar = plt.errorbar(idx_path_result + x_group[0], result["tsinf"]["return_eval_model_mean_mean"], yerr=2*(result["tsinf"]["return_eval_model_mean_std"] + result["tsinf"]["return_eval_model_std_mean"]), marker="s", markerfacecolor=colors[1], markeredgecolor="black", ecolor=colors[0], elinewidth=width_bar * 20, capsize=width_bar * 40)
        errorbar[-1][0].set_linestyle("--")
        errorbar = plt.errorbar(idx_path_result + x_group[0], result["tsinf"]["return_eval_model_mean_mean"], yerr=2*result["tsinf"]["return_eval_model_mean_std"], marker="s", markerfacecolor=colors[0], markeredgecolor="black", ecolor=colors[0], capsize=width_bar * 20)
        errorbar[-1][0].set_linewidth(0)
        plt.errorbar(idx_path_result + x_group[1], result["tsinf"]["return_eval_model_pessimistic_mean"], yerr=[[result["tsinf"]["return_eval_model_pessimistic_mean"] - result["tsinf"]["return_eval_model_pessimistic_min"]], [result["tsinf"]["return_eval_model_pessimistic_max"] - result["tsinf"]["return_eval_model_pessimistic_mean"]]], marker="D", markerfacecolor=colors[0], markeredgecolor="black", ecolor=colors[0], elinewidth=width_bar * 20, capsize=width_bar * 40)
        errorbar[-1][0].set_linestyle("-")

        # ts1
        errorbar = plt.errorbar(idx_path_result + x_group[2], result["ts1_neutral"]["return_eval_model_mean"], yerr=2*result["ts1_neutral"]["return_eval_model_std"], marker="s", markerfacecolor=colors[1], markeredgecolor="black", ecolor=colors[1], elinewidth=width_bar * 20, capsize=width_bar * 40)
        errorbar[-1][0].set_linestyle("--")
        errorbar = plt.errorbar(idx_path_result + x_group[3], result["ts1_pessimistic"]["return_eval_model_mean"], yerr=[[result["ts1_pessimistic"]["return_eval_model_mean"] - result["ts1_pessimistic"]["return_eval_model_min"]], [result["ts1_pessimistic"]["return_eval_model_max"] - result["ts1_pessimistic"]["return_eval_model_mean"]]], marker="D", markerfacecolor=colors[1], markeredgecolor="black", ecolor=colors[1], elinewidth=width_bar * 20, capsize=width_bar * 40)
        errorbar[-1][0].set_linestyle("-")

        # ds
        errorbar = plt.errorbar(idx_path_result + x_group[4], result["ds_neutral"]["return_eval_model_mean"], yerr=2*result["ds_neutral"]["return_eval_model_std"], marker="s", markerfacecolor=colors[2], markeredgecolor="black", ecolor=colors[2], elinewidth=width_bar * 20, capsize=width_bar * 40)
        errorbar[-1][0].set_linestyle("--")
        errorbar = plt.errorbar(idx_path_result + x_group[5], result["ds_pessimistic"]["return_eval_model_mean"], yerr=[[result["ds_pessimistic"]["return_eval_model_mean"] - result["ds_pessimistic"]["return_eval_model_min"]], [result["ds_pessimistic"]["return_eval_model_max"] - result["ds_pessimistic"]["return_eval_model_mean"]]], marker="D", markerfacecolor=colors[2], markeredgecolor="black", ecolor=colors[2], elinewidth=width_bar * 20, capsize=width_bar * 40)
        errorbar[-1][0].set_linestyle("-")

        plt.axvline(x=0.5, color="black", linewidth=0.1)
        plt.axvline(x=1.5, color="black", linewidth=0.1)

        plt.fill_between([-0.5,len(paths_results)-0.5], [2.0,2.0], [1.0,1.0], color="white", edgecolor="red", hatch="//")

        plt.bar(np.NaN, np.NaN, color=colors[0], edgecolor="black", label="tsinf" if idx_path_result == 0 else "")
        plt.bar(np.NaN, np.NaN, color=colors[1], edgecolor="black", label="ts1" if idx_path_result == 0 else "")
        plt.bar(np.NaN, np.NaN, color=colors[2], edgecolor="black", label="ds" if idx_path_result == 0 else "")
        plt.scatter(np.NaN, np.NaN, color="none", marker="s", edgecolor="black", label="neutral (baselines)" if idx_path_result == 0 else "")
        plt.scatter(np.NaN, np.NaN, color="none", marker="D", edgecolor="black", label="conservative (ours)" if idx_path_result == 0 else "")
    
    plt.title(title, fontsize=18)

    plt.xticks(range(len(paths_results)), ticks_x)
    plt.xlim(left=-0.5, right=len(paths_results)-0.5)

    plt.xlabel(label_x, fontsize=18)

    plt.yticks(np.arange(-0.5, 1.75, 0.5))
    plt.ylim(bottom=-0.75, top=1.5)
    plt.ylabel("Return", fontsize=15)
    plt.gca().yaxis.set_label_coords(-0.08,0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 2, 5, 6, 7, 3, 4]

    if show_legend:
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=3, fontsize=13)

    if path_visual is None:
        plt.show()
    else:
        plt.savefig(path_visual, format="pdf")
    plt.clf()

if __name__ == "__main__":
    
    dir_results = "Results"
    dir_visuals = "Visuals"

    label_x_data = "Dataset Size $|D_b|$"
    label_x_horizon = "Horizon $T$"

    configs = []

    # pendulum data
    config = {
        "title": "Pendulum; Horizon $T = 200$",
        "paths_results": ["results_pendulum_10000_noaleatoric_svgd10.0_25000_200", "results_pendulum_100000_noaleatoric_svgd10.0_25000_200", "results_pendulum_1000000_noaleatoric_svgd10.0_25000_200"],
        "ticks_x": ["$10^4$", "$10^5$", "$10^6$"],
        "label_x": label_x_data,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_pendulum_data_noaleatoric.pdf")
    }
    configs.append(config)
    config = {
        "title": "Pendulum; Horizon $T = 200$",
        "paths_results": ["results_pendulum_10000_svgd10.0_25000_200", "results_pendulum_100000_svgd10.0_25000_200", "results_pendulum_1000000_svgd10.0_25000_200"],
        "ticks_x": ["$10^4$", "$10^5$", "$10^6$"],
        "label_x": label_x_data,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_pendulum_data.pdf")
    }
    configs.append(config)

    # pendulum horizon
    config = {
        "title": "Pendulum; $|D_b| = 10^5$",
        "paths_results": ["results_pendulum_100000_noaleatoric_svgd10.0_25000_100", "results_pendulum_100000_noaleatoric_svgd10.0_25000_200", "results_pendulum_100000_noaleatoric_svgd10.0_25000_400"],
        "ticks_x": [100, 200, 400],
        "label_x": label_x_horizon,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_pendulum_horizon_noaleatoric.pdf")
    }
    configs.append(config)
    config = {
        "title": "Pendulum; $|D_b| = 10^5$",
        "paths_results": ["results_pendulum_100000_svgd10.0_25000_100", "results_pendulum_100000_svgd10.0_25000_200", "results_pendulum_100000_svgd10.0_25000_400"],
        "ticks_x": [100, 200, 400],
        "label_x": label_x_horizon,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_pendulum_horizon.pdf")
    }
    configs.append(config)

    # hopper data
    config = {
        "title": "Hopper; Horizon $T = 200$",
        "paths_results": ["results_hopper_10000_noaleatoric_svgd10.0_148000_200", "results_hopper_100000_noaleatoric_svgd10.0_148000_200", "results_hopper_1000000_noaleatoric_svgd10.0_148000_200"],
        "ticks_x": ["$10^4$", "$10^5$", "$10^6$"],
        "label_x": label_x_data,
        "show_legend": True,
        "path_visual": os.path.join(dir_visuals, "visual_ope_hopper_data_noaleatoric.pdf")
    }
    configs.append(config)
    config = {
        "title": "Hopper; Horizon $T = 200$",
        "paths_results": ["results_hopper_10000_svgd10.0_148000_200", "results_hopper_100000_svgd10.0_148000_200", "results_hopper_1000000_svgd10.0_148000_200"],
        "ticks_x": ["$10^4$", "$10^5$", "$10^6$"],
        "label_x": label_x_data,
        "show_legend": True,
        "path_visual": os.path.join(dir_visuals, "visual_ope_hopper_data.pdf")
    }
    configs.append(config)

    # hopper horizon
    config = {
        "title": "Hopper; $|D_b| = 10^5$",
        "paths_results": ["results_hopper_100000_noaleatoric_svgd10.0_148000_100", "results_hopper_100000_noaleatoric_svgd10.0_148000_200", "results_hopper_100000_noaleatoric_svgd10.0_148000_400"],
        "ticks_x": [100, 200, 400],
        "label_x": label_x_horizon,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_hopper_horizon_noaleatoric.pdf")
    }
    configs.append(config)
    config = {
        "title": "Hopper; $|D_b| = 10^5$",
        "paths_results": ["results_hopper_100000_svgd10.0_148000_100", "results_hopper_100000_svgd10.0_148000_200", "results_hopper_100000_svgd10.0_148000_400"],
        "ticks_x": [100, 200, 400],
        "label_x": label_x_horizon,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_hopper_horizon.pdf")
    }
    configs.append(config)

    # halfcheetah data
    config = {
        "title": "HalfCheetah; Horizon $T = 100$",
        "paths_results": ["results_halfcheetah_5_100000_noaleatoric_svgd10.0_wpm0.01_1000000_100", "results_halfcheetah_5_1000000_noaleatoric_svgd10.0_wpm0.01_1000000_100", "results_halfcheetah_5_4000000_noaleatoric_svgd10.0_wpm0.01_1000000_100"],
        "ticks_x": ["$10^5$", "$10^6$", r"$4 \times 10^6$"],
        "label_x": label_x_data,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_halfcheetah_data_noaleatoric.pdf")
    }
    configs.append(config)

    # halfcheetah horizon
    config = {
        "title": r"Hopper; $|D_b| = 4 \times 10^6$",
        "paths_results": ["results_halfcheetah_5_4000000_noaleatoric_svgd10.0_wpm0.01_1000000_100", "results_halfcheetah_5_4000000_noaleatoric_svgd10.0_wpm0.01_1000000_150"],
        "ticks_x": [100, 150],
        "label_x": label_x_horizon,
        "show_legend": False,
        "path_visual": os.path.join(dir_visuals, "visual_ope_halfcheetah_horizon_noaleatoric.pdf")
    }
    configs.append(config)

    # prepend results directory to lists of result paths
    for idx_config in range(len(configs)):
        for idx_path_result in range(len(configs[idx_config]["paths_results"])):
            configs[idx_config]["paths_results"][idx_path_result] = os.path.join(dir_results, configs[idx_config]["paths_results"][idx_path_result])

    for config in configs:
        plot(**config)
