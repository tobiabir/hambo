# Hallucinated Control for Conservative Offline Policy Evaluation

This is the official repository of the "Hallucinated Control for Conservative Offline Policy Evaluation" project.
Check out our paper [here](https://arxiv.org/abs/2303.01076).

## Setup

To set up the environment for running our code, start by setting up your virtual environment using your favourite tool.
If for example you use venv run the following commands.
```
python3 -m venv .venv
source .venv/bin/activate
```

After that install the requirements.
```
pip install -r requirements.txt
```

## Datasets

We provide the datasets we use in the paper for the Pendulum and Hopper environment.
To use them unpack [datasets.tar.gz](datasets.tar.gz).
```
tar xvf datasets.tar.gz
```

To use your own data bring it into the same format as our datasets.

## Agents

We provide the agents we use in the paper to evaluate.
To use them unpack [agents.tar.gz](agents.tar.gz).
```
tar xvf agents.tar.gz
```

To use your own agents bring them into the same format as our agents.

## Training

To train a transition model on offline data use the [training_offline_model.py](training_offline_model.py) script.
The following is an example of how we use it to get our results in the paper.
```
python training_offline_model.py --paths_dataset Checkpoints/Datasets/checkpoint_hopper_1000000 --path_checkpoint_model Checkpoints/Models/checkpoint_hopper_1000000_svgd10.0_${seed} --model EnsembleProbabilisticHeteroscedastic --num_h_model 4 --size_ensemble 16 --num_elites_model 16 --use_scalers --activation_model ReLU --weight_prior_model 0.0001 --seed ${seed}
```

To train an agent on offline data use the [training_offline.py](training_offline.py) script.

## Evaluation

To evaluate a policy using a trained transition model from above use the [evaluation_offline.py](evaluation_offline.py) script.
The following is an example of how we use it to get our results in the paper.
```
python evaluation_offline.py --path_checkpoint_model Checkpoints/Models/checkpoint_hopper_1000000_svgd10.0_${seed} --max_length_rollout_model 200 --method_sampling DS --hallucinate --beta 1.0 --path_agent Checkpoints/Agents/checkpoint_hopper_148000 --tau 0.001 --learn_alpha --lr_agent 0.0001 --size_batch 1024 --num_rounds 1000 --interval_rollout_model 1 --num_episodes_rollout_model 1000 --num_steps_train_agent 1000 --interval_eval_agent 40 --num_episodes_eval 10000 --replay_size 2000000 --path_results Results/results_hopper_1000000_noaleatoric_svgd10.0_148000_200_ds_1.0_${seed} --seed ${seed}
```
