import math
import numpy as np
import random
import torch

import agents
import rollout

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloader(dataset1, dataset2, num_batches, size_batch, ratio, num_workers=2):
    dataset = torch.utils.data.ConcatDataset((dataset1, dataset2))
    num_samples = num_batches * size_batch
    #sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=num_samples)
    sampler_batch = SamplerBatchRatio(len(dataset1), len(dataset2), num_batches, size_batch, ratio)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        #batch_size=size_batch,
        #sampler=sampler,
        batch_sampler=sampler_batch,
        num_workers=num_workers,
    )
    return dataloader


def get_scores_calibration(y_pred_means, y_pred_stds, y_train):
    cdfs = torch.distributions.Normal(y_pred_means, y_pred_stds).cdf(y_train)
    levels_confidence = torch.linspace(0, 1, 11, device=cdfs.device) 
    num_preds = cdfs.shape[1] * cdfs.shape[2]
    levels_confidence_empirical = [(cdfs < p).sum(dim=(1, 2)) / num_preds for p in levels_confidence]
    levels_confidence_empirical = torch.stack(levels_confidence_empirical, dim=1)
    scores = ((levels_confidence_empirical - levels_confidence)**2).sum(dim=1)
    return scores


def startup(env, agent, dataset, dataset_states_initial, num_steps_rollout):
    if num_steps_rollout <= 0:
        return
    agent_random = agents.AgentRandom(env.action_space)
    rollout.rollout_steps(env, agent_random, dataset, dataset_states_initial, num_steps_rollout)

class ScalerStandard():
    
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        return data
        return (data - self.mean) / self.std

    def inverse_transform(self, data_mean, data_std):
        return data_mean, data_std
        mean = self.mean + data_mean * self.std
        std = data_std * self.std
        return mean, std


def preprocess(model, dataset, device="cpu"):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
    state, action, reward, state_next, _ = next(iter(dataloader))
    x = torch.cat((state, action), dim=-1).to(device)
    y = torch.cat((reward, state_next - state), dim=-1).to(device)
    model.scaler_x.fit(x)
    model.scaler_y.fit(y)


class SamplerBatchRatio():

    def __init__(self, len1, len2, num_batches, size_batch, ratio):
        self.len1 = len1
        self.len2 = len2
        self.num_batches = num_batches
        if len2 == 0:
            self.size_batch1 = size_batch
        else:
            self.size_batch1 = int(ratio * size_batch)
        self.size_batch2 = size_batch - self.size_batch1

    def __iter__(self):
        for idx_batch in range(self.num_batches):
            idxs1 = random.choices(range(self.len1), k=self.size_batch1)
            idxs2 = random.choices(range(self.len1, self.len1 + self.len2), k=self.size_batch2)
            idxs = idxs1 + idxs2
            yield idxs

    def __len__(self):
        return self.num_batches


class SchedulerLinear():

    def __init__(self, val_init, val_target, step_start, step_end):
        self.val_init = val_init
        self.val_target=  val_target
        self.step_start = step_start
        self.step_end = step_end
        self.step_curr = -1
    
    def next(self):
        self.step_curr += 1
        
        if self.step_curr >= self.step_end:
            return self.val_target
        elif self.step_curr <= self.step_start:
            return self.val_init
        else:
            return self.val_init + (self.val_target - self.val_init) * ((self.step_curr - self.step_start) * 1.0 / (self.step_end - self.step_start))


class Wrapper:
      
    def __init__(self, obj):
        self.obj = obj
          
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.obj, attr)
