import math
import torch

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

def get_dataloader(dataset, num_batches, size_batch, num_workers=2):
    num_samples = num_batches * size_batch
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=size_batch,
        sampler=sampler,
        num_workers=num_workers,
    )
    return dataloader
    

class Wrapper:
      
    def __init__(self, obj):
        self.obj = obj
          
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.obj, attr)
