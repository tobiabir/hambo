import numpy as np
import random
import sys
import torch

class DatasetNumpy(torch.utils.data.Dataset):
    
    def __init__(self):
        super().__init__()
        self.data = []

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def append(self, x):
        self.data.append(x)

    def append_batch(self, batch):
        self.data += batch

    def sample(self, num, replacement=True):
        if replacement:
            batch = random.choices(self.data, k=num)
        else:
            batch = random.sample(self.data, k=num)
        batch = np.stack(batch)
        return batch


class DatasetSARS(torch.utils.data.Dataset):
    
    def __init__(self, capacity=sys.maxsize):
        super().__init__()
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def __getitem__(self, idx):
        datum = self.data[idx]
        return datum

    def __len__(self):
        return len(self.data)

    def push(self, state, action, reward, state_next, done):
        if len(self) < self.capacity:
            self.data.append(None)
        self.data[self.pos] = state, action, reward, state_next, done
        self.pos = (self.pos + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.data) < self.capacity:
            length_append = min(self.capacity - len(self.data), len(batch))
            self.data += [None] * length_append

        if self.pos + len(batch) < self.capacity:
            self.data[self.pos : self.pos + len(batch)] = batch
            self.pos += len(batch)
        else:
            self.data[self.pos : len(self.data)] = batch[:len(self.data) - self.pos]
            self.data[:len(batch) - len(self.data) + self.pos] = batch[len(self.data) - self.pos:]
            self.pos = len(batch) - len(self.data) + self.pos

    def sample(self, num, replacement=True):
        if replacement:
            batch = random.choices(self.data, k=num)
        else:
            batch = random.sample(self.data, k=num)
        batch = list(map(np.stack, zip(*batch)))
        return batch 


class ScalerStandard():
    
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.active = True

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def fit(self, data, device="cpu"):
        self.mean = torch.mean(data, dim=0).to(device)
        self.std = torch.std(data, dim=0).to(device)

    def transform(self, data):
        if not self.active:
            return data
        return (data - self.mean) / self.std

    def inverse_transform(self, data_mean, data_std):
        if not self.active:
            return data_mean, data_std
        mean = self.mean + data_mean * self.std
        std = data_std * self.std
        return mean, std


class SamplerBatchRatio():

    def __init__(self, len1, len2, num_batches, size_batch, ratio):
        self.len1 = len1
        self.len2 = len2
        self.num_batches = num_batches
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


def preprocess(model, dataset, device="cpu"):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
    state, action, reward, state_next, _ = next(iter(dataloader))
    reward = reward.unsqueeze(dim=1)
    x = torch.cat((state, action[0]), dim=-1)
    y = torch.cat((reward, state_next - state), dim=-1)
    model.scaler_x.fit(x, device)
    model.scaler_y.fit(y, device)


def get_dataloader(dataset1, dataset2, num_batches, size_batch, ratio=1.0, num_workers=1):
    if dataset2 is None or len(dataset2) == 0 or ratio == 1.0:
        num_samples = num_batches * size_batch
        sampler = torch.utils.data.RandomSampler(dataset1, replacement=True, num_samples=num_samples)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset1,
            batch_size=size_batch,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        dataset = torch.utils.data.ConcatDataset((dataset1, dataset2))
        sampler_batch = SamplerBatchRatio(len(dataset1), len(dataset2), num_batches, size_batch, ratio)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=sampler_batch,
            num_workers=num_workers,
        )
    return dataloader

