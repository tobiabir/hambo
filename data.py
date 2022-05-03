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
        datum = self._to_tensor(*datum)
        return datum

    def __len__(self):
        return len(self.data)

    def _to_tensor(self, state, action, reward, state_next, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(dim=-1)
        state_next = torch.tensor(state_next, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(dim=-1)
        return state, action, reward, state_next, done

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
        batch = tuple(map(np.stack, zip(*batch)))
        return batch 

