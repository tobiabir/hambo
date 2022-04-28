import random
import sys
import torch

class DatasetNumpy(torch.utils.data.Dataset):
    
    def __init__(self):
        super().__init__()
        self.data = []

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def append(self, x):
        datum = torch.tensor(x, dtype=torch.float32)
        self.data.append(datum)

    def sample(self, num):
        batch = random.sample(self.data, num)
        batch = torch.stack(batch, dim=0)
        return batch 

class DatasetSARS(torch.utils.data.Dataset):
    
    def __init__(self, capacity=sys.maxsize):
        super().__init__()
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def append(self, state, action, reward, state_next, done):
        if len(self) < self.capacity:
            self.data.append(None)
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(dim=-1)
        state_next = torch.tensor(state_next, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(dim=-1)
        self.data[self.pos] = state, action, reward, state_next, done
        self.pos = (self.pos + 1) % self.capacity

    def concat(self, dataset):
        for datum in dataset.data:
            if len(self) < self.capacity:
                self.data.append(None)
            self.data[self.pos] = datum
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, num):
        batch = random.choices(self.data, k=num)
        batch = tuple(map(torch.stack, zip(*batch)))
        return batch 

