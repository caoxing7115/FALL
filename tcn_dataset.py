import torch
from torch.utils.data import Dataset
import numpy as np


class TCNDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).permute(1, 0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
