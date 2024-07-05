import numpy as np
import torch

class HooktheoryDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.data_x = np.load("data/data_x.npy").astype(np.float32)
    self.prev_x = np.load("data/prev_x.npy").astype(np.float32)

  def __len__(self):
    return self.data_x.shape[0]

  def __getitem__(self, idx):
    return self.prev_x[idx], self.data_x[idx]