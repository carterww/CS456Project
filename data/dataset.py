import pandas as pd
import torch
from torch.utils.data import Dataset
import data.data_utils as du


class PollutionDataset(Dataset):
    def __init__(self, path, device=None):
        super(PollutionDataset, self).__init__()
        self.data = du.load_data_tens(path, False)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.data[idx, :, :-1], self.data[idx, :, -1])
