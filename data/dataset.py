import pandas as pd
import torch
from torch.utils.data import Dataset
import data_utils as du


class PollutionDataset(Dataset):
    def __init__(self):
        super(PollutionDataset, self).__init__()

    def __len__(self):
        return -1

    def __getitem__(self, item):
        return None
