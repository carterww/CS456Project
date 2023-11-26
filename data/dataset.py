import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import data.data_utils as du
import data.data_utils_gnn as du_gnn


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


class PollutionDatasetGNN(Dataset):
    def __init__(self, path, min_seq_size, max_seq_size, should_cache=True, device=None):
        super(PollutionDatasetGNN, self).__init__()
        if not os.path.exists('cache') or not os.path.isdir('cache'):
            os.mkdir('cache')

        found_cached = self.load_cached_dataset('cache', str(min_seq_size), str(max_seq_size))
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.distance_matrix, self.city_code_dict = du_gnn.build_distance_matrix(path, 'cache/distance_matrix')
        if not found_cached:
            self.raw_data = du.load_data(path)
            self.dataframes = du_gnn.preprocess_dataframe(self.raw_data, min_seq_size, max_seq_size)
            if should_cache:
                self.cache_dataset('cache', str(min_seq_size), str(max_seq_size))
        self.cached_data = []
        for i in range(len(self.dataframes)):
            self.cached_data.append(None)

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        if type(idx) is not int:
            raise TypeError
        if self.cached_data[idx] is not None:
            return self.cached_data[idx]
        df = self.dataframes[idx]
        unique_days_count = df['date'].nunique()
        city_code_count = df['citycode'].nunique()
        city_codes = df['citycode'].unique()
        codes = torch.zeros((unique_days_count, city_code_count))
        x = torch.zeros((unique_days_count, city_code_count, df.shape[1] - 3))
        y = torch.zeros((unique_days_count, city_code_count))
        curr_date = df['date'].min()
        for i in range(unique_days_count):
            for j in range(city_code_count):
                tmp = df[(df['date'] == curr_date) & (df['citycode'] == city_codes[j])]
                if tmp.shape[0] == 0:
                    print('Error: no data for city ' + str(city_codes[j]) + ' on date ' + str(curr_date))
                codes[i][j] = city_codes[j]
                x[i][j] = torch.tensor(tmp.drop(columns=['date', 'target', 'citycode']).values)
                y[i][j] = tmp['target'].values[0]
            curr_date = str(pd.to_datetime(curr_date) + pd.DateOffset(days=1)).split(' ')[0]
        x = x.to(self.device)
        y = y.to(self.device)
        if self.cached_data[idx] is None:
            self.cached_data[idx] = (x, y, codes)
        return (x, y, codes)

    def cache_dataset(self, cache_path, min_window_size, max_window_size):
        """
        Cache the dataset to the given path.
        :param cache_path: The path to cache the dataset to.
        :param min_window_size: The minimum window size to use.
        :param max_window_size: The maximum window size to use.
        :return: None.
        """
        part = '-' + min_window_size + '-' + max_window_size
        for i, df in enumerate(self.dataframes):
            df.to_csv(cache_path + '/df_' + str(i) + part + '.csv')

    def load_cached_dataset(self, cache_path, min_window_size, max_window_size):
        """
        Load the cached dataset from the given path.
        :param cache_path: The path to load the dataset from.
        :param min_window_size: The minimum window size to use.
        :param max_window_size: The maximum window size to use.
        :return: None.
        """
        self.dataframes = []
        is_cached = True
        part = '-' + min_window_size + '-' + max_window_size
        i = 0
        while True:
            try:
                df = pd.read_csv(cache_path + '/df_' + str(i) + part + '.csv')
            except FileNotFoundError:
                if i == 0:
                    is_cached = False
                break
            self.dataframes.append(df)
            i += 1
        return is_cached
