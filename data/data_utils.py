import pandas as pd
import numpy as np
import torch
import geopy.distance as distance
import pickle


def load_data(filename):
    """
    Load the data from the given filename.
    :param filename: The name of the file to load the data from.
    :return: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(filename, header=0, parse_dates=['date'])


def preprocess_dataframe(df, keep_cols=None):
    """
    Preprocess the given DataFrame by normalizing columns,
    dropping columns, encoding data, etc.
    :param df: The DataFrame to preprocess.
    :param keep_cols: The columns to keep in the DataFrame.
    If None, keep all columns that are not always dropped.
    :return: The preprocessed DataFrame.
    """
    city = df['citycode']
    target = df['PM25']
    df = df.drop(columns=['cityname', 'year', 'location', 'longitude',
                          'latitude', 'station', 'sunshinehours',
                          'citycode', 'province', 'PM25'])
    if keep_cols is not None:
        df = df[keep_cols]
    # This should be done on a col by col basis, change later
    df.fillna(0, inplace=True)
    # drop cyclical columns and fix them
    cyclic_df = df[['month', 'date', 'season']]
    df = df.drop(columns=['month', 'day', 'season', 'date'])
    cyclic_df = transform_cyclical(cyclic_df)
    # normalize the data
    numeric_features = df.select_dtypes(include=[np.number])
    numeric_features = numeric_features.columns
    df[numeric_features] = df[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # encode the categorical data
    categorical_features = df.select_dtypes(include=[np.object])
    categorical_features = categorical_features.columns
    df = pd.get_dummies(df, columns=categorical_features)
    # add the cyclical columns back in
    # also add cityname so it can be split later
    df = pd.concat([city, df, cyclic_df, target], axis=1)
    return df


def transform_cyclical(df):
    """
    Transform the cyclical columns in the given DataFrame
    into a cyclical representation.
    :param df: The DataFrame to transform.
    :return: The transformed DataFrame.
    """
    # Use day of year: 1-365
    # df['day'] = df['date'].apply(lambda x: int(x.strftime('%j')))
    df['day'] = df['date']
    # df['month_sin'] = df['month'].apply(lambda x: np.sin(2 * np.pi * x / 12))
    # df['month_cos'] = df['month'].apply(lambda x: np.cos(2 * np.pi * x / 12))
    # df['day_sin'] = df['day'].apply(lambda x: np.sin(2 * np.pi * x / 31))
    # df['day_cos'] = df['day'].apply(lambda x: np.cos(2 * np.pi * x / 31))
    df['season_sin'] = df['season'].apply(lambda x: np.sin(2 * np.pi * x / 4))
    df['season_cos'] = df['season'].apply(lambda x: np.cos(2 * np.pi * x / 4))
    df = df.drop(columns=['month', 'date', 'season', 'day'])
    return df


def build_distance_matrix(data_filename, save_filename=None):
    """
    Build a distance matrix from the given DataFrame.
    Gives the distance between each city.
    :param df: The DataFrame to build the distance matrix from.
    :return: A tuple containing the distance matrix and a dictionary
    mapping city codes to indices in the distance matrix.
    """
    if save_filename is not None:
        try:
            # Load from file if possible
            return (torch.load(save_filename + '.tens'),
                    pickle.load(open(save_filename + '.dict', 'rb')))
        except FileNotFoundError:
            pass
    df = load_data(data_filename)
    df = df.filter(['citycode', 'longitude', 'latitude'])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by=['citycode'])
    tensor = torch.zeros((df.shape[0], df.shape[0]))
    city_code_dict = {}
    for i, row in df.iterrows():
        city_code_dict[row['citycode']] = i
    for i, row in df.iterrows():
        for j, row2 in df.iterrows():
            if i == j:
                continue
            if tensor[i][j] != 0:
                continue
            city1 = (row['latitude'], row['longitude'])
            city2 = (row2['latitude'], row2['longitude'])
            dist = distance.distance(city1, city2).km
            tensor[i][j] = dist
            tensor[j][i] = dist
    if save_filename is not None:
        # No point in building these every time
        torch.save(tensor, save_filename + '.tens')
        pickle.dump(city_code_dict, open(save_filename + '.dict', 'wb'))
    return (tensor, city_code_dict)


def prep_df_for_graph(df):
    """
    This function takes a preprocessed dataframe returns a tensor.
    Each dataframe represents a date and contains the data for all cities.
    This will make constructing the graphs for each day easier.
    :param df: The preprocessed dataframe.
    :return: A tensor of shape (I need to figure it out).
    """
    return df


def df_to_tensors(df):
    """
    Convert the given DataFrame to a tensor. Splits the DataFrame
    by city and pads the dataframes to the same length.
    :param df: The DataFrame to convert.
    :return: A tensor containing the data from the DataFrame.
    """
    dfs = []
    curr_city = df['citycode'][0]
    curr_city_start = 0
    desired_seq_len = 15
    # split the dataframe into multiple dataframes by city
    for i, row in df.iterrows():
        if row['citycode'] != curr_city:
            dfs.append(df.iloc[curr_city_start:i])
            curr_city = row['citycode']
            curr_city_start = i
    dfs.append(df.iloc[curr_city_start:])

    for i, df in enumerate(dfs):
        dfs[i] = df.drop(columns=['citycode'])

    # split the dataframes into sequences of length desired_seq_len
    new_dfs = []
    for i, df in enumerate(dfs):
        for j in range(0, df.shape[0], desired_seq_len):
            if j + desired_seq_len > df.shape[0]:
                break
            new_dfs.append(df.iloc[j:j + desired_seq_len, :])
    dfs = new_dfs
    """
    max_len = max([df.shape[0] for df in dfs])
    # pad the dataframes to the same length
    for i, df in enumerate(dfs):
        dfs[i] = df.reindex(range(max_len), fill_value=0)
    min_len = min([df.shape[0] for df in dfs])
    # truncate the dataframes to the same length
    for i, df in enumerate(dfs):
        dfs[i] = df.iloc[:min_len]
    """
    # convert the dataframes to tensors
    arr = np.array([df.values for df in dfs])
    tens = torch.tensor(arr, dtype=torch.float32)
    return tens


def load_data_tens(csv_filename, prefer_tens_file=True, do_not_cache=False):
    """
    Load the data from the given filename. If a .tens file exists
    and prefer_tens_file is True, load the data from the .tens file.
    Otherwise, load the data from the csv file and convert it to a
    tensor. If do_not_cache is True, do not save the tensor to a .tens
    file.
    :param csv_filename: The name of the file to load the data from.
    :param prefer_tens_file: Whether to prefer loading the data from
    the .tens file.
    :param do_not_cache: Whether to save the data to a .tens file.
    :return: A tensor containing the loaded data.
    """
    if prefer_tens_file:
        try:
            return torch.load(csv_filename + '.tens')
        except FileNotFoundError:
            pass
    df = load_data(csv_filename)
    df = preprocess_dataframe(df)
    tens = df_to_tensors(df)
    if not do_not_cache:
        torch.save(tens, csv_filename + '.tens')
    return tens
