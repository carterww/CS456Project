# Need to split data into consecutive time windows with
# an equal number of cities in each window
# Can pad the windows for training with negatives or something
# and skip them in forward pass
import data_utils as du
import pandas as pd
import numpy as np
import torch
import geopy.distance as distance
import pickle


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
    df = du.load_data(data_filename)
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


def create_windows(df, ideal_window_size):
    """
    Create a list of windows from the given DataFrame.
    Each window is a list of city codes.
    :param df: The DataFrame to create windows from.
    Should already be preprocessed. Keep the citycode
    and date columns.
    :param ideal_window_size: The size of each window.
    May not be exactly this size.
    :return: A list of windows.
    """

