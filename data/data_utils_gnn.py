# Need to split data into consecutive time windows with
# an equal number of cities in each window
# Can pad the windows for training with negatives or something
# and skip them in forward pass
import data.data_utils as du
from sortedcontainers import SortedDict
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


def preprocess_dataframe(df, min_window_size=3, max_window_size=15):
    """
    Preprocess the given DataFrame.
    :param df: The DataFrame to preprocess.
    :return: The preprocessed DataFrame.
    """
    df = du.preprocess_dataframe(df)
    windows, date_list = create_windows(df, min_window_size, max_window_size)
    dfs = []
    for window in windows:
        start, end = window
        dfs.append(df[(df['date'] >= start) & (df['date'] <= end)])
        dfs[-1] = dfs[-1].reset_index(drop=True)
    return dfs


def create_windows(df, min_window_size, max_window_size):
    """
    Create a list of windows from the given DataFrame.
    Each window is a tuple containing the start and end dates.
    Windows will be consecutive days with the same cities.
    Length of windows will be between min_window_size and
    max_window_size, but they will not be equal.
    :param df: The DataFrame to create windows from.
    Should already be preprocessed. Keep the citycode
    and date columns.
    :param min_window_size: The minimum size of a window.
    :param max_window_size: The maximum size of a window.
    :return: A list of windows and a date dictionary.
    """
    # Create a sorted list of dates
    # Each date is a list of rows
    date_list = create_date_list(df)
    start, end = date_list.peekitem(0)[0], None
    curr_window_city_codes = get_city_code_day(date_list, start)
    windows = []
    prev = None
    for key in date_list:
        end = key
        new_city_codes = get_city_code_day(date_list, end)
        if prev is not None and (end - prev).days > 1:
            windows.append((start, prev))
            start = end
            curr_window_city_codes = new_city_codes
            prev = end
            continue
        if not same_cities(curr_window_city_codes, new_city_codes):
            windows.append((start, prev))
            start = end
            curr_window_city_codes = new_city_codes
            prev = end
            continue
        if (end - start).days >= max_window_size:
            windows.append((start, prev))
            start = end
            curr_window_city_codes = new_city_codes
            prev = end
            continue
        prev = end
    windows.append((start, end))
    windows = filter(lambda x: (x[1] - x[0]).days >= min_window_size, windows)
    windows = list(windows)
    return windows, date_list


def get_city_code_day(dict, date):
    """
    Get the city codes for the given date.
    :param dict: The dictionary to get the city code from.
    :param date: The date to get the city code for.
    :return: The city code for the given date.
    """
    city_codes = []
    for item in dict[date]:
        if item.citycode not in city_codes:
            city_codes.append(item.citycode)
    return city_codes


def same_cities(city_codes1, city_codes2):
    """
    Check if the given city codes are the same.
    :param city_codes1: The first list of city codes.
    :param city_codes2: The second list of city codes.
    :return: True if the city codes are the same, False otherwise.
    """
    if len(city_codes1) != len(city_codes2):
        return False
    for i in range(len(city_codes1)):
        found = False
        for j in range(len(city_codes2)):
            if city_codes1[i] == city_codes2[j]:
                found = True
                continue
        if not found:
            return False
    return True

def create_date_list(df):
    dic = SortedDict()
    for i, row in df.iterrows():
        if row['date'] not in dic:
            dic[row['date']] = []
        # Add the row to the date
        dic[row['date']].append(row)
    return dic
