import pandas as pd
from os import listdir
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# TODO change to command line argument
ORIGINAL_DATA_DIRECTORY = '../data/original/'
PREPROCESSED_DATA_DIRECTORY = '../data/preprocessed/'


def load_original_data(file_name):
    """
    loads in a file with forex pair data and returns a pandas dataframe
    """
    column_names = ['date', 'open', 'high', 'low', 'close', 'volume']
    for index in range(len(column_names)):
        if column_names[index] != 'date':
            forex_pair_name = file_name[:6].lower()
            column_names[index] = forex_pair_name + '_' + column_names[index]
    forex_pair_df = pd.read_csv(ORIGINAL_DATA_DIRECTORY + file_name, delimiter='\t', names=column_names)
    # Remove first row of original names
    forex_pair_df = forex_pair_df.iloc[1:]
    return forex_pair_df


def merge_original_data():
    """
    merges all the forex pair dataframes and removes any row with missing data
    """
    merged_df = pd.DataFrame()
    for file_name in listdir(ORIGINAL_DATA_DIRECTORY):
        forex_pair_df = load_original_data(file_name)
        if merged_df.empty:
            merged_df = forex_pair_df
        else:
            merged_df = merged_df.merge(forex_pair_df, on='date')
    merged_df.to_csv(PREPROCESSED_DATA_DIRECTORY + 'merged.csv', index=False)
    return merged_df


def load_data(filename):
    """
    loads merged forex pair data
    """
    merged_df = pd.read_csv(filename, delimiter=',')
    return merged_df


def get_forex_pair_names():
    """
    gets all forex pair names
    """
    merged_data = load_data(PREPROCESSED_DATA_DIRECTORY + 'close.csv')
    column_names = merged_data.columns
    forex_pair_names = set()
    for column_name in column_names:
        if column_name != 'date':
            forex_pair_names.add(column_name[:6])
    return list(forex_pair_names)


def transform(forex_df,
              drop_date=True,
              drop_vol=True,
              only_closing=True,
              min_max_scale=True,
              sort_order=True,
              pct_change=False,
              match_corr=True):
    """
    transforms forex pair df
    """

    if drop_date:
        forex_df = forex_df.drop(columns='date')
    if drop_vol:
        col_names = forex_df.columns
        exclude_col_names = [c for c in col_names if 'vol' in c]
        forex_df = forex_df.drop(columns=exclude_col_names)
    if only_closing:
        col_names = forex_df.columns
        exclude_col_names = [c for c in col_names if 'close' not in c]
        forex_df = forex_df.drop(columns=exclude_col_names)

        if sort_order:
            forex_pairs_graph, _ = get_forex_graph(get_forex_pair_names())
            forex_pairs_sorted = [x[0] for x in sorted(forex_pairs_graph.degree, key=lambda x: x[1], reverse=True)]
            forex_pairs_close = [s + "_close" for s in forex_pairs_sorted]
            forex_df = forex_df[forex_pairs_close]

    if pct_change:
        forex_df = forex_df.pct_change().iloc[1:]

    if min_max_scale:
        scaler = MinMaxScaler()
        forex_df[forex_df.columns] = scaler.fit_transform(forex_df[forex_df.columns])

    if match_corr:
        close_df_corr = forex_df.corr(method="pearson")
        invert_names = close_df_corr["audusd_close"].where(close_df_corr["audusd_close"] < 0).dropna().index
        for name in invert_names:
            currency_a = name[:3]
            currency_b = name[3:6]
            forex_df[name] = forex_df[name].apply(lambda x: 1 - x)
            forex_df = forex_df.rename(columns={name: currency_b + currency_a + "_close"})

    forex_df.to_csv(PREPROCESSED_DATA_DIRECTORY + 'close.csv', index=False, header=True)

    return forex_df


def split_data(forex_df, split=[0, 0.6, 0.8, 1]):
    """
    Splits data into train, test and validation
    """

    split_indexes = [int(s * forex_df.shape[0]) for s in split]
    train_data = forex_df[split_indexes[0]:split_indexes[1]]
    validation_data = forex_df[split_indexes[1]:split_indexes[2]]
    test_data = forex_df[split_indexes[2]:split_indexes[3]]

    return train_data, validation_data, test_data


def get_forex_graph(names=get_forex_pair_names()):
    """
    returns the forex graph and adjacency matrix
    """
    forex_graph = nx.Graph()
    for name in names:
        forex_graph.add_node(name)

    for name1 in names:
        for name2 in names:
            if name1 != name2 and (name1[:3] in name2 or name1[3:] in name2):
                forex_graph.add_edge(name1, name2)

    forex_adjacency_matrix = nx.adjacency_matrix(forex_graph, nodelist=names).todense()
    forex_connections_matrix = np.asarray(forex_adjacency_matrix + np.identity(len(names)), dtype=bool)

    return forex_graph, forex_connections_matrix


if __name__ == '__main__':
    #df = load_data("../data/preprocessed/merged.csv")
    #close_df = transform(df)

    g, connections_matrix = get_forex_graph()
    print(type(connections_matrix), np.shape(connections_matrix))
    embedding = np.ones((20, 8))

    for i in range(20):
        boolean_mask = connections_matrix[i][:]
        inputs = embedding[boolean_mask, :]

        print("i:", i)
        print(boolean_mask, np.shape(boolean_mask))
        print(np.shape(inputs))

