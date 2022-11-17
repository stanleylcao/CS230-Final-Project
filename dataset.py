import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

def regression_to_softmax(df, colNames, bin_width, max_rating):
    ''''
    Input:
    df: with two columns for the Elo ratings we want to bin #shape: n_steps*n_examples, n_features
    colNames: 2-vector with names of White Elo and Black Elo column in that order
    bin_width: length of bin, will divide max_rating

    Output:
    features_tensor: pytorch tensor of features without labels  #shape: n_steps*n_examples, n_features - 2
    black_labels: pytorch tensor of black labels #shape: n_steps*n_examples, n_classes
    white_labels: pytorch tensor of white labels #shape: n_steps*n_examples, n_classes
    '''
    bins = [-np.inf]
    bins = np.append(bins, np.arange(0, max_rating, bin_width))
    bins = np.append(bins, np.inf)
    df["White_bins"] = pd.cut(df[colNames[0]], bins = bins)
    df["Black_bins"] = pd.cut(df[colNames[1]], bins = bins)
    df = pd.get_dummies(data=df, columns=['White_bins', "Black_bins"])
    white_labels = torch.tensor(df.filter(regex='^White_bins').values)
    black_labels = torch.tensor(df.filter(regex='^Black_bins').values)
    df = df.drop(columns=colNames)
    df = df[df.columns.drop(list(df.filter(regex='White_bins')))]
    df = df[df.columns.drop(list(df.filter(regex='Black_bins')))]
    features_tensor = torch.tensor(df.values)
    return features_tensor, white_labels, black_labels

def normalizer(df, colNames, standardize):
    '''
    accepts a pandas df, list of column names, and a parameter standardize.
    If standardize == True, it performs standardization on columns named (subtract mean and divide by sd)
    if standardize == False, performs min-max normalization
    '''
    if standardize == True:
        for colName in colNames:
            df[colName] = (df[colName]-df[colName].mean())/df[colName].std()
    else:
        for colName in colNames:
            df[colName] = df[colName] / df[colName].abs().max()
    return df



class ChessDataset(Dataset):
    def __init__(self, csv_file_path) -> None:
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(data=df, columns=['ECO', 'Event', 'Termination', 'TimeControl',
                            'BlackTitle', 'WhiteTitle', 'Category', 'Weekday', 'destination'])

        df.apply(pd.to_numeric)
        df = normalizer(df, ["seconds_remaining", "num_move", "black_mate_in", "white_mate_in"], False)
        df = normalizer(df, ["eval_normalized", "lag_eval"], True)
        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.df.to_csv("test.csv")
        self.n_steps = 100
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]
        self.bin_width = 100
        #self.df, self.white_labels, self.black_labels = regression_to_softmax(df, ["WhiteElo", "BlackElo"], self.bin_width, 3000)


    def __len__(self):

        return self.n_examples

    def __getitem__(self, idx):
        df = self.df.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]
        labels_tensor = torch.tensor(
            (df[["WhiteElo", "BlackElo"]]/2000).values)  # (n_steps, 2)
        #df = df[["WhiteElo", "BlackElo"]]/2000
        #df = df[["eval_normalized", "white_wins", "is_check"]]
        df = df.drop(columns=["WhiteElo", "BlackElo"])
        features_tensor = torch.tensor(df.values)  # (n_steps, n_features)
        return features_tensor.double(), labels_tensor[0, :].double()
