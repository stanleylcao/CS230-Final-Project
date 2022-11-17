import numpy as np
import pandas as pd
import torch


def regression_to_softmax(df, colNames, bin_width, max_rating):
    ''''
    Input:
    df: with two columns for the Elo ratings we want to bin #shape: n_steps*n_examples, n_features
    colNames: 2-vector with names of White Elo and Black Elo column in that order
    bin_width: length of bin, will divide max_rating
    max_rating: maximum possible rating that a bin will include

    Output:
    features_tensor: pytorch tensor of features without labels  #shape: n_steps*n_examples, n_features - 2
    black_labels: pytorch tensor of black labels #shape: n_steps*n_examples, n_classes
    white_labels: pytorch tensor of white labels #shape: n_steps*n_examples, n_classes
    '''
    pd.options.mode.chained_assignment = None

    bins = np.arange(0, max_rating + bin_width, bin_width).tolist()
    df["White_bins"] = pd.cut(df[colNames[0]], bins=bins)
    df["Black_bins"] = pd.cut(df[colNames[1]], bins=bins)
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
    