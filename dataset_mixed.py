import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from utils import normalizer

class ChessDataset(Dataset):
    def __init__(self, csv_file_path, sequence_model = True) -> None:
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        static_variables = ["ECO", "Event", "Termination", "TimeControl", "BlackTitle", "WhiteTitle", "Category", "Weekday", "num_move", "white_wins"]
        df = normalizer(df, ["seconds_remaining", "num_move", "black_mate_in", "white_mate_in"], True)
        df = normalizer(df, ["eval_normalized", "lag_eval"], False)
        df_static = df[static_variables]
        df = df.drop(columns = static_variables)
        df = pd.get_dummies(data=df, columns=['destination'])
        df_static = pd.get_dummies(data=df_static, columns=['ECO', "Event", "Termination", "TimeControl", "BlackTitle", "WhiteTitle", "Category", "Weekday"])
        df.apply(pd.to_numeric)
        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.df_static = df_static
        self.df.to_csv("test.csv")
        self.n_steps = 100
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]
        self.bin_width = 100
        self.sequence_model = sequence_model

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        df = self.df.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]
        df_static = self.df_static.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]
        labels_tensor = torch.tensor(
            (df[["WhiteElo", "BlackElo"]]/2000).values)  # (n_steps, 2)
        df = df.drop(columns=["WhiteElo", "BlackElo"])
        features_tensor_dynamic = torch.tensor(df.values)  # (n_steps, n_features)
        features_tensor_static = torch.tensor(df_static.values)

        return features_tensor_dynamic.double(), features_tensor_static[0,:].double(), labels_tensor[0, :].double()
