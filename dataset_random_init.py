import pandas as pd

import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, csv_file_path) -> None:
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(data=df, columns=['ECO', 'Event', 'Termination', 'TimeControl',
                            'BlackTitle', 'WhiteTitle', 'Category', 'Weekday', 'destination'])

        df.apply(pd.to_numeric)
        df["seconds_remaining"] = df["seconds_remaining"] / df["seconds_remaining"].abs().max()
        #df["black_mate_in"] = df["black_mate_in"] / df["black_mate_in"].abs().max()
        #df["white_mate_in"] = df["white_mate_in"] / df["white_mate_in"].abs().max()
        #df["eval_normalized"] = df["eval_normalized"] / df["eval_normalized"].abs().max()
        #df["lag_eval"] = df["lag_eval"] / df["lag_eval"].abs().max()
        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.df.to_csv("test.csv")
        self.n_steps = 100
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]

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
