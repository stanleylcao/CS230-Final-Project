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
        self.df = df.drop(columns=["Index", "move", "Index.1"])

        self.n_steps = 100
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        df = self.df.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]

        labels_tensor = torch.tensor(
            df[["WhiteElo", "BlackElo"]].values)  # (n_steps, 2)
        df = df.drop(columns=["WhiteElo", "BlackElo"])
        features_tensor = torch.tensor(df.values)  # (n_steps, n_features)
        # features_tensor = labels_tensor
        return features_tensor.double(), labels_tensor[0, :].double()
