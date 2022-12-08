import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import normalizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessDataset(Dataset):
    def __init__(self, csv_file_path, sequence_model = True, pack = False) -> None:
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(data=df, columns=['ECO', 'Event', 'Termination', 'TimeControl',
                            'BlackTitle', 'WhiteTitle', 'Category', 'Weekday', 'destination'])

        df.apply(pd.to_numeric)
        df = normalizer(df, ["seconds_remaining", "black_mate_in", "white_mate_in"], False)
        df = normalizer(df, ["eval_normalized", "lag_eval"], True)
        if not pack:
            df = normalizer(df, ["num_move"], False)

        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.n_steps = 150
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]
        self.bin_width = 100
        self.sequence_model = sequence_model
        self.pack = pack
        self.df.to_csv("test.csv")


    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        df = self.df.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]
        labels_tensor = torch.tensor(
            (df[["WhiteElo", "BlackElo"]]/2000).values)  # (n_steps, 2)
        df = df.drop(columns=["WhiteElo", "BlackElo"])
        if self.pack:
            sequence_len = min(self.df["num_move"][idx * self.n_steps] + 1, self.n_steps)
            df = df.drop(columns=["num_move"])
            features_tensor = torch.tensor(df.values)  # (n_steps, n_features)
            return features_tensor.double(), labels_tensor[0, :].double(), sequence_len
        features_tensor = torch.tensor(df.values)  # (n_steps, n_features)
        if not self.sequence_model:
            features_tensor = torch.flatten(features_tensor)
        return features_tensor.double(), labels_tensor[0, :].double()
