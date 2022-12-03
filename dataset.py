import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import normalizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessDataset(Dataset):
    def __init__(self, csv_file_path, sequence_model) -> None:
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(data=df, columns=['ECO', 'Event', 'Termination', 'TimeControl',
                            'BlackTitle', 'WhiteTitle', 'Category', 'Weekday', 'destination'])

        df.apply(pd.to_numeric)
        df = normalizer(df, ["seconds_remaining", "num_move",
                        "black_mate_in", "white_mate_in"], False)
        df = normalizer(df, ["eval_normalized", "lag_eval"], True)
        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.df.to_csv("test.csv")
        self.n_steps = 100
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]
        self.bin_width = 100
        self.sequence_model = sequence_model
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
        if not self.sequence_model:
            features_tensor = torch.flatten(features_tensor)
        return features_tensor.double().to(device), labels_tensor[0, :].double().to(device)
