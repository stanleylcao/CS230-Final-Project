import pandas as pd

import torch
from torch.utils.data import Dataset

from utils import normalizer, regression_to_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessDataset(Dataset):
    def __init__(self, csv_file_path) -> None:
        bin_width = 100
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(data=df, columns=['ECO', 'Event', 'Termination', 'TimeControl',
                            'BlackTitle', 'WhiteTitle', 'Category', 'Weekday', 'destination'])

        df.apply(pd.to_numeric)
        df = normalizer(df, ["seconds_remaining", "num_move",
                        "black_mate_in", "white_mate_in"], False)
        df = normalizer(df, ["eval_normalized", "lag_eval"], True)
        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.n_steps = 100
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]
        self.bin_width = 100
        self.max_rating = 3300
        self.num_bins = (self.max_rating // self.bin_width)

        # df["WhiteElo"] = (df["WhiteElo"]//bin_width).astype(str)
        # df["BlackElo"] = (df["BlackElo"]//bin_width).astype(str)
        # df = pd.get_dummies(data=df, columns=['WhiteElo', "BlackElo"])
        # self.df = df.drop(columns=["Index", "move", "Index.1"])
        # self.df.to_csv("test.csv")
        # self.n_steps = 100
        # self.n_examples = df.shape[0] // self.n_steps
        # self.n_features = df.shape[1]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        df = self.df.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]
        features_tensor, white_labels, black_labels =\
            regression_to_softmax(
                df, ['WhiteElo', 'BlackElo'], self.bin_width, max_rating=self.max_rating)

        # Take last label
        white_labels = white_labels[-1]
        black_labels = black_labels[-1]
        assert white_labels.size(dim=0) == black_labels.size(dim=0)
        assert self.num_bins == white_labels.size(dim=0)

        # white_labels = torch.tensor(df.filter(regex='^WhiteElo').values)

        # black_labels = torch.tensor(df.filter(regex='^BlackElo').values)
        # # labels_tensor =
        # df = df[df.columns.drop(list(df.filter(regex='BlackElo')))]
        # df = df[df.columns.drop(list(df.filter(regex='WhiteElo')))]

        #df = df.drop(columns=["WhiteElo", "BlackElo"])
        # features_tensor = torch.tensor(df.values)  # (n_steps, n_features)

        # shape: n_steps, n_classes_white
        clean_white_labels = torch.unsqueeze(white_labels, dim=1)
        # shape: n_steps, n_classes_black
        clean_black_labels = torch.unsqueeze(black_labels, dim=1)

        # Shape = (num_bins, 2)
        labels_tensor = torch.cat(
            (clean_white_labels, clean_black_labels), dim=1)
        labels_tensor = torch.argmax(labels_tensor, dim=0)  # shape = (2,)
        return features_tensor.double().to(device), labels_tensor.long().to(device)
