import pandas as pd

import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, csv_file_path) -> None:
        bin_width = 100
        df = pd.read_csv(csv_file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(data=df, columns=['ECO', 'Event', 'Termination', 'TimeControl',
                            'BlackTitle', 'WhiteTitle', 'Category', 'Weekday', 'destination'])

        df.apply(pd.to_numeric)
        df["seconds_remaining"] = df["seconds_remaining"] / \
            df["seconds_remaining"].abs().max()
        #df["black_mate_in"] = df["black_mate_in"] / df["black_mate_in"].abs().max()
        #df["white_mate_in"] = df["white_mate_in"] / df["white_mate_in"].abs().max()
        #df["eval_normalized"] = df["eval_normalized"] / df["eval_normalized"].abs().max()
        #df["lag_eval"] = df["lag_eval"] / df["lag_eval"].abs().max()
        df["WhiteElo"] = (df["WhiteElo"]//bin_width).astype(str)
        df["BlackElo"] = (df["BlackElo"]//bin_width).astype(str)
        df = pd.get_dummies(data=df, columns=['WhiteElo', "BlackElo"])
        self.df = df.drop(columns=["Index", "move", "Index.1"])
        self.df.to_csv("test.csv")
        self.n_steps = 50
        self.n_examples = df.shape[0] // self.n_steps
        self.n_features = df.shape[1]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        df = self.df.iloc[idx * self.n_steps: (idx + 1) * self.n_steps]
        # shape: n_steps, n_classes_white
        white_labels = torch.tensor(df.filter(regex='^WhiteElo').values)
        # shape: n_steps, n_classes_black
        black_labels = torch.tensor(df.filter(regex='^BlackElo').values)
        # labels_tensor =
        df = df[df.columns.drop(list(df.filter(regex='BlackElo')))]
        df = df[df.columns.drop(list(df.filter(regex='WhiteElo')))]

        #df = df.drop(columns=["WhiteElo", "BlackElo"])
        features_tensor = torch.tensor(df.values)  # (n_steps, n_features)

        clean_white_labels = torch.unsqueeze(white_labels[0], dim=1)
        clean_black_labels = torch.unsqueeze(black_labels[0], dim=1)
        # Shape = (num_bins, 2)
        # labels_tensor = torch.cat(
        #     (clean_white_labels, clean_black_labels), dim=1)
        labels_tensor = torch.argmax(white_labels[0])
        return features_tensor.double(), labels_tensor.long()
