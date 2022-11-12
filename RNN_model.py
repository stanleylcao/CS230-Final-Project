import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class RNNModel(nn.Modele):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super(RNNModel, self).__init__()

        self.input_dim = input_dim  # embedding dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, nonlinearity='tanh', batch_first=True)

        self.fc = nn.Linear(in_features=self.hidden_dim,
                            out_features=self.output_dim)

    def forward(self, x):
        """
        B := batch size
        L := sequence length
        """
        out, h_n = self.rnn(x)
        ratings = self.fc(out)  # shape = (B, L, output_dim)

        # Just take final rating
        ratings = ratings[:, -1, :]  # shape = (B, output_dim)
        return ratings


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, num_examples_finished = loss.item(), batch * len(X)
            print(f'Loss = {loss} [{num_examples_finished}/{size}]')


def main():
    model = RNNModel()  # TODO: PASS IN CORRECT PARAMETERS
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        print(f'Beginning Epoch {e}')
        # TODO: need to define dataloader
        train_loop(train_dataloader, model, loss_fn, optimizer)


if __name__ == '__main__':
    main()
