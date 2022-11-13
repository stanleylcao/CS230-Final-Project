import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import ChessDataset


class RNNModel(nn.Module):
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
        self.double()

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

        if batch % 1 == 0:
            loss, num_examples_finished = loss.item(), batch * len(X)
            print(f'Loss = {loss} [{num_examples_finished}/{size}]')


def test_loop(dataloader, model, loss_fn, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    avg_test_loss = test_loss // num_batches
    print(f'Avg Test Loss = {avg_test_loss}')
    if writer is not None:
        writer.add_scalar('Avg Test Loss', avg_test_loss, epoch_num)


learning_rate = 1e-3
epochs = int(1e6)
batch_size = 64


def main():
    full_dataset = ChessDataset('for_pandas.csv')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    embedding_size = train_dataset[0][0].size()[1]

    model = RNNModel(input_dim=embedding_size, hidden_dim=100,
                     output_dim=2, num_layers=2)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir='runs/full_dataset')

    for e in range(epochs):
        print(f'Beginning Epoch {e}')
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn, writer=writer, epoch_num=e)

    writer.close()


if __name__ == '__main__':
    main()
