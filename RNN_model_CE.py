import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from dataset_softmax import ChessDataset

from datetime import datetime


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super(RNNModel, self).__init__()

        self.input_dim = input_dim  # embedding dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # number of classes
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, nonlinearity='tanh', batch_first=True)
        self.inter_dim = self.hidden_dim // 2

        self.fc1 = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.inter_dim)
        self.fc2 = nn.Linear(in_features=self.inter_dim,
                             out_features=2 * self.output_dim)
        self.double()

    def forward(self, x):
        """
        B := batch size
        L := sequence length
        """
        # h_0 = torch.randn(
        #     size=(self.num_layers, x.size(dim=0), self.hidden_dim)).double()
        out, h_n = self.rnn(x)  # out.size() = [B, L, self.hidden_dim]
        out = self.fc1(out)
        logits = self.fc2(out)  # shape = (B, L, 2 * output_dim)

        # Just take final output
        logits = logits[:, -1, :]  # shape = (B, 2 * output_dim)
        # shape = (B, output_dim, 2)
        logits = torch.reshape(
            logits, (logits.size(dim=0), self.output_dim, 2))
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_examples_in_batch = X.size(dim=0)
        avg_train_loss_per_example = loss.item()
        if writer is not None:
            writer.add_scalar('Avg Train Loss Per Batch',
                              avg_train_loss_per_example, epoch_num)
        elif batch % 100 == 0:
            loss, num_examples_finished = loss.item(), batch * len(X)
            print(f'Loss = {loss} [{num_examples_finished}/{size}]')
            # print(f'BATCH {batch}')
            # print("INPUT")
            # print(X)
            # predictions = torch.argmax(pred, dim=1)
            # print(f"PREDICTIONS = {predictions}")
            # labels = torch.argmax(y, dim=1)
            # print(f"LABELS = {y}")


def test_loop(dataloader, model, loss_fn, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            print('PREDICTIONS')
            print(torch.argmax(pred, dim=1))
            print('LABELS')
            print(y)
    avg_test_loss = test_loss / num_batches
    if writer is not None:
        writer.add_scalar('Avg Test Loss Per Example',
                          avg_test_loss, epoch_num)
    else:
        print(f'Avg Test Loss = {avg_test_loss}')


learning_rate = 1e-3
epochs = int(1e6)
batch_size = 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    full_dataset = ChessDataset('for_pandas.csv')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    embedding_size = train_dataset[0][0].size()[1]
    num_bins = full_dataset.num_bins

    model = RNNModel(input_dim=embedding_size, hidden_dim=1024,
                     output_dim=num_bins, num_layers=2)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    now = datetime.now()
    experiment_time_str = now.strftime("%d-%m-%Y-%H:%M:%S")

    # writer = SummaryWriter(
    #     log_dir=f'runs/small_dataset_normalized_CE_model/{experiment_time_str}')
    writer = None

    for e in range(epochs):
        print(f'Beginning Epoch {e}')
        train_loop(train_dataloader, model, loss_fn,
                   optimizer, writer=writer, epoch_num=e)
        test_loop(test_dataloader, model, loss_fn, writer=writer, epoch_num=e)

    writer.close()


if __name__ == '__main__':
    main()
