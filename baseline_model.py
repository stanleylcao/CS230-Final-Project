import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from dataset import ChessDataset

from datetime import datetime


class baseline_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(baseline_model, self).__init__()

        self.input_dim = input_dim  # embedding dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(in_features=self.input_dim,
                            out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim,
                             out_features=self.hidden_dim//2)
        self.output = nn.Linear(in_features=hidden_dim//2,
                            out_features=self.output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.double()

    def forward(self, x):
        """
        B := batch size
        L := sequence length
        """

        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        ratings = self.output(z)  # shape = (B, L, output_dim)
        return ratings


def train_loop(dataloader, model, loss_fn, optimizer, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        num_examples_in_batch = X.size(dim=0)
        avg_train_loss_per_example = loss.item()
        if writer is not None:
            writer.add_scalar('Avg Training Loss',
                              avg_train_loss_per_example, epoch_num)
        elif batch % 1 == 0:
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
    avg_test_loss_per_example = test_loss/num_batches
    if writer is not None:
        writer.add_scalar('Avg Test Loss',
                          avg_test_loss_per_example, epoch_num)
        print(f'Avg Test Loss = {avg_test_loss_per_example}')

    else:
        print(f'Avg Test Loss = {avg_test_loss_per_example}')


learning_rate = 0.0001
epochs = int(100)
batch_size = 32


def main():
    full_dataset = ChessDataset('for_pandas.csv', sequence_model = False)

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False)

    embedding_size = train_dataset[0][0].size()[0]

    model = baseline_model(input_dim=embedding_size, hidden_dim=256,
                     output_dim=2)

    loss_fn = nn.L1Loss(reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    now = datetime.now()
    experiment_time_str = now.strftime("%d-%m-%Y-%H:%M:%S")
    writer = SummaryWriter(
        log_dir=f'runs/small_dataset_normalized/{experiment_time_str}')

    for e in range(epochs):
        print(f'Beginning Epoch {e}')
        train_loop(train_dataloader, model, loss_fn,
                   optimizer, writer=None, epoch_num=e)
        test_loop(test_dataloader, model, loss_fn, writer=None, epoch_num=e)
    writer.close()


if __name__ == '__main__':
    main()
