import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from dataset import ChessDataset

from datetime import datetime


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size) -> None:
        super(LSTM_Model, self).__init__()

        self.input_dim = input_dim  # embedding dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, dropout = 0.2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim,
                            out_features=self.hidden_dim//2)
        self.fc2 = nn.Linear(in_features=self.hidden_dim//2,
                            out_features=self.output_dim)
        self.double()

    def forward(self, x):
        """
        B := batch size
        L := sequence length
        """
        #h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim).double()
        #c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim).double()

        out, hn = self.lstm(x) #out.shape = (B, L, hidden_dim)
        out, input_size = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
        last_seq_idxs = torch.LongTensor([x - 1 for x in input_size])
        last_seq_items = out[range(out.shape[0]), last_seq_idxs, :]
        ratings = torch.relu(self.fc1(last_seq_items))
        ratings = self.fc2(ratings)
        return ratings


def train_loop(dataloader, model, loss_fn, optimizer, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    for batch, (X, y, sequence_len) in enumerate(dataloader):
        optimizer.zero_grad()
        pack = torch.nn.utils.rnn.pack_padded_sequence(X, sequence_len, batch_first=True, enforce_sorted = False)
        pred = model(pack)
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
        for X, y, sequence_len in dataloader:
            pack = torch.nn.utils.rnn.pack_padded_sequence(X, sequence_len, batch_first=True, enforce_sorted = False)
            pred = model(pack)
            test_loss += loss_fn(pred, y).item()
            #print(pred*2000)
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
    full_dataset = ChessDataset('for_pandas.csv', pack = True)

    train_size = (int(0.9 * len(full_dataset))*batch_size)//batch_size
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False)

    embedding_size = train_dataset[0][0].size()[1]

    model = LSTM_Model(input_dim=embedding_size, hidden_dim=256,
                     output_dim=2, num_layers=3,  batch_size = batch_size)

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
