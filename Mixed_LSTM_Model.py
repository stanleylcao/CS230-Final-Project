import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from dataset_mixed import ChessDataset

from datetime import datetime


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, static_features, output_dim, num_layers) -> None:
        super(LSTM_Model, self).__init__()

        self.input_dim = input_dim  # embedding dim
        self.hidden_dim = hidden_dim
        self.static_features = static_features
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, dropout = 0.2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim + self.static_features,
                            out_features=self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=self.hidden_dim,
                            out_features=self.output_dim)
        self.double()

    def forward(self, x_dynamic, x_static):
        """
        B := batch size
        L := sequence length
        """
        h0 = torch.randn(self.num_layers, x_dynamic.size(0), self.hidden_dim).double()
        c0 = torch.randn(self.num_layers, x_dynamic.size(0), self.hidden_dim).double()

        lstm_out, hn = self.lstm(x_dynamic, (h0, c0)) #lstm_out.shape = (B, L, hidden_dim)
        lstm_out = lstm_out[:, -1, :] #lstm_out.shape = (B, hidden_dim)
        x = torch.cat((lstm_out, x_static), dim = 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        ratings = self.fc2(x)  # shape = (B,output_dim)

        return ratings


def train_loop(dataloader, model, loss_fn, optimizer, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    for batch, (X_dynamic, X_static, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X_dynamic, X_static)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        num_examples_in_batch = X_dynamic.size(dim=0)
        avg_train_loss_per_example = loss.item()
        if writer is not None:
            writer.add_scalar('Avg Training Loss',
                              avg_train_loss_per_example, epoch_num)
        elif batch % 1 == 0:
            loss, num_examples_finished = loss.item(), batch * len(X_dynamic)
            #print(f'Loss = {loss} [{num_examples_finished}/{size}]')


def test_loop(dataloader, model, loss_fn, writer=None, epoch_num=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_dynamic, X_static, y in dataloader:
            pred = model(X_dynamic, X_static)
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
batch_size = 64


def main():
    full_dataset = ChessDataset('for_pandas_50k.csv')

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False)

    embedding_size = train_dataset[0][0].size()[1]
    static_feature_size = train_dataset[0][1].size()[0]


    model = LSTM_Model(input_dim=embedding_size, hidden_dim=256, static_features = static_feature_size,
                     output_dim=1, num_layers=3)

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
