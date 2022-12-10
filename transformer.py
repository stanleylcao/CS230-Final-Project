import torch
from torch import nn, optim

import math
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from dataset import ChessDataset

from datetime import datetime



class TransformerClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 n_head,
                 dim_feedforward,
                 n_layers
                ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(input_size, dropout=dropout)

        # Only using Encoder of Transformer model
        encoder_layers = nn.TransformerEncoderLayer(input_size, n_head, dim_feedforward, dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.input_size = input_size
        self.decoder = nn.Linear(input_size, output_size)
        self.double()


    def forward(self, x):
        output = self.transformer_encoder(x)
        return self.decoder(output.mean(1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, maxlen = 5000):
        super(PositionalEncoding, self).__init__()

        # A tensor consists of all the possible positions (index) e.g 0, 1, 2, ... max length of input
        # Shape (pos) --> [max len, 1]
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_encoding = torch.zeros((maxlen, d_model))

        sin_den = 10000 ** (torch.arange(0, d_model, 2)/d_model) # sin for even item of position's dimension
        cos_den = 10000 ** (torch.arange(1, d_model, 2)/d_model) # cos for odd

        pos_encoding[:, 0::2] = torch.sin(pos / sin_den)
        pos_encoding[:, 1::2] = torch.cos(pos / cos_den)

        # Shape (pos_embedding) --> [max len, d_model]
        # Adding one more dimension in-between
        pos_encoding = pos_encoding.unsqueeze(-2)
        # Shape (pos_embedding) --> [max len, 1, d_model]

        self.dropout = nn.Dropout(dropout)

        # We want pos_encoding be saved and restored in the `state_dict`, but not trained by the optimizer
        # hence registering it!
        # Source & credits: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        # shape (token_embedding) --> [sentence len, batch size, d_model]

        # Concatenating embeddings with positional encodings
        # Note: As we made positional encoding with the size max length of sentence in our dataset
        #       hence here we are picking till the sentence length in a batch
        #       Another thing to notice is in the Transformer's paper they used FIXED positional encoding,
        #       there are methods where we can also learn them
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

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
    full_dataset = ChessDataset('for_pandas_big.csv')

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False)

    embedding_size = train_dataset[0][0].size()[1]

    print(embedding_size)

    model = TransformerClassifier(input_size = embedding_size, output_size = 2, dropout = 0.2, n_head = 6, n_layers = 6, dim_feedforward=256)
    
    #Warning: n_head must divide the number of features. If it doesn't we need to pad the number of features for the code to run

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
        test_loop(test_dataloader, model, loss_fn, writer=writer, epoch_num=e)
    writer.close()


if __name__ == '__main__':
    main()

