
from datetime import datetime

import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import BucketIterator

from data_helpers.data_gen_utils import gen_df 
from data_helpers.dataframe_dataset import DataFrameDataset


import numpy as np
import random



# set random seeds for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

# check if cuda is enabled
USE_GPU=1
# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')


def tokenize(text):
    # simple tokenizer
    words = text.lower().split()
    return words


def accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # get max values along rows
    _, indices = preds.max(dim=1)
    # values, indices = torch.max(tensor, 0)

    correct = (indices == y).float()  # convert into float for division
    acc = correct.sum()/len(correct)
    return acc




# gen the trainning data
min_seq_len = 100
max_seq_len = 300

# numer of tokenes in vocab to generate, max 10
# it is equal the number of classes
seq_tokens = 10

n_train = 1000
n_valid = 200

train_df = gen_df(n=n_train, min_seq_len=min_seq_len,
                      max_seq_len=max_seq_len, seq_tokens=seq_tokens)
valid_df = gen_df(n=n_valid, min_seq_len=min_seq_len,
                      max_seq_len=max_seq_len, seq_tokens=seq_tokens)


print(train_df)
print(valid_df)

TEXT = data.Field(sequential=True, lower=True, tokenize=tokenize,fix_length=None)
LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

fields = {"text": TEXT, "label": LABEL}


train_ds = DataFrameDataset(train_df, fields)
valid_ds = DataFrameDataset(valid_df, fields)

# numericalize the words
TEXT.build_vocab(train_ds, min_freq=1)
print(TEXT.vocab.freqs.most_common(20))

vocab = TEXT.vocab
vocab_size = len(vocab)

batch_size = 4
train_iter = BucketIterator(
    train_ds, 
    batch_size=batch_size, 
    sort_key=lambda x: len(x.text), 
    sort_within_batch=True, 
    device=device)

valid_iter = BucketIterator(
    valid_ds, 
    batch_size=batch_size, 
    sort_key=lambda x: len(x.text), 
    sort_within_batch=True,
    device=device)

#hidden size
n_hid=200
# embed size
n_embed=10
# number of layers
n_layers=1




class SeqLSTM(nn.Module):
    """
    LSTM example for long sequence
    """

    def __init__(self, vocab_size, output_size, embed_size, hidden_size, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)

        #after the embedding we can add dropout
        # self.drop = nn.Dropout(0.1)

        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=False)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        # Embed word ids to vectors
        len_seq, bs = seq.shape
        w_embed = self.embed(seq)
        # w_embed = self.drop(w_embed)

        # https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
        output, (hidden, cell) = self.lstm(w_embed)

        # use dropout
        # hidden = self.dropout(hidden[-1,:,:])

        # hidden has size [1,batch,hid dim]
        # this does .squeeze(0) now hidden has size [batch, hid dim]
        last_output = output[-1, :, :]
        # last_output = self.drop(last_output)

        out = self.linear(last_output)

        return out


model = SeqLSTM(vocab_size=vocab_size, output_size=seq_tokens,
                    embed_size=n_embed, hidden_size=n_hid)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



print("-"*80)
print(f'n_train={n_train}, n_valid={n_valid}')
print(f'min_seq_len={min_seq_len}, max_seq_len={max_seq_len}')

print(f'model params')
print(f'vocab={vocab_size}, output={seq_tokens}')
print(f'n_layers={n_layers}, n_hid={n_hid} embed={n_embed}')

epoch_loss = 0
epoch_acc = 0
epoch = 20

for e in range(epoch):

    start_time = datetime.now()
    # train loop
    model.train()
    for batch_idx, batch in enumerate(train_iter):

        # get the inputs
        inputs, labels = batch
        # move data to device (GPU if enabled, else CPU do nothing)
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        #optimizer.zero_grad()

        # get model output
        predictions = model(inputs)

        # prediction are [batch, out_dim]
        # batch.label are [1,batch] <- should be mapped to  output vector
        loss = criterion(predictions, labels)
        epoch_loss += loss.item()

        # do backward and optimization step
        loss.backward()
        optimizer.step()

    # mean epoch loss
    epoch_loss = epoch_loss / len(train_iter)

    time_elapsed = datetime.now() - start_time

    # evaluation loop
    model.eval()
    for batch_idx, batch in enumerate(valid_iter):

        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # get model output
        predictions = model(inputs)

        # compute batch validation accuracy
        acc = accuracy(predictions, labels)

        epoch_acc += acc

    epoch_acc = epoch_acc/len(valid_iter)

    # show summary

    print(
        f'Epoch {e}/{epoch} loss={epoch_loss} acc={epoch_acc} time={time_elapsed}')
    epoch_loss = 0
    epoch_acc = 0
