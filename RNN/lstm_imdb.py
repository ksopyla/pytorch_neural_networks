import torch
from torch import nn, optim
from torchtext import data, datasets
import numpy as np
import random

from datetime import datetime
from progress.bar import Bar

# set random seeds for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

USE_GPU=0
# check if cuda device is enabled
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')

def tokenize(text):
    """Simple tokenizer, change for something more sophisticated
    """
    return text.lower().split()


def accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # apply softmax
    preds = torch.nn.functional.softmax(preds, dim=1)
    # get max values along rows
    _, indices = preds.max(dim=1)
    # values, indices = torch.max(tensor, 0)
    correct = (indices == y).float()  # convert into float for division
    acc = correct.sum()/len(correct)
    return acc

##### Read the data

# set up fields
TEXT = data.Field(lower=True,
                  include_lengths=True,
                  tokenize=tokenize)
LABEL = data.LabelField()

# make splits for data
train_ds, valid_ds = datasets.IMDB.splits(TEXT, LABEL)
# take a portion of datasets, for testing :)
# train_ds, _ = train_ds.split(0.5)
# valid_ds, _ = valid_ds.split(0.5)
print(f'train={len(train_ds)} valid={len(valid_ds)}')

# build the vocabulary
TEXT.build_vocab(train_ds,
                 min_freq=10,
                 max_size=10000 ) #, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train_ds)

print(TEXT.vocab.freqs.most_common(10))
print(TEXT.vocab.freqs.most_common()[:-11:-1])
vocab = TEXT.vocab

vocab_size = len(vocab)
print(f'vocab_size={vocab_size}')
print(list(vocab.stoi.keys())[0:10])

print(LABEL.vocab.stoi.keys())

#hidden size
n_hid=256
# embed size
n_embed=100
# number of layers
n_layers=1
batch_size = 8

input_dim = vocab_size # =10002
output_dim = len(LABEL.vocab) # =2

train_iter = data.BucketIterator(
    train_ds,
    batch_size=batch_size,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True, device=device)

valid_iter = data.BucketIterator(
    valid_ds, batch_size=batch_size,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True, device=device)


print("-"*80)
print(f'model params')
print(f'input_dim={input_dim}, output={output_dim}')
print(f'n_layers={n_layers}, n_hid={n_hid} embed={n_embed}')
print(f'batch={batch_size}')

class SeqRNN(nn.Module):

    def __init__(self, input_dim,
                 output_dim, embed_size,
                 hidden_size, num_layers=1,
                 dropout=0.1,vectors=None ):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_dim, embed_size)
        # if we want to copy embedding vectors
        if vectors:
            self.embed.weight.data.copy_(vectors)

        #after the embedding we can add dropout
        self.drop = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=False)
        #output linear layer
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, seq):
        # Embed word ids to vectors
        len_seq, bs = seq.shape
        w_embed = self.embed(seq)
        w_embed = self.drop(w_embed)
    
        output, _ = self.rnn(w_embed)
        
        # this does .squeeze(0) now hidden has size [batch, hid dim]
        last_output = output[-1, :, :]
        # apply dropout
        last_output = self.drop(last_output)

        out = self.linear(last_output)
        return out

model = SeqRNN(input_dim=input_dim,
               output_dim=output_dim,
               embed_size=n_embed, hidden_size=n_hid)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epoch_loss = 0
epoch_acc = 0
epoch = 10

for e in range(epoch):

    start_time = datetime.now()
    # train loop
    model.train()
    # progress
    bar = Bar(f'Training Epoch {e}/{epoch}', max=len(train_iter))
    for batch_idx, batch in enumerate(train_iter):

        model.zero_grad()
        # move data to device (GPU if enabled, else CPU do nothing)
        batch_text = batch.text[0].to(device) # include lengths at [1]
        batch_label = batch.label.to(device)
        
        predictions = model(batch_text)
        # compute loss
        loss = criterion(predictions, batch_label)
        epoch_loss += loss.item()

        # do back propagation for bptt steps in time
        loss.backward()
        optimizer.step()

        bar.next()

    bar.finish()
    # mean epoch loss
    epoch_loss = epoch_loss / len(train_iter)

    time_elapsed = datetime.now() - start_time

    # progress
    bar = Bar(f'Validation Epoch {e}/{epoch}', max=len(valid_iter))
    # evaluation loop
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_iter):
            # print(f'batch_idx={batch_idx}')
            batch_text = batch.text[0] #batch.text is a tuple
            batch_label = batch.label
            # get model output
            predictions = model(batch_text)
            # compute batch validation accuracy
            acc = accuracy(predictions, batch_label)

            epoch_acc += acc
            bar.next()

    epoch_acc = epoch_acc/len(valid_iter)
    bar.finish()

    # show summary
    print(
        f'Epoch {e}/{epoch} loss={epoch_loss} acc={epoch_acc} time={time_elapsed}')
    epoch_loss = 0
    epoch_acc = 0
