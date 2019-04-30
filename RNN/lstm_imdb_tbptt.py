
from datetime import datetime
import torch
from torch import nn, optim
from torchtext import data
#from torchtext.data import BucketIterator
from torchtext import datasets


from progress.bar import Bar

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

    # apply softmax
    preds = torch.nn.functional.softmax(preds, dim=1)

    # get max values along rows
    _, indices = preds.max(dim=1)
    # values, indices = torch.max(tensor, 0)

    correct = (indices == y).float()  # convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def split_batch(batch, bptt):
    """
    Split torch.tensor batch by bptt steps, 
    Split seqence dim by bptt
    """
    batch_splits = batch.split(bptt,dim=0)
    return batch_splits




class LongSeqTbttRnn(nn.Module):
    """
    RNN example for long sequence
    with TBPTT truncated backpropagation throu time
    """

    def __init__(self, input_dim, output_dim, embed_size, hidden_size, num_layers=1, dropout=0.1,vectors=None ):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embed = nn.Embedding(input_dim, embed_size)
        # if we want to copy embedding vectors
        if vectors:
            self.embed.weight.data.copy_(vectors)

        #after the embedding we can add dropout
        self.drop = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=False)
        
        # we need this for storing last rnn state
        self.rnn_state = None

        self.linear = nn.Linear(hidden_size, output_dim)


    def repackage_rnn_state(self):

        self.rnn_state = self._detach_rnn_state(self.rnn_state)

    def _detach_rnn_state(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history.
        based on repackage_hidden function from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._detach_rnn_state(v) for v in h)

    def forward(self, seq):
        # Embed word ids to vectors
        len_seq, bs = seq.shape
        w_embed = self.embed(seq)
        w_embed = self.drop(w_embed)
    
        output, self.rnn_state = self.rnn(w_embed, self.rnn_state)
        
        # this does .squeeze(0) now hidden has size [batch, hid dim]
        last_output = output[-1, :, :]
        last_output = self.drop(last_output)

        out = self.linear(last_output)
        return out



# set up fields
TEXT = data.Field(lower=True, include_lengths=True)
LABEL = data.LabelField()

# make splits for data
train_ds, valid_ds = datasets.IMDB.splits(TEXT, LABEL)


# take a portion of datasets, for testing :)
# train_ds, _ = train_ds.split(0.5)
# valid_ds, _ = valid_ds.split(0.5)

print(f'train={len(train_ds)} valid={len(valid_ds)}')

# build the vocabulary
TEXT.build_vocab(train_ds,min_freq=10, max_size=10000 ) #, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train_ds)

print(TEXT.vocab.freqs.most_common(20))

vocab = TEXT.vocab

vocab_size = len(vocab)
print(f'vocab_size={vocab_size}')
print(list(vocab.stoi.keys())[0:20])
print(vocab.itos[0:20])
print(vocab.vectors)

print(LABEL.vocab.stoi)



#hidden size
n_hid=512
# embed size
n_embed=300
# number of layers
n_layers=1
batch_size = 256

#split batch text in to bptt chunks
bptt = 50

input_dim = vocab_size
output_dim = len(LABEL.vocab)

print("-"*80)
print(f'model params')
print(f'input_dim={input_dim}, output={output_dim}')
print(f'n_layers={n_layers}, n_hid={n_hid} embed={n_embed}')
print(f'batch={batch_size}, bptt={bptt}')

train_iter = data.BucketIterator(
    train_ds, batch_size=batch_size, sort_key=lambda x: len(x.text), sort_within_batch=True, device=device)

valid_iter = data.BucketIterator(
    valid_ds, batch_size=batch_size, sort_key=lambda x: len(x.text), sort_within_batch=True, device=device)




model = LongSeqTbttRnn(input_dim=input_dim, output_dim=output_dim,
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
        # before each bptt zero state
        # model.zero_rnn_state(batch_size=batch_size)
        model.rnn_state = None

        # move data to device (GPU if enabled, else CPU do nothing)
        batch_text = batch.text[0].to(device) # include lengths at [1]
        batch_label = batch.label.to(device)

        bptt_loss= 0
        bptt_batch_chunks = split_batch(batch_text, bptt)
        # second TBPTT loop, split batch and learn in chunks of batch
        for text_chunk in bptt_batch_chunks:

            model.zero_grad()
            predictions = model(text_chunk)
            
            # for each bptt size we have the same batch_labels
            loss = criterion(predictions, batch_label)
            bptt_loss += loss.item()

            # do back propagation for bptt steps in time
            loss.backward()
            optimizer.step()
            # after doing back prob, detach rnn state in order to implement TBPTT (truncated backpropagation through time startegy)
            # now rnn_state was detached and chain of gradeints was broken
            model.repackage_rnn_state()


        bar.next()
        epoch_loss += bptt_loss
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
            
            #reset to zero model state
            model.rnn_state = None
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
