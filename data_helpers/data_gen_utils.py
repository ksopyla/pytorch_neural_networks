
import random
import collections
import pandas as pd

def gen_sequence(seq_len, vocab_max_size=10, prob=None):
    """
    Generate sequence for chars from vocabulary of first 10 chars

    Attributes:
    -----------
    seq_len: int
        sequence lenght
    vocab_max_size: int
        how many different chars our sequence will have, max=10, for simplicity 10 first alphabet letters is considered only.
    """
    # 10 letters
    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    vocab = vocab[0:vocab_max_size]

    vocab_len = len(vocab)

    # uniform distribution, you can change it if you want different probability distribution
    if prob is None:
        prob = ([1/vocab_len]*vocab_len)

    seq = random.choices(vocab, weights=prob, k=seq_len)
    counts = collections.Counter(seq)

    # we choose the most common letter in seq as label
    seq_label = counts.most_common(1)[0]
    label_txt = seq_label[0]
    label = vocab.index(label_txt)  # start from 0

    # join array of chars into string
    return ' '.join(seq), label, label_txt


def gen_df(n=2, min_seq_len=10, max_seq_len=10, seq_tokens=10):
    """
    Generates pandas dataframe with `n` number of examples of seq len from [min,max]

    Attributes:
    ------------
    n : int
        number of examples (sequences)
    min_seq_len: int
        minimal sequence length
    max_seq_len: int
        maximal sequence length

    seq_tokens: int
        number of different tokens in sequence (chars)

    """
    data = {
        'text': [],
        'label': [],
        'label_txt': [],
    }

    for i in range(n):
        # random seq_len
        seq_len = random.randint(min_seq_len, max_seq_len)


        prob = ([1/seq_tokens]*seq_tokens)
        # it-h token will have increased probalility
        prob[i% seq_tokens]*=2

        seq, label, label_txt = gen_sequence(seq_len, vocab_max_size=seq_tokens, prob=prob)

        data['text'].append(seq)
        data['label'].append(label)
        data['label_txt'].append(label_txt)

    df = pd.DataFrame(data)

    return df
