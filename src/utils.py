"""shared helpers used by train.py and inference.py"""

import random
import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import set_seed

OPTS = list("ABCDE")
# convert option letters to ids and back
LETTER2ID = {o: i for i, o in enumerate(OPTS)}
ID2LETTER = {i: o for o, i in LETTER2ID.items()}


def format_mcq(row):
    # combine question and all options into one text
    return (f"{row['prompt']} "
            f"A) {row['A']} B) {row['B']} C) {row['C']} D) {row['D']} E) {row['E']}")


def clean_text(t):
    # lowercase and remove special characters
    return re.sub(r"[^a-z0-9\s]", " ", str(t).lower())


# average precision for one sample
def apk(actual, predicted, k=3):
    for i, p in enumerate(predicted[:k]):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


# mean average precision over all samples
def mapk(actuals, predictions, k=3):
    return sum(apk(a, p, k) for a, p in zip(actuals, predictions)) / len(actuals)


def top1_accuracy(actuals, predictions):
    return round(accuracy_score(actuals, [p[0] for p in predictions]), 4)


def top1_f1(actuals, predictions):
    return round(f1_score(actuals, [p[0] for p in predictions], average='macro'), 4)


# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_with_retry(fn, *args, retries=3, **kwargs):
    # retry loading if temporary error happens
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  retry {i+1}/{retries}: {e}")


SEED = 42


def seed_everything(seed):
    # same seed gives same results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def build_transformer_inputs(train):
    train_inputs = [format_mcq(row) for _, row in train.iterrows()]
    train_labels = [LETTER2ID[a] for a in train['answer']]
    return train_inputs, train_labels


# split text into words
def toks(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def build_vocab(train):
    # special tokens
    vocab = {"<pad>": 0, "<unk>": 1}

    # build vocabulary from training data
    for _, r in train.iterrows():
        for w in toks(format_mcq(r)):
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


MAXL = 64  # maximum sequence length


def enc(t, vocab):
    # convert words to ids
    ids = [vocab.get(w, 1) for w in toks(t)[:MAXL]]
    # pad shorter sequences with zeros
    return ids + [0] * (MAXL - len(ids))


def build_sequence_data(train, vocab):
    cnn_X = np.array([enc(format_mcq(r), vocab) for _, r in train.iterrows()])
    cnn_y = np.array([LETTER2ID[a] for a in train['answer']])

    # keep class ratio same in train and validation
    cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva = train_test_split(
        cnn_X,
        cnn_y,
        test_size=0.1,
        stratify=cnn_y,
        random_state=42
    )

    cnn_ytr_l = [ID2LETTER[i] for i in cnn_ytr]
    cnn_yva_l = [ID2LETTER[i] for i in cnn_yva]

    return cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva, cnn_ytr_l, cnn_yva_l
