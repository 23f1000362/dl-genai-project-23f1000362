"""loads the trained models and writes submission.csv"""

import pickle

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import BiLSTMAttn, TextCNN, build_row_features, fit_tfidf
from utils import ID2LETTER, device, enc, format_mcq

DATA_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge"
MODELS_DIR = "../models"


def softmax(v):
    v = np.asarray(v, dtype=float)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def transformer_logits(model, tok, row):
    inputs = tok(format_mcq(row), return_tensors="pt",
                 truncation=True, max_length=256).to(device)
    with torch.no_grad():
        return model(**inputs).logits[0].cpu().numpy()


def probs_cnn_like(model, row, vocab):
    x = torch.tensor(enc(format_mcq(row), vocab)).unsqueeze(0).to(device)
    with torch.no_grad():
        return softmax(model(x).cpu().numpy()[0])


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

# custom model vocabulary and tf-idf classifier
vocab = pickle.load(open(f"{MODELS_DIR}/vocab.pkl", "rb"))
tfidf_clf = pickle.load(open(f"{MODELS_DIR}/tfidf_clf.pkl", "rb"))
word_tfidf, char_tfidf = fit_tfidf(train, test)

# roberta with its lora adapter
cls_tok = AutoTokenizer.from_pretrained(f"{MODELS_DIR}/roberta")
cls_base = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
lora_model = PeftModel.from_pretrained(cls_base, f"{MODELS_DIR}/roberta").to(device)
lora_model.eval()

# deberta with its lora adapter
deberta_tok = AutoTokenizer.from_pretrained(f"{MODELS_DIR}/deberta")
deberta_base = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=5)
lora_deberta = PeftModel.from_pretrained(deberta_base, f"{MODELS_DIR}/deberta").to(device)
lora_deberta.eval()

cnn_model = TextCNN(len(vocab)).to(device)
cnn_model.load_state_dict(torch.load(f"{MODELS_DIR}/textcnn.pt", map_location=device))
cnn_model.eval()

rnn_model = BiLSTMAttn(len(vocab)).to(device)
rnn_model.load_state_dict(torch.load(f"{MODELS_DIR}/bilstm.pt", map_location=device))
rnn_model.eval()

# weights: roberta dominates (strongest single)
W = {"roberta": 0.50, "deberta": 0.15, "cnn": 0.12, "bilstm": 0.13, "tfidf": 0.10}
print("ensemble models:", list(W.keys()), "| weights:", W)

test_preds = []
for _, row in test.iterrows():
    p = np.zeros(5)
    p += W["roberta"] * softmax(transformer_logits(lora_model, cls_tok, row))
    p += W["deberta"] * softmax(transformer_logits(lora_deberta, deberta_tok, row))
    p += W["cnn"] * probs_cnn_like(cnn_model, row, vocab)
    p += W["bilstm"] * probs_cnn_like(rnn_model, row, vocab)
    p += W["tfidf"] * tfidf_clf.predict_proba(
        np.array(build_row_features(row, word_tfidf, char_tfidf)).reshape(1, -1))[0]
    order = np.argsort(-p)[:3]
    test_preds.append(" ".join(ID2LETTER[i] for i in order))

submission = pd.DataFrame({'ID': test['id'].values, 'Prediction': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nsubmission head:")
print(submission.head(10))
print("shape:", submission.shape)
assert submission['Prediction'].str.split().apply(len).eq(3).all()
assert list(submission.columns) == ['ID', 'Prediction']
print("OK: 3 letters per row, columns correct.")
