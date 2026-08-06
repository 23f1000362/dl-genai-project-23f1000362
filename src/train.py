"""training code for all five models"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from utils import (ID2LETTER, LETTER2ID, OPTS, SEED, build_sequence_data,
                   build_transformer_inputs, build_vocab, clean_text, device,
                   format_mcq, load_with_retry, mapk, seed_everything,
                   top1_accuracy, top1_f1)

WANDB_PROJECT = "23f1000362-t22026"


def train_one(model, cnn_Xtr, cnn_ytr, cnn_Xva, cnn_yva, epochs=40, bs=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    # adam optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(cnn_Xtr)

    # validation data kept on gpu, used every epoch
    Xva_t = torch.tensor(cnn_Xva).to(device)
    yva_t = torch.tensor(cnn_yva).to(device)
    for ep in range(epochs):
        model.train()
        # shuffle training samples
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(
                model(torch.tensor(cnn_Xtr[idx]).to(device)),
                torch.tensor(cnn_ytr[idx]).to(device)
            )
            loss.backward()  # compute gradients
            opt.step()       # update weights
            tot += loss.item() * len(idx)

        # validation loss for this epoch
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(Xva_t), yva_t).item()
        # save train and val loss to wandb every epoch
        if wandb.run is not None:
            wandb.log({"train/loss": tot / n, "eval/loss": val_loss, "train/epoch": ep + 1})
        # print loss every 10 epochs
        if (ep + 1) % 10 == 0:
            print(f"  ep{ep+1} train_loss {tot/n:.3f} val_loss {val_loss:.3f}")


def evaluate(model, Xe, ye_l, name):
    model.eval()

    # no gradients needed during evaluation
    with torch.no_grad():
        lg = model(torch.tensor(Xe).to(device)).cpu().numpy()

    # top 3 predicted options
    preds = [[ID2LETTER[i] for i in np.argsort(-lg[k])[:3]] for k in range(len(Xe))]

    m3, acc = mapk(ye_l, preds), top1_accuracy(ye_l, preds)

    print(f"{name}: MAP@3 {m3:.4f}  acc {acc:.4f}")
    return m3, acc


# model 1 : tf-idf + logistic regression

def fit_tfidf(train, test):
    full_corpus = []
    # collect all question and option text
    for df in [train, test]:
        for _, row in df.iterrows():
            full_corpus.append(row['prompt'])
            for o in OPTS:
                full_corpus.append(row[o])

    # word level tf-idf
    word_tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),   # use unigrams and bigrams
        max_features=25000,
        sublinear_tf=True
    )

    # character level tf-idf
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',   # character ngrams inside word boundaries
        ngram_range=(3, 5),
        max_features=15000,
        sublinear_tf=True
    )

    # build vocabulary from full corpus
    word_tfidf.fit(full_corpus)
    char_tfidf.fit(full_corpus)

    print(f"word tfidf vocab: {len(word_tfidf.vocabulary_)} | char tfidf vocab: {len(char_tfidf.vocabulary_)}")
    return word_tfidf, char_tfidf


def build_row_features(row, word_tfidf, char_tfidf):
    # tf-idf vectors for the question
    p_word = word_tfidf.transform([row['prompt']])
    p_char = char_tfidf.transform([row['prompt']])
    feats = []
    for o in OPTS:
        o_word = word_tfidf.transform([row[o]])
        o_char = char_tfidf.transform([row[o]])
        # word similarity
        feats.append(cosine_similarity(p_word, o_word)[0][0])
        # character similarity
        feats.append(cosine_similarity(p_char, o_char)[0][0])
        # option length compared to question
        feats.append(len(row[o].split()) / max(len(row['prompt'].split()), 1))
        # token overlap score
        p_tok = set(clean_text(row['prompt']).split())
        o_tok = set(clean_text(row[o]).split())
        feats.append(len(p_tok & o_tok) / max(len(p_tok | o_tok), 1))
    return feats


def train_tfidf(train, test, actuals):
    word_tfidf, char_tfidf = fit_tfidf(train, test)

    print("building features for all train rows")

    X = np.array([build_row_features(row, word_tfidf, char_tfidf) for _, row in train.iterrows()])
    y = train['answer'].map(LETTER2ID).values

    # out-of-fold predictions for fair evaluation
    oof_probs = np.zeros((len(train), 5))

    # stratified split keeps class balance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(
            max_iter=2000,   # allow more iterations to converge
            C=2.0,
            random_state=42
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X[va_idx])
        print(f"fold {fold+1}/5 done")

    # train final model on full training data
    tfidf_clf = LogisticRegression(max_iter=2000, C=2.0, random_state=42)
    tfidf_clf.fit(X, y)

    # top 3 predictions from oof probabilities
    oof_ranked = [[ID2LETTER[i] for i in np.argsort(-p)[:3]] for p in oof_probs]
    map3_tfidf = round(mapk(actuals, oof_ranked), 4)
    acc_tfidf = top1_accuracy(actuals, oof_ranked)
    f1_tfidf = top1_f1(actuals, oof_ranked)

    print(f"\ntfidf (5-fold OOF)  MAP@3 : {map3_tfidf} | acc: {acc_tfidf}, f1: {f1_tfidf}")

    # log metrics to wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        name="model1-tfidf-supervised",
        config={
            "features": "word+char tfidf cosine, len_ratio, jaccard",
            "cv": "5fold_oof",
            "C": 2.0
        },
        tags=["milestone-5", "from-scratch", "tfidf"]
    )
    wandb.log({
        "map3": map3_tfidf,
        "accuracy": acc_tfidf,
        "f1": f1_tfidf
    })
    wandb.finish()

    print("Model 1 complete - tfidf trained")
    return tfidf_clf, word_tfidf, char_tfidf, map3_tfidf, acc_tfidf, f1_tfidf


# model 2 : roberta + lora

def train_roberta(train, train_inputs, train_labels, actuals):
    cls_tok = load_with_retry(AutoTokenizer.from_pretrained, "roberta-base")
    cls_base = load_with_retry(
        AutoModelForSequenceClassification.from_pretrained,
        "roberta-base",
        num_labels=5  # five output classes
    )
    # tokenize training data
    cls_enc = cls_tok(train_inputs, truncation=True, max_length=256, padding=False)
    cls_enc['labels'] = train_labels
    hf_cls = Dataset.from_dict(cls_enc)

    # keep some data for validation
    _split = hf_cls.train_test_split(test_size=0.1, seed=42)
    hf_cls_train, hf_cls_val = _split['train'], _split['test']
    print(f"train: {len(hf_cls_train)}  val: {len(hf_cls_val)}")

    # set random seeds
    seed_everything(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # lora settings
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],  # apply lora here
        lora_dropout=0.1,
        bias="none"
    )
    lora_model = get_peft_model(cls_base, lora_cfg).to(device)
    lora_model.print_trainable_parameters()

    # pad batches automatically
    cls_collator = DataCollatorWithPadding(cls_tok)

    cls_args = TrainingArguments(
        output_dir="/kaggle/working/lora_cls",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,  # bigger effective batch
        learning_rate=2e-4,
        warmup_steps=50,
        fp16=False,
        logging_steps=25,
        eval_strategy="epoch",  # validate after every epoch
        save_strategy="no",
        report_to="wandb",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED
    )

    wandb.init(
        project=WANDB_PROJECT,
        name="model4-lora-roberta",
        config={
            "lora_r": 16,
            "lora_alpha": 32,
            "lr": 2e-4,
            "epochs": 6,
            "seed": SEED,
            "max_length": 256,
            "fp16": False,
            "deterministic": True,
            "eval": "90/10 split"
        },
        tags=["milestone-5", "lora", "roberta"]
    )
    cls_trainer = Trainer(
        model=lora_model,
        args=cls_args,
        train_dataset=hf_cls_train,
        eval_dataset=hf_cls_val,
        data_collator=cls_collator,
        processing_class=cls_tok
    )

    print("starting lora roberta finetuning")
    cls_trainer.train()
    # switch to eval mode
    lora_model.eval()

    def rank_lora(row):
        logits = get_lora_logits(lora_model, cls_tok, row)
        scores = {ID2LETTER[i]: logits[i] for i in range(5)}
        # sort by prediction score
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # get top 3 predictions
    lora_preds = [rank_lora(row)[:3] for _, row in train.iterrows()]
    map3_lora = round(mapk(actuals, lora_preds), 4)
    acc_lora = top1_accuracy(actuals, lora_preds)
    f1_lora = top1_f1(actuals, lora_preds)

    print(f"roberta lora  MAP@3 : {map3_lora} | acc: {acc_lora}, f1: {f1_lora}")

    # log final metrics
    wandb.log({
        "map3": map3_lora,
        "accuracy": acc_lora,
        "f1": f1_lora
    })
    wandb.finish()
    print("Model 2 complete - roberta lora trained")
    return lora_model, cls_tok, map3_lora, acc_lora, f1_lora


def get_lora_logits(model, cls_tok, row):
    inputs = cls_tok(
        format_mcq(row),
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        return model(**inputs).logits[0].cpu().numpy()


# model 3 : deberta + lora

def train_deberta(train, train_inputs, train_labels, actuals):
    deberta_tok = load_with_retry(AutoTokenizer.from_pretrained, "microsoft/deberta-v3-base")
    deberta_base = load_with_retry(AutoModelForSequenceClassification.from_pretrained,
                                   "microsoft/deberta-v3-base", num_labels=5)  # 5 output classes

    # tokenize all training data
    deberta_enc = deberta_tok(train_inputs, truncation=True, max_length=256, padding=False)
    deberta_enc['labels'] = train_labels
    hf_deberta = Dataset.from_dict(deberta_enc)

    # split into train and validation
    _split = hf_deberta.train_test_split(test_size=0.1, seed=42)
    hf_deberta_train, hf_deberta_val = _split['train'], _split['test']
    print(f"train: {len(hf_deberta_train)}  val: {len(hf_deberta_val)}")

    # lora configuration
    lora_cfg_deberta = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                                  target_modules=["query_proj", "value_proj"],  # attention layers
                                  lora_dropout=0.1, bias="none")
    lora_deberta = get_peft_model(deberta_base, lora_cfg_deberta).to(device)
    lora_deberta.print_trainable_parameters()

    # dynamic padding
    deberta_collator = DataCollatorWithPadding(deberta_tok)
    args_deberta = TrainingArguments(
        output_dir="/kaggle/working/lora_deberta", num_train_epochs=6,
        per_device_train_batch_size=16, gradient_accumulation_steps=2,  # bigger effective batch
        learning_rate=2e-4, warmup_steps=50,
        fp16=False, logging_steps=25,
        eval_strategy="epoch",  # evaluate every epoch
        save_strategy="no", report_to="wandb",
        dataloader_pin_memory=False, dataloader_num_workers=0, seed=42
    )

    # start wandb run
    wandb.init(project=WANDB_PROJECT, name="model5-lora-deberta",
               config={"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 6,
                       "base_model": "microsoft/deberta-v3-base", "eval": "90/10 split"},
               tags=["milestone-5", "lora", "deberta"])

    trainer_deberta = Trainer(model=lora_deberta, args=args_deberta,
                              train_dataset=hf_deberta_train, eval_dataset=hf_deberta_val,
                              data_collator=deberta_collator, processing_class=deberta_tok)

    print("starting lora deberta finetuning")
    trainer_deberta.train()

    # switch to eval mode
    lora_deberta.eval()

    def rank_lora_deberta(row):
        logits = get_deberta_logits(lora_deberta, deberta_tok, row)
        scores = {ID2LETTER[i]: logits[i] for i in range(5)}
        # sort by score
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # top 3 predictions
    deberta_preds = [rank_lora_deberta(row)[:3] for _, row in train.iterrows()]
    map3_deberta = round(mapk(actuals, deberta_preds), 4)
    acc_deberta = top1_accuracy(actuals, deberta_preds)
    f1_deberta = top1_f1(actuals, deberta_preds)
    print(f"deberta lora  MAP@3 : {map3_deberta} | acc: {acc_deberta}, f1: {f1_deberta}")

    # log metrics
    wandb.log({"map3": map3_deberta, "accuracy": acc_deberta, "f1": f1_deberta})
    wandb.finish()
    print("Model 3 complete - deberta lora trained")
    return lora_deberta, deberta_tok, map3_deberta, acc_deberta, f1_deberta


def get_deberta_logits(lora_deberta, deberta_tok, row):
    inputs = deberta_tok(format_mcq(row), return_tensors="pt",
                         truncation=True, max_length=256).to(device)
    # inference only
    with torch.no_grad():
        return lora_deberta(**inputs).logits[0].cpu().numpy()


# model 4 : textcnn

class TextCNN(nn.Module):
    def __init__(self, V, D=100, C=128, kernels=(3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=0)
        # different kernel sizes capture different n-grams
        self.convs = nn.ModuleList([nn.Conv1d(D, C, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(C * len(kernels), 5)

    def forward(self, x):
        # embedding output -> conv1d format
        e = self.emb(x).transpose(1, 2)                    # B,D,L

        # max pooling after each convolution
        pooled = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]  # B,C each

        # combine all features
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


def train_textcnn(vocab, cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva, cnn_ytr_l, cnn_yva_l):
    cnn_model = TextCNN(len(vocab)).to(device)
    # total model parameters
    print("TextCNN params:", sum(p.numel() for p in cnn_model.parameters()))

    # wandb.init for train_loss / val_loss logged every epoch
    run = wandb.init(project=WANDB_PROJECT, name="model6-textcnn-custom-dl",
                     config={"arch": "TextCNN", "kernels": [3, 4, 5], "channels": 128,
                             "emb": 100, "epochs": 40, "lr": 1e-3},
                     tags=["milestone-5", "custom-dl", "textcnn"])

    train_one(cnn_model, cnn_Xtr, cnn_ytr, cnn_Xva, cnn_yva)

    # evaluate on train and validation
    m3_cnn_tr, acc_cnn_tr = evaluate(cnn_model, cnn_Xtr, cnn_ytr_l, "TextCNN train")
    m3_cnn_va, acc_cnn_va = evaluate(cnn_model, cnn_Xva, cnn_yva_l, "TextCNN val  ")

    # top-3 ranked letters per row, needed for macro f1
    def rank_cnn(X):
        cnn_model.eval()
        with torch.no_grad():
            lg = cnn_model(torch.tensor(X).to(device)).cpu().numpy()
        return [[ID2LETTER[i] for i in np.argsort(-lg[k])[:3]] for k in range(len(X))]

    f1_cnn_tr = top1_f1(cnn_ytr_l, rank_cnn(cnn_Xtr))
    f1_cnn_va = top1_f1(cnn_yva_l, rank_cnn(cnn_Xva))
    print(f"TextCNN f1 (macro)  train {f1_cnn_tr:.4f}  val {f1_cnn_va:.4f}")

    # save train and val metrics
    wandb.log({"map3_train": m3_cnn_tr, "map3_val": m3_cnn_va,
               "acc_train": acc_cnn_tr, "acc_val": acc_cnn_va,
               "f1_train": f1_cnn_tr, "f1_val": f1_cnn_va})
    wandb.finish()
    print("Model 4 complete - TextCNN trained")
    return cnn_model, m3_cnn_tr, m3_cnn_va, acc_cnn_tr, acc_cnn_va, f1_cnn_tr, f1_cnn_va


# model 5 : bilstm with attention

class BiLSTMAttn(nn.Module):
    def __init__(self, V, D=100, H=64):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=0)
        # bidirectional lstm
        self.lstm = nn.LSTM(D, H, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * H, 1)          # attention score for each token
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * H, 5)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))          # B,L,2H
        # calculate attention scores
        scores = self.attn(out).squeeze(-1)      # B,L
        # ignore padding tokens
        scores = scores.masked_fill(x == 0, float("-inf"))
        # attention weights
        w = torch.softmax(scores, dim=1).unsqueeze(-1)       # B,L,1
        # weighted sum of all outputs
        v = (out * w).sum(dim=1)                 # B,2H
        return self.fc(self.drop(v))


def train_bilstm(vocab, cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva, cnn_ytr_l, cnn_yva_l):
    rnn_model = BiLSTMAttn(len(vocab)).to(device)

    # total model parameters
    print("BiLSTM params:", sum(p.numel() for p in rnn_model.parameters()))

    # wandb.init for train_loss / val_loss logged every epoch
    run = wandb.init(project=WANDB_PROJECT, name="model7-bilstm-rnn",
                     config={"arch": "BiLSTM+attention", "hidden": 64, "emb": 100,
                             "epochs": 40, "lr": 1e-3},
                     tags=["milestone-5", "rnn", "bilstm"])

    train_one(rnn_model, cnn_Xtr, cnn_ytr, cnn_Xva, cnn_yva)
    # evaluate on train and validation
    m3_rnn_tr, acc_rnn_tr = evaluate(rnn_model, cnn_Xtr, cnn_ytr_l, "BiLSTM train")
    m3_rnn_va, acc_rnn_va = evaluate(rnn_model, cnn_Xva, cnn_yva_l, "BiLSTM val  ")

    # top-3 ranked letters per row, needed for macro f1
    def rank_rnn(X):
        rnn_model.eval()
        with torch.no_grad():
            lg = rnn_model(torch.tensor(X).to(device)).cpu().numpy()
        return [[ID2LETTER[i] for i in np.argsort(-lg[k])[:3]] for k in range(len(X))]

    f1_rnn_tr = top1_f1(cnn_ytr_l, rank_rnn(cnn_Xtr))
    f1_rnn_va = top1_f1(cnn_yva_l, rank_rnn(cnn_Xva))
    print(f"BiLSTM f1 (macro)  train {f1_rnn_tr:.4f}  val {f1_rnn_va:.4f}")

    wandb.log({"map3_train": m3_rnn_tr, "map3_val": m3_rnn_va,
               "acc_train": acc_rnn_tr, "acc_val": acc_rnn_va,
               "f1_train": f1_rnn_tr, "f1_val": f1_rnn_va})

    wandb.finish()
    print("Model 5 complete - BiLSTM trained")
    return rnn_model, m3_rnn_tr, m3_rnn_va, acc_rnn_tr, acc_rnn_va, f1_rnn_tr, f1_rnn_va


def main(data_dir="/kaggle/input/competitions/smart-mcq-solver-challenge",
         models_dir="../models"):
    import pickle

    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    actuals = train['answer'].tolist()

    train_inputs, train_labels = build_transformer_inputs(train)
    vocab = build_vocab(train)
    cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva, cnn_ytr_l, cnn_yva_l = build_sequence_data(train, vocab)
    print("Utility Functions ready - vocab:", len(vocab), "| train:", len(cnn_Xtr), "val:", len(cnn_Xva))

    os.makedirs(models_dir, exist_ok=True)
    with open(f"{models_dir}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    tfidf_clf, word_tfidf, char_tfidf, *_ = train_tfidf(train, test, actuals)
    with open(f"{models_dir}/tfidf_clf.pkl", "wb") as f:
        pickle.dump(tfidf_clf, f)

    lora_model, cls_tok, *_ = train_roberta(train, train_inputs, train_labels, actuals)
    lora_model.save_pretrained(f"{models_dir}/roberta")
    cls_tok.save_pretrained(f"{models_dir}/roberta")

    lora_deberta, deberta_tok, *_ = train_deberta(train, train_inputs, train_labels, actuals)
    lora_deberta.save_pretrained(f"{models_dir}/deberta")
    deberta_tok.save_pretrained(f"{models_dir}/deberta")

    cnn_model, *_ = train_textcnn(vocab, cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva, cnn_ytr_l, cnn_yva_l)
    torch.save(cnn_model.state_dict(), f"{models_dir}/textcnn.pt")

    rnn_model, *_ = train_bilstm(vocab, cnn_Xtr, cnn_Xva, cnn_ytr, cnn_yva, cnn_ytr_l, cnn_yva_l)
    torch.save(rnn_model.state_dict(), f"{models_dir}/bilstm.pt")


if __name__ == "__main__":
    main()
