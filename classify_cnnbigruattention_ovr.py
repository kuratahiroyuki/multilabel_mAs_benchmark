#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN + BiGRU + Attention / one-vs-rest classifier for RMBase sequence windows with 5-fold CV.

This version supports multiple sequence encodings concatenated along the channel axis.
Examples:
    --encodes OH
    --encodes W2V1
    --encodes OH+W2V1
    --encodes OH+W2V1+NUCPROP+KMER2
    --encodes OH,W2V1,OH+W2V1+NUCPROP

Supported encoding blocks:
    OH       : 5-channel one-hot encoding for A/C/G/U/N.
    W2V/W2V1: Word2Vec embedding, k-mer inferred from suffix or --w2v_kmer.
    W2V2     : Word2Vec embedding with k=2.
    W2V3     : Word2Vec embedding with k=3.
    NUCPROP  : simple nucleotide physicochemical-style properties per position.
    KMER2    : position-wise dinucleotide one-hot channels.
    KMER3    : position-wise trinucleotide one-hot channels.
    GC       : GC/AU/N binary indicator channels per position.


Main changes from classify_cnn_cv_improved.py
---------------------------------------------
1. Trains CNN as one-vs-rest multilabel classifier using BCE/Focal-BCE loss.
2. Evaluates macro/per-class metrics using the same multilabel protocol as classify_ml_ext.py.
3. Tunes a separate validation threshold for each class to maximize per-class F1.
4. Uses a CNN + BiGRU + Attention backbone: CNN local motif extraction,
   bidirectional recurrent context modeling, and attention pooling.
5. Supports enhanced multi-encoding input by concatenating OH, W2V, NUCPROP,
   KMER2/KMER3, and GC-style channels.
6. Uses positive-class weights in BCEWithLogitsLoss for minority-class sensitivity.
7. Keeps output files similar to classify_ml_ext.py / classify_cnn_cv_improved.py.

Expected CSV columns:
    site_id,gene_name,transcript_id,sequence,
    label_m6A,label_m1A,label_Am,label_AtoI,label_nonA

Expected input layout:
    split_dir/
    ├── test.csv
    └── cross_val/
        ├── 1/cv_train_1.csv, cv_val_1.csv
        ├── 2/cv_train_2.csv, cv_val_2.csv
        ...
        └── 5/cv_train_5.csv, cv_val_5.csv
"""

import os
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    label_ranking_average_precision_score,
)

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

LABEL_NAMES = ["m6A", "m1A", "Am", "AtoI", "nonA"]
LABEL_COLUMNS = [f"label_{name}" for name in LABEL_NAMES]
#METADATA_COLUMNS = ["site_id", "gene_name", "transcript_id", "sequence"]
METADATA_COLUMNS = ["site_id", "sequence"] ###
BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4, "-": 4}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_and_filter_csv(path, sequence_col="sequence", label_cols=None):
    """Load a multilabel CSV and validate the five binary target columns."""
    label_cols = LABEL_COLUMNS if label_cols is None else list(label_cols)
    df = pd.read_csv(path, sep=",")

    required = list(dict.fromkeys([*METADATA_COLUMNS, sequence_col, *label_cols]))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True).copy()
    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
        invalid = ~df[col].isin([0, 1])
        if invalid.any():
            bad_values = sorted(df.loc[invalid, col].unique().tolist())
            raise ValueError(
                f"Column '{col}' in {path} must contain only 0/1 values; "
                f"found {bad_values[:10]}"
            )
        df[col] = df[col].astype(np.float32)
    return df


def normalize_sequence(seq, max_len):
    seq = str(seq).upper().replace("T", "U")
    seq = "".join(ch if ch in "ACGUN-" else "N" for ch in seq)
    return seq[:max_len].ljust(max_len, "N")


def one_hot_encode(seq, max_len, feature_dim=5):
    seq = normalize_sequence(seq, max_len)
    x = np.zeros((feature_dim, max_len), dtype=np.float32)
    for i, base in enumerate(seq):
        x[BASE_TO_ID.get(base, 4), i] = 1.0
    return x


def seq_to_w2v_embedding(seq, w2v_model, w2v_kmer=1, max_len=None):
    """Return W2V encoding as [embedding_dim, max_len] for Conv1d.

    For k-mer W2V, valid token positions are 0..max_len-k.
    Remaining right-side positions are zero padded so W2V can be concatenated
    with OH / NUCPROP / KMER channels that use length=max_len.
    """
    if w2v_model is None:
        raise ValueError("w2v_model is required for W2V encoding.")

    kv = w2v_model.wv if hasattr(w2v_model, "wv") else w2v_model
    embedding_dim = int(kv.vector_size)
    seq = normalize_sequence(seq, max_len) if max_len is not None else str(seq).upper().replace("T", "U")
    out_len = int(max_len) if max_len is not None else len(seq)
    x = np.zeros((embedding_dim, out_len), dtype=np.float32)

    token_len = max(min(len(seq) - int(w2v_kmer) + 1, out_len), 1)
    for j in range(token_len):
        token = seq[j:j + int(w2v_kmer)]
        if hasattr(kv, "key_to_index") and token in kv.key_to_index:
            x[:, j] = np.asarray(kv[token], dtype=np.float32)
    return x


def nucleotide_property_encode(seq, max_len):
    """Simple per-position nucleotide property channels.

    Channels:
      0 purine(A/G)=1, pyrimidine(C/U)=0, unknown=0.5
      1 amino(A/C)=1, keto(G/U)=0, unknown=0.5
      2 weak H-bond(A/U)=1, strong H-bond(C/G)=0, unknown=0.5
      3 normalized base id A,C,G,U,N -> 0,1/3,2/3,1,0.5
    """
    seq = normalize_sequence(seq, max_len)
    maps = {
        "A": (1.0, 1.0, 1.0, 0.0),
        "C": (0.0, 1.0, 0.0, 1.0 / 3.0),
        "G": (1.0, 0.0, 0.0, 2.0 / 3.0),
        "U": (0.0, 0.0, 1.0, 1.0),
        "T": (0.0, 0.0, 1.0, 1.0),
        "N": (0.5, 0.5, 0.5, 0.5),
        "-": (0.5, 0.5, 0.5, 0.5),
    }
    x = np.zeros((4, max_len), dtype=np.float32)
    for i, base in enumerate(seq):
        x[:, i] = np.asarray(maps.get(base, maps["N"]), dtype=np.float32)
    return x


def gc_indicator_encode(seq, max_len):
    """Three per-position channels: GC, AU, and unknown/gap."""
    seq = normalize_sequence(seq, max_len)
    x = np.zeros((3, max_len), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in "GC":
            x[0, i] = 1.0
        elif base in "AU":
            x[1, i] = 1.0
        else:
            x[2, i] = 1.0
    return x


def kmer_position_onehot_encode(seq, max_len, k=2):
    """Position-wise k-mer one-hot channels, zero padded on the right.

    For KMER2, channels are 4^2=16. For KMER3, channels are 4^3=64.
    Ambiguous k-mers containing N/gap remain all-zero.
    """
    seq = normalize_sequence(seq, max_len)
    alphabet = "ACGU"
    vocab = [a + b if k == 2 else a + b + c for a in alphabet for b in alphabet for c in (alphabet if k == 3 else "")]
    if k == 2:
        vocab = [a + b for a in alphabet for b in alphabet]
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    x = np.zeros((len(vocab), max_len), dtype=np.float32)
    token_len = max(max_len - k + 1, 1)
    for j in range(token_len):
        token = seq[j:j + k]
        idx = token_to_id.get(token)
        if idx is not None:
            x[idx, j] = 1.0
    return x


def parse_encoding_blocks(encode_type):
    """Split one encoding specification into blocks.

    'OH+W2V1+NUCPROP' -> ['OH', 'W2V1', 'NUCPROP']
    A single legacy encoding such as 'W2V2' also works.
    """
    if encode_type is None:
        raise ValueError("--encodes must be specified, e.g. --encodes OH or --encodes OH+W2V1")
    return [part.strip().upper() for part in str(encode_type).replace(" ", "").split("+") if part.strip()]


def infer_w2v_kmer(block, default_kmer):
    if block == "W2V":
        return int(default_kmer)
    suffix = block.replace("W2V", "")
    return int(suffix) if suffix.isdigit() else int(default_kmer)


def load_w2v_models_for_encode(encode_type, args):
    """Load only the W2V models needed by the requested encoding blocks."""
    blocks = parse_encoding_blocks(encode_type)
    models = {}
    for block in blocks:
        if block.startswith("W2V"):
            if args.w2v_path is None:
                raise ValueError("Please specify --w2v_path for W2V encoding.")
            k = infer_w2v_kmer(block, args.w2v_kmer)
            if k not in models:
                model_path = args.w2v_path / f"rna_w2v_{k}_{args.w2v_dim}_64_40_1.pt"
                print(f"[INFO] Loading Word2Vec model for {block}: {model_path}")
                models[k] = Word2Vec.load(str(model_path))
    return models


def get_feature_dim_for_encode(encode_type, w2v_models=None, default_w2v_dim=64, default_w2v_kmer=1):
    dim = 0
    for block in parse_encoding_blocks(encode_type):
        if block == "OH":
            dim += 5
        elif block.startswith("W2V"):
            k = infer_w2v_kmer(block, default_w2v_kmer)
            if w2v_models is not None and k in w2v_models:
                kv = w2v_models[k].wv if hasattr(w2v_models[k], "wv") else w2v_models[k]
                dim += int(kv.vector_size)
            else:
                dim += int(default_w2v_dim)
        elif block in {"NUCPROP", "PROP", "PHYS"}:
            dim += 4
        elif block == "GC":
            dim += 3
        elif block == "KMER2":
            dim += 16
        elif block == "KMER3":
            dim += 64
        else:
            raise ValueError(f"Unsupported encoding block '{block}' in '{encode_type}'")
    return dim


def encode_one_sequence_multi(encode_type, seq, max_len, w2v_models=None, default_w2v_kmer=1):
    arrays = []
    for block in parse_encoding_blocks(encode_type):
        if block == "OH":
            arrays.append(one_hot_encode(seq, max_len, 5))
        elif block.startswith("W2V"):
            k = infer_w2v_kmer(block, default_w2v_kmer)
            if w2v_models is None or k not in w2v_models:
                raise ValueError(f"W2V model for k={k} is not loaded.")
            arrays.append(seq_to_w2v_embedding(seq, w2v_models[k], k, max_len=max_len))
        elif block in {"NUCPROP", "PROP", "PHYS"}:
            arrays.append(nucleotide_property_encode(seq, max_len))
        elif block == "GC":
            arrays.append(gc_indicator_encode(seq, max_len))
        elif block == "KMER2":
            arrays.append(kmer_position_onehot_encode(seq, max_len, k=2))
        elif block == "KMER3":
            arrays.append(kmer_position_onehot_encode(seq, max_len, k=3))
        else:
            raise ValueError(f"Unsupported encoding block '{block}' in '{encode_type}'")
    return np.concatenate(arrays, axis=0).astype(np.float32)


def encode_sequence(encode_type, feature_dim, seqs, max_len=None, w2v_models=None, w2v_kmer=1):
    dict_encoding = {}
    expected_dim = int(feature_dim)
    for idx, seq in enumerate(seqs):
        x = encode_one_sequence_multi(encode_type, seq, int(max_len), w2v_models=w2v_models, default_w2v_kmer=w2v_kmer)
        if x.shape[0] != expected_dim:
            raise RuntimeError(f"Feature dimension mismatch for index {idx}: got {x.shape[0]}, expected {expected_dim}")
        dict_encoding[idx] = x
    return dict_encoding


def dataframe_to_multilabel(df, label_cols=None):
    """Return multilabel targets as an [n_samples, n_labels] float32 array."""
    label_cols = LABEL_COLUMNS if label_cols is None else list(label_cols)
    return df[label_cols].to_numpy(dtype=np.float32, copy=True)


class RNAMultiLabelDataset(Dataset):
    def __init__(self, encode, feature_dim, w2v_models, w2v_kmer, df, sequence_col, label_cols, max_len, label_smoothing=0.0):
        self.encode = encode
        self.feature_dim = feature_dim
        self.w2v_models = w2v_models
        self.w2v_kmer = w2v_kmer
        self.df = df.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.label_cols = list(label_cols)
        self.max_len = int(max_len)
        self.label_smoothing = float(label_smoothing)
        self.y = dataframe_to_multilabel(self.df, self.label_cols)
        if self.label_smoothing > 0:
            # Independent binary targets: positive -> 1-eps, negative -> eps.
            eps = self.label_smoothing
            self.y = self.y * (1.0 - eps) + (1.0 - self.y) * eps
        self.x = encode_sequence(
            self.encode,
            self.feature_dim,
            self.df[sequence_col].values,
            max_len=int(max_len),
            w2v_models=w2v_models,
            w2v_kmer=w2v_kmer,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class ConvBNGELU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class AttentionPooling(nn.Module):
    """Additive attention pooling over sequence positions.

    Input:  [batch, length, hidden_dim]
    Output: [batch, hidden_dim]
    """
    def __init__(self, hidden_dim, attention_dim=128, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, h):
        attn_logits = self.score(h).squeeze(-1)          # [B, L]
        attn_weights = torch.softmax(attn_logits, dim=1) # [B, L]
        context = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)
        return context


class CNNBiGRUAttentionClassifier(nn.Module):
    """CNN + BiGRU + Attention model for RNA multilabel classification.

    The CNN block learns local RNA motifs.
    The BiGRU block learns bidirectional sequence context.
    Attention pooling learns which positions are most informative for classification.
    """
    def __init__(
        self,
        num_classes=5,
        in_channels=5,
        num_filters=128,
        kernel_sizes=(3, 5, 7),
        gru_hidden=128,
        gru_layers=1,
        attention_dim=128,
        hidden_dim=256,
        dropout=0.35,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNGELU(in_channels, num_filters, k, dropout=dropout * 0.25),
                ConvBNGELU(num_filters, num_filters, k, dropout=dropout * 0.25),
            )
            for k in kernel_sizes
        ])
        cnn_dim = num_filters * len(kernel_sizes)
        self.proj = nn.Conv1d(in_channels, cnn_dim, kernel_size=1)
        self.cnn_bn = nn.BatchNorm1d(cnn_dim)
        self.cnn_dropout = nn.Dropout(dropout * 0.5)

        self.bigru = nn.GRU(
            input_size=cnn_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        rnn_dim = gru_hidden * 2
        self.attention = AttentionPooling(rnn_dim, attention_dim=attention_dim, dropout=dropout * 0.5)

        # Combine attention context with max/average pooled GRU features.
        classifier_in = rnn_dim * 3
        self.fc = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: [B, C, L]
        h = torch.cat([branch(x) for branch in self.branches], dim=1)
        h = h + self.proj(x)
        h = self.cnn_dropout(self.cnn_bn(h))             # [B, cnn_dim, L]
        h = h.transpose(1, 2).contiguous()               # [B, L, cnn_dim]

        h, _ = self.bigru(h)                             # [B, L, 2*gru_hidden]
        h_attn = self.attention(h)                       # [B, 2*gru_hidden]
        h_max = torch.max(h, dim=1).values
        h_avg = torch.mean(h, dim=1)
        h = torch.cat([h_attn, h_max, h_avg], dim=1)
        return self.fc(h)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        focal = torch.pow(1.0 - pt, self.gamma)
        return (focal * bce).mean()


def compute_pos_weight(y_train, max_pos_weight=20.0, smoothing=0.05):
    positives = y_train.sum(axis=0).astype(np.float64)
    negatives = y_train.shape[0] - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    if max_pos_weight and max_pos_weight > 0:
        pos_weight = np.minimum(pos_weight, max_pos_weight)
    if smoothing > 0:
        pos_weight = (1.0 - smoothing) * pos_weight + smoothing * np.ones_like(pos_weight)
    return torch.tensor(pos_weight, dtype=torch.float32), positives.astype(int), negatives.astype(int)


def make_weighted_sampler(y_train, minority_multiplier=1.0):
    """Create multilabel-aware sample weights from inverse positive frequencies."""
    positives = y_train.sum(axis=0).astype(np.float64)
    label_weights = len(y_train) / np.maximum(positives, 1.0)
    active = y_train > 0.5
    sample_w = np.ones(len(y_train), dtype=np.float64)
    for i in range(len(y_train)):
        if active[i].any():
            sample_w[i] = label_weights[active[i]].mean()
    if minority_multiplier != 1.0:
        positive_counts = positives[positives > 0]
        if len(positive_counts) > 0:
            median = np.median(positive_counts)
            minority = positives < median
            sample_w[active[:, minority].any(axis=1)] *= minority_multiplier
    sample_w /= np.mean(sample_w)
    return WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def predict_with_thresholds(prob_matrix, thresholds):
    return (prob_matrix >= thresholds.reshape(1, -1)).astype(int)


def tune_thresholds_by_f1(y_true, prob_matrix, label_names, min_threshold=0.01, max_threshold=0.99, step=0.01):
    best_thresholds = []
    records = []
    grid = np.arange(min_threshold, max_threshold + 1e-12, step)
    for i, label in enumerate(label_names):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred_i = (prob_matrix[:, i] >= t).astype(int)
            score = f1_score(y_true[:, i], pred_i, zero_division=0)
            if score > best_f1:
                best_t, best_f1 = float(t), float(score)
        best_thresholds.append(best_t)
        records.append({"label": label, "best_threshold": best_t, "validation_f1": best_f1})
    return np.array(best_thresholds, dtype=np.float32), pd.DataFrame(records)


def safe_binary_roc_auc(y_true_i, prob_i):
    if len(np.unique(y_true_i)) < 2:
        return np.nan
    return roc_auc_score(y_true_i, prob_i)


def evaluate_multilabel(y_true, y_pred, prob_matrix, label_names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        per_class_mcc = np.array([matthews_corrcoef(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
        macro_mcc = float(np.nanmean(per_class_mcc))
        per_class_pr_auc = average_precision_score(y_true, prob_matrix, average=None)
        macro_pr_auc = average_precision_score(y_true, prob_matrix, average="macro")
        per_class_roc_auc = np.array([safe_binary_roc_auc(y_true[:, i], prob_matrix[:, i]) for i in range(y_true.shape[1])])
        macro_roc_auc = float(np.nanmean(per_class_roc_auc))
        try:
            lrap = label_ranking_average_precision_score(y_true, prob_matrix)
        except ValueError:
            lrap = np.nan
        per_class_bal_acc = np.array([balanced_accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
        macro_bal_acc = float(np.nanmean(per_class_bal_acc))

    metrics_df = pd.DataFrame({
        "label": label_names,
        "positive_ratio": y_true.mean(axis=0),
        "F1": per_class_f1,
        "MCC": per_class_mcc,
        "PR_AUC": per_class_pr_auc,
        "ROC_AUC": per_class_roc_auc,
        "Balanced_Accuracy": per_class_bal_acc,
    })
    summary = {
        "Macro_F1": macro_f1,
        "Micro_F1": micro_f1,
        "Weighted_F1": weighted_f1,
        "Macro_MCC": macro_mcc,
        "Macro_PR_AUC": macro_pr_auc,
        "Macro_ROC_AUC": macro_roc_auc,
        "LRAP": lrap,
        "Macro_Balanced_Accuracy": macro_bal_acc,
    }
    return summary, metrics_df


@torch.no_grad()
def collect_predictions(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    probs, y_true = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(y.cpu().numpy())
    return {
        "loss": total_loss / len(loader.dataset),
        "prob": np.vstack(probs),
        "y_true": np.vstack(y_true),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def save_predictions(df_original, y_true, y_pred, prob_matrix, output_path):
    out = df_original.copy()
    for i, label in enumerate(LABEL_NAMES):
        out[f"true_{label}"] = y_true[:, i].astype(int)
        out[f"pred_{label}"] = y_pred[:, i].astype(int)
        out[f"prob_{label}"] = prob_matrix[:, i]
    out.to_csv(output_path, index=False)


def add_fold_to_summary(summary, fold_id, split_name):
    row = {"fold": fold_id, "split": split_name}
    row.update(summary)
    return row


def add_fold_to_per_class(metrics_df, fold_id, split_name):
    out = metrics_df.copy()
    out.insert(0, "split", split_name)
    out.insert(0, "fold", fold_id)
    return out


def summarize_cv(summary_df, output_path):
    metric_cols = [c for c in summary_df.columns if c not in ["fold", "split"]]
    rows = []
    for split_name, sub_df in summary_df.groupby("split"):
        for metric in metric_cols:
            rows.append({
                "split": split_name,
                "metric": metric,
                "mean": sub_df[metric].mean(),
                "std": sub_df[metric].std(ddof=1),
                "min": sub_df[metric].min(),
                "max": sub_df[metric].max(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def summarize_per_class(per_class_df, output_path):
    metric_cols = ["positive_ratio", "F1", "MCC", "PR_AUC", "ROC_AUC", "Balanced_Accuracy"]
    rows = []
    for (split_name, label), sub_df in per_class_df.groupby(["split", "label"]):
        row = {"split": split_name, "label": label}
        for metric in metric_cols:
            row[f"{metric}_mean"] = sub_df[metric].mean()
            row[f"{metric}_std"] = sub_df[metric].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def print_label_counts(name, df, label_cols=None):
    label_cols = LABEL_COLUMNS if label_cols is None else list(label_cols)
    print(f"\n[INFO] {name} samples: {len(df)}")
    print(f"[INFO] {name} positive label counts:")
    counts = df[label_cols].sum(axis=0).astype(int)
    ratios = df[label_cols].mean(axis=0)
    for col in label_cols:
        print(f"  {col}: positives={counts[col]} ratio={ratios[col]:.4f}")


def train_and_evaluate_one_fold(deep, encode, fold_id, train_path, valid_path, test_path, args):

    fold_dir = os.path.join(args.output_dir, deep, encode, str(fold_id))
    os.makedirs(fold_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print(f"[INFO] CNN + BiGRU + Attention one-vs-rest Fold={fold_id}")
    print("=" * 70)

    set_seed(args.seed + fold_id)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")

    encoding_blocks = parse_encoding_blocks(encode)
    w2v_models = load_w2v_models_for_encode(encode, args)
    feature_dim = get_feature_dim_for_encode(
        encode,
        w2v_models=w2v_models,
        default_w2v_dim=args.w2v_dim,
        default_w2v_kmer=args.w2v_kmer,
    )
    print(f"[INFO] Encoding blocks: {encoding_blocks}")
    print(f"[INFO] Total input channels/features: {feature_dim}")

    train_df = load_and_filter_csv(train_path, args.sequence_col, args.label_cols)
    valid_df = load_and_filter_csv(valid_path, args.sequence_col, args.label_cols)
    test_df = load_and_filter_csv(test_path, args.sequence_col, args.label_cols)
    print_label_counts(f"Fold {fold_id} train", train_df, args.label_cols)
    print_label_counts(f"Fold {fold_id} validation", valid_df, args.label_cols)
    print_label_counts("Held-out test", test_df, args.label_cols)

    max_len = args.seqwin or max(
        int(train_df[args.sequence_col].str.len().max()),
        int(valid_df[args.sequence_col].str.len().max()),
        int(test_df[args.sequence_col].str.len().max()),
    )
    print(f"[INFO] max_len={max_len}")

    y_train = dataframe_to_multilabel(train_df, args.label_cols)
    pos_weight, positives, negatives = compute_pos_weight(
        y_train,
        max_pos_weight=args.max_pos_weight,
        smoothing=args.pos_weight_smoothing,
    )
    print("[INFO] Positives/negatives/pos_weight:")
    for label, p, n, w in zip(LABEL_NAMES, positives, negatives, pos_weight):
        print(f"  {label}: pos={p} neg={n} pos_weight={float(w):.4f}")

    train_ds = RNAMultiLabelDataset(encode, feature_dim, w2v_models, args.w2v_kmer, train_df, args.sequence_col, args.label_cols, max_len, args.label_smoothing)
    valid_ds = RNAMultiLabelDataset(encode, feature_dim, w2v_models, args.w2v_kmer, valid_df, args.sequence_col, args.label_cols, max_len, 0.0)
    test_ds = RNAMultiLabelDataset(encode, feature_dim, w2v_models, args.w2v_kmer, test_df, args.sequence_col, args.label_cols, max_len, 0.0)

    if args.use_sampler:
        sampler = make_weighted_sampler(y_train, minority_multiplier=args.minority_sampler_multiplier)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = CNNBiGRUAttentionClassifier(
        num_classes=len(LABEL_NAMES),
        in_channels=feature_dim,
        num_filters=args.num_filters,
        kernel_sizes=tuple(args.kernel_sizes),
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    pos_weight = pos_weight.to(device) if args.use_pos_weight else None
    if args.loss == "focal_bce":
        criterion = FocalBCEWithLogitsLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience)

    best_score = -1.0
    best_model_path = os.path.join(fold_dir, "best_cnnbigruattention_ovr_model.pt")
    history = []
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=args.grad_clip)
        val_pred = collect_predictions(model, valid_loader, criterion, device)

        # Use quick fixed 0.5 threshold during training, but select by ranking metric + F1.
        val_y_pred = predict_with_thresholds(val_pred["prob"], np.full(len(LABEL_NAMES), 0.5, dtype=np.float32))
        val_summary, _ = evaluate_multilabel(val_pred["y_true"], val_y_pred, val_pred["prob"], LABEL_NAMES)
        select_score = val_summary[args.select_metric]
        scheduler.step(select_score)
        lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "learning_rate": lr, "train_loss": train_loss, "val_loss": val_pred["loss"]}
        row.update({f"val_{k}": v for k, v in val_summary.items()})
        history.append(row)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_pred['loss']:.4f} "
            f"val_macroF1={val_summary['Macro_F1']:.4f} val_macroAUC={val_summary['Macro_ROC_AUC']:.4f} "
            f"val_macroPR={val_summary['Macro_PR_AUC']:.4f} lr={lr:.6g}"
        )

        if select_score > best_score:
            best_score = select_score
            epochs_no_improve = 0
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_names": LABEL_NAMES,
                "args": vars(args),
                "max_len": max_len,
                "fold": fold_id,
                "best_score": best_score,
                "select_metric": args.select_metric,
            }, best_model_path)
            print(f"[INFO] Saved best model: {best_model_path}")
            
        else:
            epochs_no_improve += 1

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"[INFO] Early stopping at epoch {epoch}; best {args.select_metric}={best_score:.4f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "training_history.csv"), index=False)

    print("\n[INFO] Loading best model and tuning thresholds...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_raw = collect_predictions(model, valid_loader, criterion, device)
    test_raw = collect_predictions(model, test_loader, criterion, device)

    thresholds, threshold_df = tune_thresholds_by_f1(
        val_raw["y_true"],
        val_raw["prob"],
        LABEL_NAMES,
        min_threshold=args.threshold_min,
        max_threshold=args.threshold_max,
        step=args.threshold_step,
    )
    threshold_df.to_csv(os.path.join(fold_dir, "thresholds.csv"), index=False)
    print("\n[INFO] Tuned thresholds using validation set:")
    print(threshold_df.round(4))

    y_val_pred = predict_with_thresholds(val_raw["prob"], thresholds)
    y_test_pred = predict_with_thresholds(test_raw["prob"], thresholds)

    val_summary, val_metrics_df = evaluate_multilabel(val_raw["y_true"], y_val_pred, val_raw["prob"], LABEL_NAMES)
    test_summary, test_metrics_df = evaluate_multilabel(test_raw["y_true"], y_test_pred, test_raw["prob"], LABEL_NAMES)

    for split, summary, metrics in [("Validation", val_summary, val_metrics_df), ("Test", test_summary, test_metrics_df)]:
        print("\n=========================")
        print(f"Fold {fold_id} {split} Evaluation")
        print("=========================")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")
        print(f"\n[INFO] {split} per-class metrics:")
        print(metrics.round(4))

    val_metrics_df.to_csv(os.path.join(fold_dir, "validation_per_class_metrics.csv"), index=False)
    test_metrics_df.to_csv(os.path.join(fold_dir, "test_per_class_metrics.csv"), index=False)
    pd.DataFrame([val_summary]).to_csv(os.path.join(fold_dir, "validation_summary_metrics.csv"), index=False)
    pd.DataFrame([test_summary]).to_csv(os.path.join(fold_dir, "test_summary_metrics.csv"), index=False)
    save_predictions(valid_df, val_raw["y_true"], y_val_pred, val_raw["prob"], os.path.join(fold_dir, "val_roc.csv"))
    save_predictions(test_df, test_raw["y_true"], y_test_pred, test_raw["prob"], os.path.join(fold_dir, "test_roc.csv"))

    multilabel_report = classification_report(
        test_raw["y_true"], y_test_pred, target_names=LABEL_NAMES,
        digits=4, zero_division=0
    )
    with open(os.path.join(fold_dir, "test_multilabel_classification_report.txt"), "w") as f:
        f.write(multilabel_report)
    print("\n[INFO] Test multilabel classification report:")
    print(multilabel_report)


    return {
        "fold": fold_id,
        "validation_summary": val_summary,
        "test_summary": test_summary,
        "validation_per_class": val_metrics_df,
        "test_per_class": test_metrics_df,
    }


def main(args):
    if len(args.label_cols) != len(LABEL_NAMES):
        raise ValueError(
            f"Expected {len(LABEL_NAMES)} label columns in this order: {LABEL_COLUMNS}; "
            f"received {args.label_cols}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    #test_path = args.test_csv or os.path.join(args.split_dir, "test.csv")
    test_path = os.path.join(args.split_dir, args.test_csv) or os.path.join(args.split_dir, "test.csv") ###
    cv_dir = args.cv_dir or os.path.join(args.split_dir, "cross_val")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test csv not found: {test_path}")
    if not os.path.isdir(cv_dir):
        raise FileNotFoundError(f"CV fold directory not found: {cv_dir}")

    deeps = args.deeps.strip().split(",")
    encodes = args.encodes.strip().split(",")

    for deep in deeps:
        for encode in encodes:
            summary_rows, per_class_rows = [], []
            for fold_id in range(1, args.n_splits + 1):
                train_path = os.path.join(cv_dir, f"{fold_id}", f"cv_train_{fold_id}.csv")
                valid_path = os.path.join(cv_dir, f"{fold_id}", f"cv_val_{fold_id}.csv")
                if not os.path.exists(train_path):
                    raise FileNotFoundError(f"Fold train CSV not found: {train_path}")
                if not os.path.exists(valid_path):
                    raise FileNotFoundError(f"Fold valid CSV not found: {valid_path}")

                result = train_and_evaluate_one_fold(deep, encode, fold_id, train_path, valid_path, test_path, args)
                summary_rows.append(add_fold_to_summary(result["validation_summary"], fold_id, "validation"))
                summary_rows.append(add_fold_to_summary(result["test_summary"], fold_id, "test"))
                per_class_rows.append(add_fold_to_per_class(result["validation_per_class"], fold_id, "validation"))
                per_class_rows.append(add_fold_to_per_class(result["test_per_class"], fold_id, "test"))

            result_dir = os.path.join(args.output_dir, deep, encode)
            os.makedirs(result_dir, exist_ok=True)
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(os.path.join(result_dir, "all_fold_summary_metrics.csv"), index=False)
            print("\n[INFO] CV summary:")
            print(summarize_cv(summary_df, os.path.join(result_dir, "cv_summary_metrics.csv")).round(4))

            per_class_df = pd.concat(per_class_rows, axis=0, ignore_index=True)
            per_class_df.to_csv(os.path.join(result_dir, "all_fold_per_class_metrics.csv"), index=False)
            summarize_per_class(per_class_df, os.path.join(result_dir, "cv_per_class_metrics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deeps", default=None)
    parser.add_argument("--encodes", default=None, help="Comma-separated experiments. Each experiment may combine blocks with +, e.g. OH,W2V1,OH+W2V1+NUCPROP+KMER2")
    parser.add_argument("--w2v_path", type=Path, default=None)
    parser.add_argument("--w2v_kmer", type=int, default=1)
    parser.add_argument("--w2v_dim", type=int, default=64)
    parser.add_argument("--split_dir", default="./dataset")
    parser.add_argument("--cv_dir", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--output_dir", default="./result")
    parser.add_argument("--sequence_col", default="sequence")
    parser.add_argument(
        "--label_cols", nargs="+", default=LABEL_COLUMNS,
        help="Binary multilabel target columns in model-output order."
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seqwin", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--gru_hidden", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--loss", choices=["bce", "focal_bce"], default="focal_bce")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--use_pos_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--pos_weight_smoothing", type=float, default=0.05)
    parser.add_argument("--use_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minority_sampler_multiplier", type=float, default=1.5)

    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--select_metric", choices=["Macro_F1", "Macro_ROC_AUC", "Macro_PR_AUC", "Macro_Balanced_Accuracy"], default="Macro_PR_AUC")

    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4) 
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    main(parser.parse_args())
