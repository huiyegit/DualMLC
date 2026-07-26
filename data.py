#!/usr/bin/env python
"""
data.py -- datasets, collation and dataloader construction.

Each document is tokenized TWICE (once per encoder) because the two branches use
different vocabularies.
"""
from pathlib import Path

import scipy.sparse as smat
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


# ---------------------------------------------------------------- tokenizers


def load_tokenizers(qwen_name, bert_name):
    qtok = AutoTokenizer.from_pretrained(str(qwen_name))
    if qtok.pad_token is None:
        qtok.pad_token = qtok.eos_token
    btok = AutoTokenizer.from_pretrained(str(bert_name))
    return qtok, btok


def encode_dual(text, qtok, btok, qmax, bmax):
    qenc = qtok(text, padding="max_length", truncation=True,
                max_length=qmax, return_tensors="pt")
    benc = btok(text, padding="max_length", truncation=True,
                max_length=bmax, return_tensors="pt")
    out = {
        "q_input_ids": qenc["input_ids"].squeeze(0),
        "q_attention_mask": qenc["attention_mask"].squeeze(0),
        "b_input_ids": benc["input_ids"].squeeze(0),
        "b_attention_mask": benc["attention_mask"].squeeze(0),
    }
    if "token_type_ids" in benc:
        out["b_token_type_ids"] = benc["token_type_ids"].squeeze(0)
    return out


# ---------------------------------------------------------------- datasets


def split_paths(data_dir, split):
    d = Path(data_dir)
    return d / f"X.{split}.txt", d / f"Y.{split}.npz"


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


class DualLabeledDataset(Dataset):
    """(text, multi-hot label vector) pairs, tokenized for both encoders."""

    def __init__(self, texts_path, labels_path, qwen_tok, bert_tok,
                 qwen_max_len, bert_max_len):
        self.texts = read_texts(texts_path)
        self.Y = smat.load_npz(str(labels_path)).tocsr()
        if len(self.texts) != self.Y.shape[0]:
            raise ValueError(
                f"text/label row mismatch: {len(self.texts)} texts vs "
                f"{self.Y.shape[0]} label rows ({texts_path} / {labels_path})")
        self.num_labels = int(self.Y.shape[1])
        self.qtok, self.btok = qwen_tok, bert_tok
        self.qmax, self.bmax = qwen_max_len, bert_max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        out = encode_dual(self.texts[idx], self.qtok, self.btok, self.qmax, self.bmax)
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        y[self.Y[idx].indices] = 1.0
        out["labels"] = y
        return out


class DualTextDataset(Dataset):
    """Unlabeled documents, for inference."""

    def __init__(self, texts, qwen_tok, bert_tok, qwen_max_len, bert_max_len):
        self.texts = list(texts)
        self.qtok, self.btok = qwen_tok, bert_tok
        self.qmax, self.bmax = qwen_max_len, bert_max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return encode_dual(self.texts[idx], self.qtok, self.btok, self.qmax, self.bmax)


# Backwards-compatible alias for the original class name.
Wiki10DualDataset = DualLabeledDataset


# ---------------------------------------------------------------- batching


_TENSOR_KEYS = ("q_input_ids", "q_attention_mask", "b_input_ids",
                "b_attention_mask", "b_token_type_ids", "labels")


def dual_collate(batch):
    out = {}
    for key in _TENSOR_KEYS:
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    return out


def batch_to_device(batch, device):
    """Model kwargs only -- 'labels' is deliberately left out."""
    kw = {
        "q_input_ids": batch["q_input_ids"].to(device, non_blocking=True),
        "q_attention_mask": batch["q_attention_mask"].to(device, non_blocking=True),
        "b_input_ids": batch["b_input_ids"].to(device, non_blocking=True),
        "b_attention_mask": batch["b_attention_mask"].to(device, non_blocking=True),
    }
    if "b_token_type_ids" in batch:
        kw["b_token_type_ids"] = batch["b_token_type_ids"].to(device, non_blocking=True)
    return kw


# ---------------------------------------------------------------- loaders


def build_train_loader(dataset, cfg, rank=0, world_size=1):
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler,
                            num_workers=cfg["num_workers"], pin_memory=True,
                            drop_last=True, collate_fn=dual_collate)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], pin_memory=True,
                            drop_last=True, collate_fn=dual_collate)
    return loader, sampler


def build_eval_loader(dataset, cfg, rank=0, world_size=1):
    """Shard the eval set across ranks by strided slicing.

    DistributedSampler would pad the last shard by repeating samples, which
    silently biases the metrics. Strided Subset slicing partitions exactly, so
    all_reduce-ing the hit counts gives the same number a single-GPU run would.
    """
    if world_size > 1:
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
    return DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                      num_workers=cfg["num_workers"], pin_memory=True,
                      collate_fn=dual_collate)


def build_predict_loader(texts, qtok, btok, cfg, num_workers=0):
    ds = DualTextDataset(texts, qtok, btok,
                         cfg["qwen_max_length"], cfg["bert_max_length"])
    return DataLoader(ds, batch_size=cfg["eval_batch_size"], shuffle=False,
                      num_workers=num_workers, collate_fn=dual_collate)
