#!/usr/bin/env python
"""Single-tokenizer datasets, collation, and dataloaders."""
from pathlib import Path

import scipy.sparse as smat
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


def model_name(cfg):
    return cfg["bert_name"] if cfg["encoder"] == "bert" else cfg["qwen_name"]


def max_length(cfg):
    return cfg["bert_max_length"] if cfg["encoder"] == "bert" else cfg["qwen_max_length"]


def load_tokenizer(cfg, source=None):
    tok = AutoTokenizer.from_pretrained(str(source or model_name(cfg)))
    if tok.pad_token is None:
        if tok.eos_token is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tok.pad_token = tok.eos_token
    return tok


def encode_single(text, tokenizer, max_len):
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    out = {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
    }
    if "token_type_ids" in enc:
        out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
    return out


def split_paths(data_dir, split):
    d = Path(data_dir)
    return d / ("X.%s.txt" % split), d / ("Y.%s.npz" % split)


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class SingleLabeledDataset(Dataset):
    def __init__(self, texts_path, labels_path, tokenizer, max_len):
        self.texts = read_texts(texts_path)
        self.Y = smat.load_npz(str(labels_path)).tocsr()
        if len(self.texts) != self.Y.shape[0]:
            raise ValueError("text/label row mismatch: %d texts vs %d label rows" %
                             (len(self.texts), self.Y.shape[0]))
        self.num_labels = int(self.Y.shape[1])
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        out = encode_single(self.texts[idx], self.tokenizer, self.max_len)
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        y[self.Y[idx].indices] = 1.0
        out["labels"] = y
        return out


class SingleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return encode_single(self.texts[idx], self.tokenizer, self.max_len)


_TENSOR_KEYS = ("input_ids", "attention_mask", "token_type_ids", "labels")


def single_collate(batch):
    out = {}
    for key in _TENSOR_KEYS:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def batch_to_device(batch, device):
    out = {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }
    if "token_type_ids" in batch:
        out["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)
    return out


def build_train_loader(dataset, cfg, rank=0, world_size=1):
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler,
                            num_workers=cfg["num_workers"], pin_memory=True,
                            drop_last=True, collate_fn=single_collate)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], pin_memory=True,
                            drop_last=True, collate_fn=single_collate)
    return loader, sampler


def build_eval_loader(dataset, cfg, rank=0, world_size=1):
    if world_size > 1:
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
    return DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                      num_workers=cfg["num_workers"], pin_memory=True,
                      collate_fn=single_collate)


def build_predict_loader(texts, tokenizer, cfg, num_workers=0):
    ds = SingleTextDataset(texts, tokenizer, max_length(cfg))
    return DataLoader(ds, batch_size=cfg["eval_batch_size"], shuffle=False,
                      num_workers=num_workers, collate_fn=single_collate)
