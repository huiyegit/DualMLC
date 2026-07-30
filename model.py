#!/usr/bin/env python
"""Single-encoder BERT/Qwen model, evaluation, and checkpoint I/O."""
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel

from config import arch_config
from data import batch_to_device, load_tokenizer

LOG = logging.getLogger(__name__)
_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


class SingleEncoderClassifier(nn.Module):
    def __init__(self, cfg, num_labels, lora_cfg=None, qwen_adapter_dir=None,
                 qwen_dtype=torch.bfloat16, lora_trainable=True,
                 bert_source=None, qwen_source=None):
        super().__init__()
        self.encoder_type = cfg["encoder"]
        self.num_labels = int(num_labels)

        if self.encoder_type == "qwen":
            source = qwen_source or cfg["qwen_name"]
            self.backbone = AutoModel.from_pretrained(
                str(source), **{_DTYPE_KW: qwen_dtype})
            if lora_cfg is not None:
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            elif qwen_adapter_dir is not None:
                self.backbone = PeftModel.from_pretrained(
                    self.backbone, str(qwen_adapter_dir), is_trainable=lora_trainable)
            self.pool = cfg["qwen_pool"]
            self.layers = int(cfg["qwen_layers"])
            self.reduce_factor = int(cfg["reduce_factor"])
            hidden = int(self.backbone.config.hidden_size)
            if hidden % self.reduce_factor != 0:
                raise ValueError("reduce_factor %d does not divide hidden size %d" %
                                 (self.reduce_factor, hidden))
            self.output_dim = hidden // self.reduce_factor
            self.dropout = nn.Dropout(cfg["qwen_dropout"])
        elif self.encoder_type == "bert":
            source = bert_source or cfg["bert_name"]
            try:
                self.backbone = AutoModel.from_pretrained(
                    str(source), add_pooling_layer=False)
            except TypeError:
                self.backbone = AutoModel.from_pretrained(str(source))
            self.pool = cfg["bert_pool"]
            self.layers = int(cfg["bert_layers"])
            self.reduce_factor = 1
            self.output_dim = int(self.backbone.config.hidden_size)
            self.dropout = nn.Dropout(cfg["bert_dropout"])
        else:
            raise ValueError("unsupported encoder: %s" % self.encoder_type)

        self.head = nn.Linear(self.output_dim, self.num_labels)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _multi_layer_hidden(out, k):
        if k <= 1:
            return out.last_hidden_state
        return torch.stack(out.hidden_states[-k:], dim=0).mean(dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": self.layers > 1,
        }
        if self.encoder_type == "bert" and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)
        hidden = self._multi_layer_hidden(out, self.layers)

        if self.pool == "mean":
            pooled = self._mean_pool(hidden, attention_mask)
        elif self.encoder_type == "qwen" and self.pool == "last":
            seq_lens = attention_mask.sum(dim=1) - 1
            idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
            pooled = hidden.gather(1, idx).squeeze(1)
        else:
            pooled = hidden[:, 0]

        pooled = pooled.to(torch.float32)
        if self.encoder_type == "qwen" and self.reduce_factor > 1:
            pooled = pooled.view(
                pooled.shape[0], self.output_dim, self.reduce_factor).mean(dim=-1)
        return self.head(self.dropout(pooled))


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def build_lora_config(cfg):
    return LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        target_modules=list(cfg["lora_targets"]),
        lora_dropout=cfg["lora_dropout"], bias="none", task_type=None)


def build_model(cfg, num_labels):
    lora_cfg = build_lora_config(cfg) if cfg["encoder"] == "qwen" else None
    return SingleEncoderClassifier(cfg, num_labels, lora_cfg=lora_cfg)


@torch.no_grad()
def evaluate(model, loader, device, topk=(1, 3, 5, 10),
             use_amp=True, reduce_ddp=False):
    raw = unwrap(model)
    name = raw.encoder_type
    was_training = model.training
    model.eval()
    topk = tuple(sorted({int(k) for k in topk}))
    counts = torch.zeros(len(topk), dtype=torch.float64, device=device)
    total = torch.zeros((), dtype=torch.float64, device=device)
    amp_on = bool(use_amp) and device.type == "cuda"

    for batch in loader:
        kwargs = batch_to_device(batch, device)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
            logits = model(**kwargs)
        logits = logits.float()
        kmax = min(max(topk), logits.size(-1))
        idx = logits.topk(kmax, dim=-1).indices
        hits = y.gather(1, idx)
        for j, k in enumerate(topk):
            counts[j] += hits[:, :min(k, kmax)].sum().double() / k
        total += y.size(0)

    if reduce_ddp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
        dist.all_reduce(total)

    if was_training:
        model.train()
    n = max(total.item(), 1.0)
    return {name: {k: 100.0 * counts[j].item() / n for j, k in enumerate(topk)}}


def format_metrics(tag, metrics, topk=(1, 3, 5, 10)):
    lines = []
    for name, values in metrics.items():
        cells = " ".join("P@%d=%.2f" % (k, values[k]) for k in topk if k in values)
        lines.append("%s | %-5s %s" % (tag, name.upper(), cells))
    return "\n".join(lines)


ENCODER_DIR = "encoder"
QWEN_ADAPTER_DIR = "qwen_lora_adapter"
TOKENIZER_DIR = "tokenizer"
HEAD_FILE = "head.pt"
ARCH_FILE = "config.json"
METRICS_FILE = "best_metrics.json"


def save_checkpoint(model, tokenizer, out_dir, cfg, num_labels,
                    metrics=None, step=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = unwrap(model)

    if raw.encoder_type == "qwen":
        raw.backbone.save_pretrained(str(out_dir / QWEN_ADAPTER_DIR))
    else:
        raw.backbone.save_pretrained(str(out_dir / ENCODER_DIR))
    torch.save(raw.head.state_dict(), out_dir / HEAD_FILE)
    tokenizer.save_pretrained(str(out_dir / TOKENIZER_DIR))
    (out_dir / ARCH_FILE).write_text(
        json.dumps(arch_config(cfg, num_labels), indent=2), encoding="utf-8")

    if metrics is not None:
        payload = {
            "step": step,
            "metrics": {name: {"P@%d" % k: value for k, value in vals.items()}
                        for name, vals in metrics.items()},
        }
        (out_dir / METRICS_FILE).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    return out_dir


def load_arch(ckpt_dir):
    path = Path(ckpt_dir) / ARCH_FILE
    if not path.exists():
        raise FileNotFoundError("missing checkpoint architecture: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(ckpt_dir, device="cuda", qwen_name=None,
                    merge_lora=False, qwen_dtype=torch.bfloat16):
    ckpt = Path(ckpt_dir)
    arch = load_arch(ckpt)
    cfg = dict(arch)
    device = torch.device(device)

    if cfg["encoder"] == "qwen":
        adapter = ckpt / QWEN_ADAPTER_DIR
        if not adapter.exists():
            raise FileNotFoundError("missing Qwen LoRA adapter: %s" % adapter)
        model = SingleEncoderClassifier(
            cfg, arch["num_labels"], qwen_adapter_dir=adapter,
            qwen_source=qwen_name or cfg["qwen_name"],
            qwen_dtype=qwen_dtype, lora_trainable=False)
        if merge_lora and hasattr(model.backbone, "merge_and_unload"):
            model.backbone = model.backbone.merge_and_unload()
    else:
        source = ckpt / ENCODER_DIR
        if not source.exists():
            source = Path(cfg["bert_name"])
        model = SingleEncoderClassifier(
            cfg, arch["num_labels"], bert_source=source)

    model.head.load_state_dict(torch.load(ckpt / HEAD_FILE, map_location="cpu"))
    model.to(device).eval()
    model.requires_grad_(False)

    tok_source = ckpt / TOKENIZER_DIR
    tokenizer = load_tokenizer(cfg, tok_source if tok_source.exists() else None)
    return model, tokenizer, arch
