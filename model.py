#!/usr/bin/env python
"""
model.py -- the dual-encoder architecture, its checkpoint I/O, and the shared
evaluation routine.

  Branch A: Qwen2.5 (LoRA)          -> pool -> mean-of-N reduce -> head_qwen
  Branch B: bert-base-uncased (full) -> CLS/mean pool           -> head_bert

Each branch has its own BCE loss; predictions are a weighted logit ensemble
``alpha * logits_qwen + (1 - alpha) * logits_bert``.
"""
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
from data import batch_to_device, load_tokenizers

LOG = logging.getLogger(__name__)

# transformers >= 5 renamed from_pretrained(torch_dtype=) to from_pretrained(dtype=).
_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


# ---------------------------------------------------------------- module


class QwenBertDual(nn.Module):
    """Two live encoders, two classifier heads.

    Exactly one of ``lora_cfg`` (fresh training run) or ``qwen_adapter_dir``
    (reload a trained adapter) should normally be given.
    """

    def __init__(self, qwen_name, bert_name, num_labels,
                 lora_cfg=None, qwen_adapter_dir=None,
                 qwen_pool="mean", bert_pool="cls", reduce_factor=4,
                 qwen_dropout=0.1, bert_dropout=0.1,
                 qwen_layers=1, bert_layers=1,
                 qwen_dtype=torch.bfloat16, lora_trainable=True):
        super().__init__()
        self.qwen_layers = int(qwen_layers)     # avg last-k hidden layers (1 = last only)
        self.bert_layers = int(bert_layers)

        # --- Branch A: Qwen + LoRA ---
        self.qwen = AutoModel.from_pretrained(str(qwen_name), **{_DTYPE_KW: qwen_dtype})
        if lora_cfg is not None:
            self.qwen = get_peft_model(self.qwen, lora_cfg)
        elif qwen_adapter_dir is not None:
            self.qwen = PeftModel.from_pretrained(
                self.qwen, str(qwen_adapter_dir), is_trainable=lora_trainable)
        self.qwen_hidden = self.qwen.config.hidden_size
        self.qwen_pool = qwen_pool
        self.reduce_factor = int(reduce_factor)
        if self.qwen_hidden % self.reduce_factor != 0:
            raise ValueError(
                f"reduce_factor {self.reduce_factor} does not divide Qwen hidden "
                f"size {self.qwen_hidden}")
        self.qwen_dim = self.qwen_hidden // self.reduce_factor

        # --- Branch B: BERT (full fine-tune) ---
        # add_pooling_layer=False: we pool CLS from last_hidden_state ourselves, so
        # the BERT pooler would be an unused parameter and would break DDP with
        # find_unused_parameters=False.
        try:
            self.bert = AutoModel.from_pretrained(str(bert_name), add_pooling_layer=False)
        except TypeError:  # encoder without that kwarg
            self.bert = AutoModel.from_pretrained(str(bert_name))
        self.bert_hidden = self.bert.config.hidden_size
        self.bert_pool = bert_pool

        self.num_labels = int(num_labels)
        self.drop_qwen = nn.Dropout(qwen_dropout)
        self.drop_bert = nn.Dropout(bert_dropout)

        self.head_qwen = nn.Linear(self.qwen_dim, self.num_labels)
        self.head_bert = nn.Linear(self.bert_hidden, self.num_labels)
        for h in (self.head_qwen, self.head_bert):
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    # ---- pooling helpers ----
    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    @staticmethod
    def _multi_layer_hidden(out, k):
        """Average the last k hidden-state layers. ``out.hidden_states`` is a tuple
        of (num_layers + 1) tensors [embeddings, layer1, ..., layerL]."""
        if k <= 1:
            return out.last_hidden_state
        return torch.stack(out.hidden_states[-k:], dim=0).mean(dim=0)

    # ---- encoders ----
    def encode_qwen(self, input_ids, attention_mask):
        need_hs = self.qwen_layers > 1
        out = self.qwen(input_ids=input_ids, attention_mask=attention_mask,
                        return_dict=True, output_hidden_states=need_hs)
        lh = self._multi_layer_hidden(out, self.qwen_layers)
        if self.qwen_pool == "mean":
            pooled = self._mean_pool(lh, attention_mask)
        else:  # last non-pad token
            seq_lens = attention_mask.sum(dim=1) - 1
            idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, lh.size(-1))
            pooled = lh.gather(1, idx).squeeze(1)
        pooled = pooled.to(torch.float32)
        if self.reduce_factor > 1:  # parameter-free mean-of-N reduce
            pooled = pooled.view(pooled.shape[0], self.qwen_dim, self.reduce_factor).mean(dim=-1)
        return pooled  # (B, qwen_dim)

    def encode_bert(self, input_ids, attention_mask, token_type_ids=None):
        need_hs = self.bert_layers > 1
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True,
                        output_hidden_states=need_hs)
        lh = self._multi_layer_hidden(out, self.bert_layers)
        pooled = lh[:, 0] if self.bert_pool == "cls" else self._mean_pool(lh, attention_mask)
        return pooled.to(torch.float32)  # (B, bert_hidden)

    def forward(self, q_input_ids, q_attention_mask,
                b_input_ids, b_attention_mask, b_token_type_ids=None):
        qvec = self.encode_qwen(q_input_ids, q_attention_mask)
        bvec = self.encode_bert(b_input_ids, b_attention_mask, b_token_type_ids)
        logits_q = self.head_qwen(self.drop_qwen(qvec))
        logits_b = self.head_bert(self.drop_bert(bvec))
        return logits_q, logits_b


def unwrap(model):
    """Strip a DDP wrapper if present."""
    return model.module if hasattr(model, "module") else model


def build_lora_config(cfg):
    return LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        target_modules=list(cfg["lora_targets"]), lora_dropout=cfg["lora_dropout"],
        bias="none", task_type=None,
    )


def build_model(cfg, num_labels, lora_cfg=None):
    """Fresh model for training."""
    return QwenBertDual(
        cfg["qwen_name"], cfg["bert_name"], num_labels,
        lora_cfg=build_lora_config(cfg) if lora_cfg is None else lora_cfg,
        qwen_pool=cfg["qwen_pool"], bert_pool=cfg["bert_pool"],
        reduce_factor=cfg["reduce_factor"],
        qwen_dropout=cfg["qwen_dropout"], bert_dropout=cfg["bert_dropout"],
        qwen_layers=cfg["qwen_layers"], bert_layers=cfg["bert_layers"],
    )


# ---------------------------------------------------------------- evaluation


def _ens_key(alpha, single):
    return "ens" if single else f"ens@{alpha:g}"


@torch.no_grad()
def evaluate(model, loader, device, alphas=0.5, topk=(1, 3, 5, 10),
             use_amp=True, reduce_ddp=False):
    """Precision@k for the Qwen branch, the BERT branch, and the ensemble.

    ``alphas`` may be a float or a list of floats; every alpha is scored in the
    same single pass over the data. Returns
    ``{"ens": {k: P@k}, "qwen": {...}, "bert": {...}}`` for a single alpha, or
    ``"ens@<alpha>"`` keys when several are given.

    With ``reduce_ddp=True`` the hit counts are all_reduced, so every rank may
    iterate its own shard of the eval set (see data.build_eval_loader).

    NOTE: all ranks must call this together -- a DDP forward pass can still
    broadcast module buffers, so running it on rank 0 only will deadlock.
    """
    if isinstance(alphas, (int, float)):
        alphas, single = [float(alphas)], True
    else:
        alphas, single = [float(a) for a in alphas], False
    topk = tuple(sorted({int(k) for k in topk}))
    names = [_ens_key(a, single) for a in alphas] + ["qwen", "bert"]

    was_training = model.training
    model.eval()
    amp_on = bool(use_amp) and device.type == "cuda"
    counts = torch.zeros(len(names), len(topk), dtype=torch.float64, device=device)
    total = torch.zeros((), dtype=torch.float64, device=device)

    for batch in loader:
        kw = batch_to_device(batch, device)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
            logits_q, logits_b = model(**kw)
        logits_q = logits_q.float()
        logits_b = logits_b.float()
        scored = [a * logits_q + (1.0 - a) * logits_b for a in alphas] + [logits_q, logits_b]
        kmax = min(max(topk), logits_q.size(-1))
        for i, logits in enumerate(scored):
            idx = logits.topk(kmax, dim=-1).indices
            hits = y.gather(1, idx)
            for j, k in enumerate(topk):
                counts[i, j] += hits[:, :min(k, kmax)].sum().double() / k
        total += y.size(0)

    if reduce_ddp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
        dist.all_reduce(total)

    if was_training:
        model.train()
    n = max(total.item(), 1.0)
    return {names[i]: {k: 100.0 * counts[i, j].item() / n for j, k in enumerate(topk)}
            for i in range(len(names))}


def format_metrics(tag, metrics, topk=(1, 3, 5, 10)):
    """One line per branch, e.g. ``step 500 | ENS  P@1=90.41 P@3=...``."""
    lines = []
    width = max(len(n) for n in metrics)
    for name, m in metrics.items():
        cells = " ".join(f"P@{k}={m[k]:.2f}" for k in topk if k in m)
        lines.append(f"{tag} | {name.upper():<{width}} {cells}")
    return "\n".join(lines)


# ---------------------------------------------------------------- checkpoint I/O

QWEN_ADAPTER_DIR = "qwen_lora_adapter"
BERT_DIR = "bert_encoder"
QWEN_TOK_DIR = "qwen_tokenizer"
BERT_TOK_DIR = "bert_tokenizer"
HEAD_QWEN_FILE = "head_qwen.pt"
HEAD_BERT_FILE = "head_bert.pt"
ARCH_FILE = "config.json"
METRICS_FILE = "best_metrics.json"


def save_checkpoint(model, qtok, btok, out_dir, cfg, num_labels,
                    metrics=None, step=None, alpha=None):
    """Write everything test.py needs to rebuild and run this model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = unwrap(model)

    raw.qwen.save_pretrained(str(out_dir / QWEN_ADAPTER_DIR))   # LoRA adapter only
    raw.bert.save_pretrained(str(out_dir / BERT_DIR))           # full fine-tuned encoder
    torch.save(raw.head_qwen.state_dict(), out_dir / HEAD_QWEN_FILE)
    torch.save(raw.head_bert.state_dict(), out_dir / HEAD_BERT_FILE)
    if qtok is not None:
        qtok.save_pretrained(str(out_dir / QWEN_TOK_DIR))
    if btok is not None:
        btok.save_pretrained(str(out_dir / BERT_TOK_DIR))

    (out_dir / ARCH_FILE).write_text(
        json.dumps(arch_config(cfg, num_labels), indent=2), encoding="utf-8")

    if metrics is not None:
        meta = {
            "step": step,
            "ensemble_alpha": alpha if alpha is not None else cfg["ensemble_alpha"],
            "metrics": {n: {f"P@{k}": v for k, v in m.items()} for n, m in metrics.items()},
        }
        (out_dir / METRICS_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_arch(ckpt_dir):
    path = Path(ckpt_dir) / ARCH_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Checkpoints written by older versions of this script "
            f"did not record the architecture; re-save it or pass the arch flags by hand.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(ckpt_dir, device="cuda", qwen_name=None, merge_lora=False,
                    qwen_dtype=torch.bfloat16):
    """Rebuild a trained model from disk.

    Returns ``(model, qwen_tokenizer, bert_tokenizer, arch)``. The model is in
    eval mode with grads disabled.
    """
    ckpt = Path(ckpt_dir)
    arch = load_arch(ckpt)
    device = torch.device(device)

    bert_src = ckpt / BERT_DIR if (ckpt / BERT_DIR).exists() else arch["bert_name"]
    qwen_src = qwen_name or arch["qwen_name"]
    adapter = ckpt / QWEN_ADAPTER_DIR
    if not adapter.exists():
        raise FileNotFoundError(f"missing LoRA adapter dir: {adapter}")

    model = QwenBertDual(
        qwen_src, bert_src, arch["num_labels"],
        lora_cfg=None, qwen_adapter_dir=adapter,
        qwen_pool=arch["qwen_pool"], bert_pool=arch["bert_pool"],
        reduce_factor=arch["reduce_factor"],
        qwen_dropout=arch["qwen_dropout"], bert_dropout=arch["bert_dropout"],
        qwen_layers=arch["qwen_layers"], bert_layers=arch["bert_layers"],
        qwen_dtype=qwen_dtype, lora_trainable=False,
    )
    model.head_qwen.load_state_dict(torch.load(ckpt / HEAD_QWEN_FILE, map_location="cpu"))
    model.head_bert.load_state_dict(torch.load(ckpt / HEAD_BERT_FILE, map_location="cpu"))

    if merge_lora and hasattr(model.qwen, "merge_and_unload"):
        model.qwen = model.qwen.merge_and_unload()

    model.to(device).eval()
    model.requires_grad_(False)

    qtok_src = ckpt / QWEN_TOK_DIR if (ckpt / QWEN_TOK_DIR).exists() else qwen_src
    btok_src = ckpt / BERT_TOK_DIR if (ckpt / BERT_TOK_DIR).exists() else arch["bert_name"]
    qtok, btok = load_tokenizers(qtok_src, btok_src)
    return model, qtok, btok, arch
