#!/usr/bin/env python
"""Configuration and CLI parsing for single-encoder XMC training."""
import argparse

CONFIG = {
    "encoder": "bert",             # bert | qwen
    "data_dir": "xmc-base/wiki10-31k",
    "model_dir": "models/wiki10_bert",

    # Qwen
    "qwen_name": "models/Qwen2.5-7B",
    "qwen_max_length": 256,
    "qwen_pool": "mean",           # mean | last
    "qwen_layers": 4,
    "qwen_dropout": 0.3,
    "reduce_factor": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # BERT
    "bert_name": "models/bert-base-uncased",
    "bert_max_length": 256,
    "bert_pool": "cls",            # cls | mean
    "bert_layers": 4,
    "bert_dropout": 0.1,

    # Optimization. Qwen uses lr/head_lr; BERT uses bert_lr/bert_head_lr.
    "batch_size": 2,
    "grad_accum": 1,
    "lr": 5e-5,
    "head_lr": 5e-4,
    "bert_lr": 1e-4,
    "bert_head_lr": 2e-3,
    "weight_decay": 0.01,
    "warmup_steps": 250,
    "max_steps": 4000,
    "max_grad_norm": 1.0,

    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000,

    "eval_batch_size": 32,
    "topk": (1, 3, 5, 10),

    "seed": 42,
    "num_workers": 4,
    "use_amp": True,
    "per_rank_dropout": False,
}

ARCH_KEYS = (
    "encoder",
    "qwen_name", "qwen_max_length", "qwen_pool", "qwen_layers",
    "qwen_dropout", "reduce_factor", "lora_r", "lora_alpha",
    "lora_dropout", "lora_targets",
    "bert_name", "bert_max_length", "bert_pool", "bert_layers",
    "bert_dropout",
)


def arch_config(cfg, num_labels):
    out = {}
    for key in ARCH_KEYS:
        value = cfg[key]
        out[key] = list(value) if isinstance(value, tuple) else value
    out["num_labels"] = int(num_labels)
    return out


def _add_common_args(p):
    p.add_argument("--data-dir", default=CONFIG["data_dir"])
    p.add_argument("--model-dir", default=CONFIG["model_dir"])
    p.add_argument("--encoder", choices=["bert", "qwen"], default=CONFIG["encoder"])
    p.add_argument("--qwen-name", default=CONFIG["qwen_name"])
    p.add_argument("--bert-name", default=CONFIG["bert_name"])


def build_train_parser():
    p = argparse.ArgumentParser(
        description="Single-encoder BERT or Qwen training for extreme multi-label classification.")
    _add_common_args(p)

    p.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    p.add_argument("--warmup-steps", type=int, default=CONFIG["warmup_steps"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--grad-accum", type=int, default=CONFIG["grad_accum"])

    p.add_argument("--lr", type=float, default=CONFIG["lr"], help="Qwen LoRA learning rate")
    p.add_argument("--head-lr", type=float, default=CONFIG["head_lr"], help="Qwen head learning rate")
    p.add_argument("--bert-lr", type=float, default=CONFIG["bert_lr"], help="BERT encoder learning rate")
    p.add_argument("--bert-head-lr", type=float, default=CONFIG["bert_head_lr"], help="BERT head learning rate")
    p.add_argument("--weight-decay", type=float, default=CONFIG["weight_decay"])
    p.add_argument("--max-grad-norm", type=float, default=CONFIG["max_grad_norm"])

    p.add_argument("--reduce-factor", type=int, default=CONFIG["reduce_factor"])
    p.add_argument("--lora-r", type=int, default=CONFIG["lora_r"])
    p.add_argument("--lora-alpha", type=int, default=CONFIG["lora_alpha"])
    p.add_argument("--lora-dropout", type=float, default=CONFIG["lora_dropout"])
    p.add_argument("--lora-targets", nargs="+", default=CONFIG["lora_targets"])
    p.add_argument("--qwen-max-length", type=int, default=CONFIG["qwen_max_length"])
    p.add_argument("--bert-max-length", type=int, default=CONFIG["bert_max_length"])
    p.add_argument("--qwen-pool", choices=["mean", "last"], default=CONFIG["qwen_pool"])
    p.add_argument("--bert-pool", choices=["cls", "mean"], default=CONFIG["bert_pool"])
    p.add_argument("--qwen-layers", type=int, default=CONFIG["qwen_layers"])
    p.add_argument("--bert-layers", type=int, default=CONFIG["bert_layers"])
    p.add_argument("--qwen-dropout", type=float, default=CONFIG["qwen_dropout"])
    p.add_argument("--bert-dropout", type=float, default=CONFIG["bert_dropout"])

    p.add_argument("--logging-steps", type=int, default=CONFIG["logging_steps"])
    p.add_argument("--eval-steps", type=int, default=CONFIG["eval_steps"])
    p.add_argument("--save-steps", type=int, default=CONFIG["save_steps"])
    p.add_argument("--eval-batch-size", type=int, default=CONFIG["eval_batch_size"])
    p.add_argument("--topk", nargs="+", type=int, default=None)

    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--per-rank-dropout", action="store_true")
    p.add_argument("--save-last", action="store_true")
    p.add_argument("--resume", default=None,
                   help="path to last_state.pt, or 'auto' to use <model-dir>/last_state.pt")
    return p


def build_test_parser():
    p = argparse.ArgumentParser(
        description="Evaluate a single-encoder checkpoint or predict labels for raw text.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="tst")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--topk", nargs="+", type=int, default=None)
    p.add_argument("--metrics-out", default=None)

    p.add_argument("--text", nargs="+", default=None)
    p.add_argument("--input-file", default=None)
    p.add_argument("--predict-topk", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--label-file", default=None)

    p.add_argument("--qwen-name", default=None,
                   help="override the base Qwen path when loading a Qwen checkpoint")
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--merge-lora", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p


def cfg_from_args(args, base=None):
    cfg = dict(CONFIG if base is None else base)
    for key, value in vars(args).items():
        if value is not None and key in cfg:
            cfg[key] = value
    if getattr(args, "topk", None):
        cfg["topk"] = tuple(sorted({int(k) for k in args.topk}))
    else:
        cfg["topk"] = tuple(cfg["topk"])
    if getattr(args, "no_amp", False):
        cfg["use_amp"] = False
    return cfg
