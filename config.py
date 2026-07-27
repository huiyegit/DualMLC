#!/usr/bin/env python
"""
config.py -- single source of truth for defaults and CLI parsing.

CONFIG holds every tunable used by train.py / test.py.

ARCH_KEYS is the subset that must be written next to a checkpoint (as
``config.json``) so that test.py can rebuild the exact same architecture without
the caller having to remember which flags the training run used. The original
single-file script saved only step/alpha/metrics, which made a saved checkpoint
un-loadable without guessing.
"""
import argparse

CONFIG = {
    "data_dir": "xmc-base/wiki10-31k",
    "model_dir": "models/wiki10_dualmlc",

    # ---- Qwen (branch A) ----
    "qwen_name": "models/Qwen2.5-7B",
    "qwen_max_length": 256,
    "qwen_pool": "mean",             # mean | last
    "reduce_factor": 4,              # parameter-free mean-of-N reduce on pooled vec
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # ---- BERT (branch B) ----
    "bert_name": "models/bert-base-uncased",
    "bert_max_length": 256,
    "bert_pool": "cls",              # cls | mean

    "qwen_dropout": 0.3,             # dropout on Qwen pooled vec before its head
    "bert_dropout": 0.1,             # dropout on BERT pooled vec before its head
    "qwen_layers": 4,                # avg last-k hidden-state layers
    "bert_layers": 4,

    # ---- ensemble ----
    "ensemble_alpha": 0.6,           # weight on Qwen logits at eval; (1-alpha) on BERT

    # ---- optimisation ----
    "batch_size": 2,                 # per-GPU micro-batch
    "grad_accum": 1,
    "lr": 5e-5,                      # Qwen LoRA
    "bert_lr": 1e-4,                 # BERT full fine-tune
    "head_lr": 5e-4,                 # Qwen classifier head
    "bert_head_lr": 2e-3,            # BERT classifier head (needs a higher LR)
    "weight_decay": 0.01,
    "warmup_steps": 250,
    "max_steps": 4000,
    "max_grad_norm": 1.0,

    # ---- cadence ----
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000,              # cadence for the resumable last_state.pt (--save-last)

    # ---- eval ----
    "eval_batch_size": 32,
    "topk": (1, 3, 5, 10),

    # ---- misc ----
    "seed": 42,
    "num_workers": 4,
    "use_amp": True,
    # Independent dropout masks per rank. The reference runs shared one mask
    # across ranks; enabling this shifts final P@1 by ~0.15 on Wiki10-31K.
    "per_rank_dropout": False,
}

# Everything test.py needs in order to rebuild the model that produced a checkpoint.
ARCH_KEYS = (
    "qwen_name", "bert_name",
    "qwen_max_length", "bert_max_length",
    "qwen_pool", "bert_pool",
    "reduce_factor", "qwen_layers", "bert_layers",
    "qwen_dropout", "bert_dropout",
    "lora_r", "lora_alpha", "lora_dropout", "lora_targets",
    "ensemble_alpha",
)


def arch_config(cfg, num_labels):
    """The dict persisted as ``<ckpt>/config.json``."""
    out = {}
    for k in ARCH_KEYS:
        v = cfg[k]
        out[k] = list(v) if isinstance(v, tuple) else v
    out["num_labels"] = int(num_labels)
    return out


# ---------------------------------------------------------------- CLI


def build_train_parser():
    p = argparse.ArgumentParser(
        description="Dual-encoder (Qwen2.5 LoRA + BERT) co-training for extreme "
                    "multi-label classification.")
    # paths
    p.add_argument("--data-dir", default=CONFIG["data_dir"])
    p.add_argument("--model-dir", default=CONFIG["model_dir"])
    p.add_argument("--qwen-name", default=CONFIG["qwen_name"])
    p.add_argument("--bert-name", default=CONFIG["bert_name"])
    # schedule
    p.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    p.add_argument("--warmup-steps", type=int, default=CONFIG["warmup_steps"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--grad-accum", type=int, default=CONFIG["grad_accum"])
    # learning rates
    p.add_argument("--lr", type=float, default=CONFIG["lr"], help="Qwen LoRA lr")
    p.add_argument("--bert-lr", type=float, default=CONFIG["bert_lr"], help="BERT full-FT lr")
    p.add_argument("--head-lr", type=float, default=CONFIG["head_lr"], help="Qwen head lr")
    p.add_argument("--bert-head-lr", type=float, default=CONFIG["bert_head_lr"],
                   help="BERT head lr (separate from Qwen head; needs ~2e-3)")
    p.add_argument("--weight-decay", type=float, default=CONFIG["weight_decay"])
    p.add_argument("--max-grad-norm", type=float, default=CONFIG["max_grad_norm"])
    # architecture
    p.add_argument("--ensemble-alpha", type=float, default=CONFIG["ensemble_alpha"],
                   help="eval ensemble weight on Qwen logits; (1-alpha) on BERT")
    p.add_argument("--reduce-factor", type=int, default=CONFIG["reduce_factor"])
    p.add_argument("--lora-r", type=int, default=CONFIG["lora_r"])
    p.add_argument("--lora-alpha", type=int, default=CONFIG["lora_alpha"])
    p.add_argument("--lora-dropout", type=float, default=CONFIG["lora_dropout"])
    p.add_argument("--lora-targets", nargs="+", default=CONFIG["lora_targets"])
    p.add_argument("--qwen-max-length", type=int, default=CONFIG["qwen_max_length"])
    p.add_argument("--bert-max-length", type=int, default=CONFIG["bert_max_length"])
    p.add_argument("--qwen-pool", choices=["mean", "last"], default=CONFIG["qwen_pool"])
    p.add_argument("--bert-pool", choices=["cls", "mean"], default=CONFIG["bert_pool"])
    p.add_argument("--qwen-layers", type=int, default=CONFIG["qwen_layers"],
                   help="avg last-k hidden layers for the Qwen repr")
    p.add_argument("--bert-layers", type=int, default=CONFIG["bert_layers"],
                   help="avg last-k hidden layers for the BERT repr")
    p.add_argument("--qwen-dropout", type=float, default=CONFIG["qwen_dropout"])
    p.add_argument("--bert-dropout", type=float, default=CONFIG["bert_dropout"])
    # cadence / eval
    p.add_argument("--logging-steps", type=int, default=CONFIG["logging_steps"])
    p.add_argument("--eval-steps", type=int, default=CONFIG["eval_steps"])
    p.add_argument("--save-steps", type=int, default=CONFIG["save_steps"])
    p.add_argument("--eval-batch-size", type=int, default=CONFIG["eval_batch_size"])
    p.add_argument("--topk", nargs="+", type=int, default=None)
    # misc
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--per-rank-dropout", action="store_true",
                   help="use independent dropout masks per rank (off by default, to "
                        "match the reference runs)")
    p.add_argument("--save-last", action="store_true",
                   help="also write a resumable last_state.pt (weights + optimizer + "
                        "scheduler) every --save-steps")
    p.add_argument("--resume", default=None,
                   help="path to a last_state.pt (or 'auto' to look in --model-dir)")
    return p


def build_test_parser():
    """Defaults are None on purpose: only explicitly-passed flags override the
    architecture recorded in the checkpoint's config.json."""
    p = argparse.ArgumentParser(
        description="Evaluate a dual-encoder checkpoint, or predict labels for raw text.")
    p.add_argument("--ckpt", required=True, help="checkpoint dir written by train.py")
    p.add_argument("--data-dir", default=None, help="xmc-base dataset dir (for --split eval)")
    p.add_argument("--split", default="tst", help="tst | trn | val ...")
    p.add_argument("--no-eval", action="store_true", help="skip the P@k evaluation")

    p.add_argument("--alpha", dest="ensemble_alpha", type=float, default=None,
                   help="ensemble weight on Qwen logits (default: value from checkpoint)")
    p.add_argument("--sweep-alpha", nargs="+", type=float, default=None,
                   help="also report P@k for each of these alphas (single pass)")
    p.add_argument("--topk", nargs="+", type=int, default=None)
    p.add_argument("--metrics-out", default=None, help="write eval metrics to this json file")

    # prediction
    p.add_argument("--text", nargs="+", default=None, help="one or more raw documents")
    p.add_argument("--input-file", default=None, help="file with one document per line")
    p.add_argument("--predict-topk", type=int, default=10)
    p.add_argument("--out", default=None, help="write predictions here as jsonl")
    p.add_argument("--label-file", default=None,
                   help="optional file with one label name per line, in label-index order")

    # runtime
    p.add_argument("--qwen-name", default=None, help="override the base Qwen path/repo")
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cuda:1 | cpu (default: auto)")
    p.add_argument("--merge-lora", action="store_true",
                   help="merge LoRA into the base weights for faster inference "
                        "(tiny numeric drift)")
    p.add_argument("--no-amp", action="store_true")
    return p


def cfg_from_args(args, base=None):
    """Overlay a parsed namespace onto a config dict.

    Any namespace attribute whose name matches a CONFIG key overrides it, unless
    the value is None (which is how the test parser signals "not specified").
    """
    cfg = dict(CONFIG if base is None else base)
    for key, val in vars(args).items():
        if val is None:
            continue
        if key in cfg:
            cfg[key] = val
    if getattr(args, "topk", None):
        cfg["topk"] = tuple(sorted({int(k) for k in args.topk}))
    else:
        cfg["topk"] = tuple(cfg["topk"])
    if getattr(args, "no_amp", False):
        cfg["use_amp"] = False
    return cfg