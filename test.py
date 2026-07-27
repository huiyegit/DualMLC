#!/usr/bin/env python
"""
test.py -- inference / evaluation entrypoint for a checkpoint written by train.py.

Single-GPU evaluation:

  python test.py --ckpt models/wiki10_dualmlc --data-dir xmc-base/wiki10-31k

Multi-GPU evaluation:

  torchrun --nproc_per_node=8 --master_port=29516 test.py \
      --ckpt models/wiki10_dualmlc --data-dir xmc-base/wiki10-31k

Find the best ensemble weight without retraining (single pass over the data):

  python test.py --ckpt ... --data-dir ... --sweep-alpha 0 0.25 0.5 0.75 1.0

Predict labels for raw documents (single process only):

  python test.py --ckpt ... --no-eval --text "the quick brown fox ..." --predict-topk 5
  python test.py --ckpt ... --no-eval --input-file docs.txt --out preds.jsonl \
      --label-file xmc-base/wiki10-31k/output-items.txt
"""
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from config import CONFIG, build_test_parser, cfg_from_args
from data import (DualLabeledDataset, batch_to_device, build_eval_loader,
                  build_predict_loader, read_texts, split_paths)
from model import evaluate, format_metrics, load_checkpoint

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("test")


def distributed_setup(cli_local_rank=None):
    """Initialize torch.distributed when launched with torchrun."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", cli_local_rank or 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    return rank, local_rank, world_size


def distributed_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def pick_device(explicit=None, local_rank=0, world_size=1):
    """Choose one local GPU per torchrun process, or honor --device otherwise."""
    if world_size > 1:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_names(path, num_labels):
    names = read_texts(path)
    if len(names) != num_labels:
        LOG.warning("--label-file has %d lines but the model has %d labels; "
                    "indices beyond the file will be reported as raw ints",
                    len(names), num_labels)
    return names


@torch.no_grad()
def predict(model, loader, device, alpha, topk, use_amp=True):
    """Top-k label indices + scores per document, in input order."""
    amp_on = bool(use_amp) and device.type == "cuda"
    out = []
    for batch in loader:
        kw = batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
            logits_q, logits_b = model(**kw)
        logits_q = logits_q.float()
        logits_b = logits_b.float()
        ens = alpha * logits_q + (1.0 - alpha) * logits_b
        k = min(int(topk), ens.size(-1))
        vals, idx = ens.topk(k, dim=-1)
        probs = torch.sigmoid(vals)
        for row in range(idx.size(0)):
            out.append([
                {"label_id": int(i), "logit": float(v), "score": float(p)}
                for i, v, p in zip(idx[row].tolist(), vals[row].tolist(), probs[row].tolist())
            ])
    return out


def run_eval(model, toks, cfg, arch, device, split, alphas,
             metrics_out=None, rank=0, world_size=1):
    qtok, btok = toks
    texts_path, labels_path = split_paths(cfg["data_dir"], split)
    for p in (texts_path, labels_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"{p} not found -- point --data-dir at the xmc-base dataset dir "
                f"(or pass --no-eval to only run prediction)")
    ds = DualLabeledDataset(texts_path, labels_path, qtok, btok,
                            cfg["qwen_max_length"], cfg["bert_max_length"])
    if ds.num_labels != arch["num_labels"]:
        raise ValueError(f"label-space mismatch: checkpoint has {arch['num_labels']} "
                         f"labels, {labels_path} has {ds.num_labels}")

    if rank == 0:
        LOG.info("eval on %s: %d docs, %d labels, %d process(es)",
                 split, len(ds), ds.num_labels, world_size)

    # build_eval_loader partitions the dataset exactly across ranks.
    loader = build_eval_loader(ds, cfg, rank=rank, world_size=world_size)
    metrics = evaluate(model, loader, device, alphas, cfg["topk"],
                       use_amp=cfg["use_amp"], reduce_ddp=world_size > 1)

    if rank == 0:
        print(format_metrics(f"{split}", metrics, cfg["topk"]))

        if metrics_out:
            payload = {
                "ckpt": str(cfg.get("_ckpt", "")),
                "split": split,
                "num_docs": len(ds),
                "world_size": world_size,
                "alphas": alphas if isinstance(alphas, list) else [alphas],
                "metrics": {
                    n: {f"P@{k}": v for k, v in m.items()}
                    for n, m in metrics.items()
                },
            }
            Path(metrics_out).write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
            LOG.info("wrote metrics to %s", metrics_out)

    return metrics


def run_predict(model, toks, cfg, arch, device, texts, topk, alpha,
                label_file=None, out=None):
    qtok, btok = toks
    LOG.info("predicting for %d document(s)", len(texts))
    loader = build_predict_loader(texts, qtok, btok, cfg)
    preds = predict(model, loader, device, alpha, topk, use_amp=cfg["use_amp"])

    names = load_label_names(label_file, arch["num_labels"]) if label_file else None
    if names:
        for row in preds:
            for item in row:
                if item["label_id"] < len(names):
                    item["label"] = names[item["label_id"]]

    if out:
        with open(out, "w", encoding="utf-8") as f:
            for text, row in zip(texts, preds):
                f.write(json.dumps({"text": text[:500], "predictions": row},
                                   ensure_ascii=False) + "\n")
        LOG.info("wrote %d predictions to %s", len(preds), out)
    else:
        for i, (text, row) in enumerate(zip(texts, preds)):
            print(f"\n[{i}] {text[:120]}{'...' if len(text) > 120 else ''}")
            for rank_, item in enumerate(row, 1):
                label = item.get("label", item["label_id"])
                print(f"   {rank_:>2}. {label}  (score {item['score']:.4f})")
    return preds


def main():
    parser = build_test_parser()
    # Accept both spellings for compatibility with torchrun/PyTorch versions.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = parser.parse_args()
    rank, local_rank, world_size = distributed_setup(args.local_rank)

    try:
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

        if world_size > 1 and args.device and rank == 0:
            LOG.warning("--device is ignored under torchrun; each process uses its LOCAL_RANK")

        device = pick_device(args.device, local_rank=local_rank, world_size=world_size)

        if rank == 0:
            LOG.info("loading checkpoint %s on %d process(es)", args.ckpt, world_size)
        model, qtok, btok, arch = load_checkpoint(
            args.ckpt, device=device, qwen_name=args.qwen_name,
            merge_lora=args.merge_lora)
        toks = (qtok, btok)

        # Checkpoint architecture is the base; explicitly passed CLI flags win.
        cfg = dict(CONFIG)
        cfg.update({k: v for k, v in arch.items() if k in cfg})
        cfg = cfg_from_args(args, cfg)
        cfg["_ckpt"] = args.ckpt
        alpha = cfg["ensemble_alpha"]

        if rank == 0:
            LOG.info("num_labels %d | qwen_max %d | bert_max %d | "
                     "alpha %.3f | amp %s",
                     arch["num_labels"], cfg["qwen_max_length"],
                     cfg["bert_max_length"], alpha, cfg["use_amp"])

        texts = list(args.text) if args.text else []
        if args.input_file:
            texts += read_texts(args.input_file)

        if world_size > 1 and texts:
            if rank == 0:
                LOG.error("raw-text prediction is single-process only; "
                          "run it with `python test.py`, not `torchrun`")
            return 2

        if not args.no_eval:
            alphas = args.sweep_alpha if args.sweep_alpha else alpha
            run_eval(model, toks, cfg, arch, device, args.split, alphas,
                     metrics_out=args.metrics_out, rank=rank,
                     world_size=world_size)

        if texts:
            run_predict(model, toks, cfg, arch, device, texts,
                        args.predict_topk, alpha,
                        label_file=args.label_file, out=args.out)
        elif args.no_eval:
            if rank == 0:
                LOG.error("nothing to do: --no-eval was set but no "
                          "--text/--input-file was given")
            return 2

        return 0
    finally:
        distributed_cleanup()


if __name__ == "__main__":
    sys.exit(main())
