#!/usr/bin/env python
"""Single-encoder distributed evaluation and single-process prediction."""
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from config import CONFIG, build_test_parser, cfg_from_args
from data import (SingleLabeledDataset, batch_to_device, build_eval_loader,
                  build_predict_loader, max_length, read_texts, split_paths)
from model import evaluate, format_metrics, load_checkpoint

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("test")


def distributed_setup(cli_local_rank=None):
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
    if world_size > 1:
        return torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, device, topk, use_amp=True):
    amp_on = bool(use_amp) and device.type == "cuda"
    rows = []
    for batch in loader:
        kwargs = batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
            logits = model(**kwargs).float()
        k = min(int(topk), logits.size(-1))
        vals, idx = logits.topk(k, dim=-1)
        probs = torch.sigmoid(vals)
        for row in range(idx.size(0)):
            rows.append([
                {"label_id": int(i), "logit": float(v), "score": float(p)}
                for i, v, p in zip(idx[row].tolist(), vals[row].tolist(), probs[row].tolist())
            ])
    return rows


def run_eval(model, tokenizer, cfg, arch, device, split,
             metrics_out=None, rank=0, world_size=1):
    texts_path, labels_path = split_paths(cfg["data_dir"], split)
    for path in (texts_path, labels_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    ds = SingleLabeledDataset(texts_path, labels_path, tokenizer, max_length(cfg))
    if ds.num_labels != arch["num_labels"]:
        raise ValueError("label-space mismatch: checkpoint has %d labels, data has %d" %
                         (arch["num_labels"], ds.num_labels))
    if rank == 0:
        LOG.info("eval on %s: %d docs, %d labels, %d process(es)",
                 split, len(ds), ds.num_labels, world_size)
    loader = build_eval_loader(ds, cfg, rank=rank, world_size=world_size)
    metrics = evaluate(model, loader, device, cfg["topk"],
                       use_amp=cfg["use_amp"], reduce_ddp=world_size > 1)
    if rank == 0:
        print(format_metrics(split, metrics, cfg["topk"]))
        if metrics_out:
            payload = {
                "ckpt": str(cfg.get("_ckpt", "")),
                "split": split,
                "num_docs": len(ds),
                "world_size": world_size,
                "metrics": {name: {"P@%d" % k: value for k, value in vals.items()}
                            for name, vals in metrics.items()},
            }
            Path(metrics_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics


def load_label_names(path, num_labels):
    names = read_texts(path)
    if len(names) != num_labels:
        LOG.warning("label file has %d lines but checkpoint has %d labels",
                    len(names), num_labels)
    return names


def run_predict(model, tokenizer, cfg, arch, device, texts, topk,
                label_file=None, out=None):
    loader = build_predict_loader(texts, tokenizer, cfg)
    preds = predict(model, loader, device, topk, use_amp=cfg["use_amp"])
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
    else:
        for i, (text, row) in enumerate(zip(texts, preds)):
            print("\n[%d] %s" % (i, text[:120]))
            for rank_, item in enumerate(row, 1):
                label = item.get("label", item["label_id"])
                print("   %2d. %s (score %.4f)" % (rank_, label, item["score"]))
    return preds


def main():
    parser = build_test_parser()
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = parser.parse_args()
    rank, local_rank, world_size = distributed_setup(args.local_rank)
    try:
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
        device = pick_device(args.device, local_rank, world_size)
        model, tokenizer, arch = load_checkpoint(
            args.ckpt, device=device, qwen_name=args.qwen_name,
            merge_lora=args.merge_lora)

        cfg = dict(CONFIG)
        cfg.update({key: value for key, value in arch.items() if key in cfg})
        cfg = cfg_from_args(args, cfg)
        cfg["_ckpt"] = args.ckpt

        texts = list(args.text) if args.text else []
        if args.input_file:
            texts += read_texts(args.input_file)
        if world_size > 1 and texts:
            if rank == 0:
                LOG.error("raw-text prediction is single-process only")
            return 2

        if not args.no_eval:
            run_eval(model, tokenizer, cfg, arch, device, args.split,
                     metrics_out=args.metrics_out, rank=rank, world_size=world_size)
        if texts:
            run_predict(model, tokenizer, cfg, arch, device, texts,
                        args.predict_topk, args.label_file, args.out)
        elif args.no_eval:
            if rank == 0:
                LOG.error("nothing to do")
            return 2
        return 0
    finally:
        distributed_cleanup()


if __name__ == "__main__":
    sys.exit(main())
