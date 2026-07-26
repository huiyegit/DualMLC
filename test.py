#!/usr/bin/env python
"""
test.py -- inference / evaluation entrypoint for a checkpoint written by train.py.

Evaluate the saved model on a split:

  python test.py --ckpt models/wiki10_qwen_bert_dual --data-dir xmc-base/wiki10-31k

Find the best ensemble weight without retraining (single pass over the data):

  python test.py --ckpt ... --data-dir ... --sweep-alpha 0 0.25 0.5 0.75 1.0

Predict labels for raw documents:

  python test.py --ckpt ... --no-eval --text "the quick brown fox ..." --predict-topk 5
  python test.py --ckpt ... --no-eval --input-file docs.txt --out preds.jsonl \
      --label-file xmc-base/wiki10-31k/output-items.txt

Single-process only (no torchrun needed).
"""
import json
import logging
import sys
from pathlib import Path

import torch

from config import CONFIG, build_test_parser, cfg_from_args
from data import (DualLabeledDataset, batch_to_device, build_eval_loader,
                  build_predict_loader, read_texts, split_paths)
from model import evaluate, format_metrics, load_checkpoint

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("test")


def pick_device(explicit=None):
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


def run_eval(model, toks, cfg, arch, device, split, alphas, metrics_out=None):
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
    LOG.info("eval on %s: %d docs, %d labels", split, len(ds), ds.num_labels)

    loader = build_eval_loader(ds, cfg, rank=0, world_size=1)
    metrics = evaluate(model, loader, device, alphas, cfg["topk"],
                       use_amp=cfg["use_amp"], reduce_ddp=False)
    print(format_metrics(f"{split}", metrics, cfg["topk"]))

    if metrics_out:
        payload = {
            "ckpt": str(cfg.get("_ckpt", "")),
            "split": split,
            "num_docs": len(ds),
            "alphas": alphas if isinstance(alphas, list) else [alphas],
            "metrics": {n: {f"P@{k}": v for k, v in m.items()} for n, m in metrics.items()},
        }
        Path(metrics_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    args = build_test_parser().parse_args()
    device = pick_device(args.device)

    LOG.info("loading checkpoint %s onto %s", args.ckpt, device)
    model, qtok, btok, arch = load_checkpoint(
        args.ckpt, device=device, qwen_name=args.qwen_name, merge_lora=args.merge_lora)
    toks = (qtok, btok)

    # checkpoint arch is the base; CLI flags that were actually passed win
    cfg = dict(CONFIG)
    cfg.update({k: v for k, v in arch.items() if k in cfg})
    cfg = cfg_from_args(args, cfg)
    cfg["_ckpt"] = args.ckpt
    alpha = cfg["ensemble_alpha"]
    LOG.info("num_labels %d | qwen_max %d | bert_max %d | alpha %.3f | amp %s",
             arch["num_labels"], cfg["qwen_max_length"], cfg["bert_max_length"],
             alpha, cfg["use_amp"])

    texts = list(args.text) if args.text else []
    if args.input_file:
        texts += read_texts(args.input_file)

    if not args.no_eval:
        alphas = args.sweep_alpha if args.sweep_alpha else alpha
        run_eval(model, toks, cfg, arch, device, args.split, alphas, args.metrics_out)

    if texts:
        run_predict(model, toks, cfg, arch, device, texts, args.predict_topk, alpha,
                    label_file=args.label_file, out=args.out)
    elif args.no_eval:
        LOG.error("nothing to do: --no-eval was set but no --text/--input-file was given")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
