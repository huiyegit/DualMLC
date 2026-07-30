#!/usr/bin/env python
"""Distributed single-encoder training entry point."""
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup

from config import build_train_parser, cfg_from_args
from data import (SingleLabeledDataset, batch_to_device, build_eval_loader,
                  build_train_loader, load_tokenizer, max_length, split_paths)
from model import build_model, evaluate, format_metrics, save_checkpoint

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("train")
LAST_STATE_FILE = "last_state.pt"


def ddp_setup():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def barrier(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def cleanup(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def rank0_log(rank, message):
    if is_main(rank):
        LOG.info(message)


def save_last_state(path, raw, optimizer, scheduler, step, epoch, best_p1):
    payload = {
        "step": step,
        "epoch": epoch,
        "best_p1": best_p1,
        "trainable": {name: param.detach().cpu()
                      for name, param in raw.named_parameters() if param.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_last_state(path, raw, optimizer, scheduler):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    _, unexpected = raw.load_state_dict(state["trainable"], strict=False)
    if unexpected:
        LOG.warning("resume: %d unexpected keys", len(unexpected))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"]), int(state["epoch"]), float(state["best_p1"])


def optimizer_groups(raw, cfg):
    if cfg["encoder"] == "qwen":
        lora = [p for name, p in raw.backbone.named_parameters()
                if "lora_" in name and p.requires_grad]
        if not lora:
            raise RuntimeError("no trainable LoRA parameters found")
        return [
            {"params": lora, "lr": cfg["lr"]},
            {"params": list(raw.head.parameters()), "lr": cfg["head_lr"]},
        ]
    return [
        {"params": list(raw.backbone.parameters()), "lr": cfg["bert_lr"]},
        {"params": list(raw.head.parameters()), "lr": cfg["bert_head_lr"]},
    ]


def train(cfg, resume=None, save_last=False):
    rank, local_rank, world_size = ddp_setup()
    device = torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
    rank0_log(rank, "encoder: %s | world_size: %d | device: %s" %
              (cfg["encoder"], world_size, device))
    rank0_log(rank, "per-GPU batch: %d | grad_accum: %d | effective batch: %d" %
              (cfg["batch_size"], cfg["grad_accum"],
               cfg["batch_size"] * cfg["grad_accum"] * world_size))

    tokenizer = load_tokenizer(cfg)
    max_len = max_length(cfg)
    trn = SingleLabeledDataset(*split_paths(cfg["data_dir"], "trn"),
                               tokenizer, max_len)
    tst = SingleLabeledDataset(*split_paths(cfg["data_dir"], "tst"),
                               tokenizer, max_len)
    rank0_log(rank, "train: %d docs | test: %d docs | labels: %d" %
              (len(trn), len(tst), trn.num_labels))
    trn_loader, trn_sampler = build_train_loader(trn, cfg, rank, world_size)
    tst_loader = build_eval_loader(tst, cfg, rank, world_size)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build_model(cfg, trn.num_labels).to(device)
    if cfg["per_rank_dropout"]:
        torch.manual_seed(cfg["seed"] + 1000 * rank)
        np.random.seed(cfg["seed"] + 1000 * rank)

    if world_size > 1:
        kwargs = {"find_unused_parameters": False, "broadcast_buffers": False}
        if device.type == "cuda":
            kwargs.update(device_ids=[local_rank], output_device=local_rank)
        model = DDP(model, **kwargs)
    raw = model.module if world_size > 1 else model

    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    total = sum(p.numel() for p in raw.parameters())
    rank0_log(rank, "trainable: %s / %s (%.3f%%)" %
              (format(trainable, ","), format(total, ","), 100.0 * trainable / total))

    optimizer = torch.optim.AdamW(
        optimizer_groups(raw, cfg), weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, cfg["warmup_steps"], cfg["max_steps"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    trainable_params = [p for p in raw.parameters() if p.requires_grad]
    amp_on = bool(cfg["use_amp"]) and device.type == "cuda"

    out_dir = Path(cfg["model_dir"])
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    step, epoch, best_p1 = 0, 0, -1.0
    if resume:
        path = out_dir / LAST_STATE_FILE if resume == "auto" else Path(resume)
        if path.exists():
            step, epoch, best_p1 = load_last_state(
                path, raw, optimizer, scheduler)
            rank0_log(rank, "resumed from %s at step %d" % (path, step))
        elif resume != "auto":
            raise FileNotFoundError(path)

    micro_step = 0
    loss_sum = 0.0
    loss_n = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < cfg["max_steps"]:
        if trn_sampler is not None:
            trn_sampler.set_epoch(epoch)
        epoch += 1
        for batch in trn_loader:
            if step >= cfg["max_steps"]:
                break
            model.train()
            kwargs = batch_to_device(batch, device)
            y = batch["labels"].to(device, non_blocking=True)
            is_boundary = ((micro_step + 1) % cfg["grad_accum"] == 0)
            sync_ctx = nullcontext() if (world_size == 1 or is_boundary) else model.no_sync()
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp_on):
                    logits = model(**kwargs)
                loss = loss_fn(logits.float(), y)
                (loss / cfg["grad_accum"]).backward()

            loss_sum += loss.item()
            loss_n += 1
            micro_step += 1
            if micro_step % cfg["grad_accum"] != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg["logging_steps"] == 0:
                rank0_log(rank, "step %d/%d | loss %.4f | lr %.2e | elapsed %.1fm" %
                          (step, cfg["max_steps"], loss_sum / max(loss_n, 1),
                           scheduler.get_last_lr()[0], (time.time() - t0) / 60.0))
                loss_sum = 0.0
                loss_n = 0

            if step % cfg["eval_steps"] == 0 or step == cfg["max_steps"]:
                rank0_log(rank, "--- eval at step %d ---" % step)
                metrics = evaluate(model, tst_loader, device, cfg["topk"],
                                   use_amp=cfg["use_amp"], reduce_ddp=world_size > 1)
                if is_main(rank):
                    LOG.info("\n%s", format_metrics("step %d" % step, metrics, cfg["topk"]))
                    p1 = metrics[cfg["encoder"]][1]
                    if p1 > best_p1:
                        best_p1 = p1
                        LOG.info("new best P@1=%.2f, saving to %s", best_p1, out_dir)
                        save_checkpoint(raw, tokenizer, out_dir, cfg, trn.num_labels,
                                        metrics=metrics, step=step)
                barrier(world_size)

            if save_last and is_main(rank) and step % cfg["save_steps"] == 0:
                save_last_state(out_dir / LAST_STATE_FILE, raw, optimizer,
                                scheduler, step, epoch, best_p1)

    rank0_log(rank, "=== final eval ===")
    metrics = evaluate(model, tst_loader, device, cfg["topk"],
                       use_amp=cfg["use_amp"], reduce_ddp=world_size > 1)
    if is_main(rank):
        LOG.info("\n%s", format_metrics("FINAL", metrics, cfg["topk"]))
        LOG.info("best P@1: %.2f (checkpoint: %s)", best_p1, out_dir)
        if save_last:
            save_last_state(out_dir / LAST_STATE_FILE, raw, optimizer,
                            scheduler, step, epoch, best_p1)
    barrier(world_size)
    cleanup(world_size)
    return best_p1


def main():
    args = build_train_parser().parse_args()
    cfg = cfg_from_args(args)
    train(cfg, resume=args.resume, save_last=args.save_last)


if __name__ == "__main__":
    main()
