#!/usr/bin/env python
"""
train.py -- dual-encoder co-training.
"""
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
from data import (DualLabeledDataset, build_eval_loader, build_train_loader,
                  batch_to_device, load_tokenizers, split_paths)
from model import build_model, build_lora_config, evaluate, format_metrics, save_checkpoint

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("train")

LAST_STATE_FILE = "last_state.pt"


# ---------------------------------------------------------------- DDP helpers


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


def ddp_barrier(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def ddp_cleanup(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def rank0_log(rank, msg):
    if is_main(rank):
        LOG.info(msg)


# ---------------------------------------------------------------- resume state


def save_last_state(path, raw, optimizer, scheduler, step, epoch, best_p1):
    """Only trainable tensors are stored -- the frozen 7B base would be ~15 GB."""
    payload = {
        "step": step,
        "epoch": epoch,
        "best_p1": best_p1,
        "trainable": {n: p.detach().cpu()
                      for n, p in raw.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_last_state(path, raw, optimizer, scheduler):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    missing, unexpected = raw.load_state_dict(state["trainable"], strict=False)
    if unexpected:
        LOG.warning("resume: %d unexpected keys (first: %s)", len(unexpected), unexpected[0])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"]), int(state["epoch"]), float(state["best_p1"])


# ---------------------------------------------------------------- train


def train(cfg, resume=None, save_last=False):
    rank, local_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    rank0_log(rank, f"world_size: {world_size}  device: {device}")
    rank0_log(rank, f"per-GPU micro-batch: {cfg['batch_size']}  grad_accum: {cfg['grad_accum']}"
                    f"  EFFECTIVE BATCH: "
                    f"{cfg['batch_size'] * cfg['grad_accum'] * world_size}")
    rank0_log(rank, f"ensemble_alpha: {cfg['ensemble_alpha']}  "
                    f"qwen_dropout: {cfg['qwen_dropout']}  bert_dropout: {cfg['bert_dropout']}")

    # ---- data ----
    qtok, btok = load_tokenizers(cfg["qwen_name"], cfg["bert_name"])
    rank0_log(rank, "loading datasets...")
    rank0_log(rank, f"qwen_max {cfg['qwen_max_length']}, bert_max {cfg['bert_max_length']}")
    trn = DualLabeledDataset(*split_paths(cfg["data_dir"], "trn"), qtok, btok,
                             cfg["qwen_max_length"], cfg["bert_max_length"])
    tst = DualLabeledDataset(*split_paths(cfg["data_dir"], "tst"), qtok, btok,
                             cfg["qwen_max_length"], cfg["bert_max_length"])
    rank0_log(rank, f"train: {len(trn)} docs, test: {len(tst)} docs, "
                    f"num_labels: {trn.num_labels}")
    trn_loader, trn_sampler = build_train_loader(trn, cfg, rank, world_size)
    tst_loader = build_eval_loader(tst, cfg, rank, world_size)

    # ---- model ----
    rank0_log(rank, "constructing model...")
    torch.manual_seed(cfg["seed"])          # identical init on every rank
    np.random.seed(cfg["seed"])
    model = build_model(cfg, trn.num_labels, lora_cfg=build_lora_config(cfg)).to(device)
    if cfg["per_rank_dropout"]:
        # Independent dropout masks per rank (conventional). Note this lowers
        # gradient variance vs. sharing one mask across ranks, which measurably
        # shifts final P@1 on Wiki10-31K -- keep it off to match the reference runs.
        torch.manual_seed(cfg["seed"] + 1000 * rank)
        np.random.seed(cfg["seed"] + 1000 * rank)

    if world_size > 1:
        # broadcast_buffers=False: every buffer here (rotary inv_freq, BERT
        # position_ids) is deterministic and identical on all ranks, so syncing
        # them buys nothing and turns any rank-asymmetric forward pass into a
        # deadlock.
        ddp_kw = {"find_unused_parameters": False, "broadcast_buffers": False}
        if device.type == "cuda":
            ddp_kw.update(device_ids=[local_rank], output_device=local_rank)
        model = DDP(model, **ddp_kw)
    raw = model.module if world_size > 1 else model
    rank0_log(rank, "model on device")

    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    total = sum(p.numel() for p in raw.parameters())
    rank0_log(rank, f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # ---- optimizer: four groups, four learning rates ----
    qwen_lora = [p for n, p in raw.qwen.named_parameters() if "lora_" in n and p.requires_grad]
    if not qwen_lora:
        raise RuntimeError("no trainable LoRA parameters found on the Qwen branch")
    groups = [
        {"params": qwen_lora, "lr": cfg["lr"]},
        {"params": list(raw.bert.parameters()), "lr": cfg["bert_lr"]},
        {"params": list(raw.head_qwen.parameters()), "lr": cfg["head_lr"]},
        {"params": list(raw.head_bert.parameters()), "lr": cfg["bert_head_lr"]},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, cfg["warmup_steps"], cfg["max_steps"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    trainable_params = [p for p in raw.parameters() if p.requires_grad]
    alpha = cfg["ensemble_alpha"]
    amp_on = bool(cfg["use_amp"]) and device.type == "cuda"
    # No GradScaler: it exists for fp16, and this pipeline is bf16 (Qwen weights
    # are loaded in bf16), which has fp32's exponent range and needs no scaling.

    out_dir = Path(cfg["model_dir"])
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    step, epoch, best_p1 = 0, 0, -1.0
    if resume:
        path = out_dir / LAST_STATE_FILE if resume == "auto" else Path(resume)
        if path.exists():
            step, epoch, best_p1 = load_last_state(path, raw, optimizer, scheduler)
            rank0_log(rank, f"resumed from {path} at step {step} (best ENS P@1 {best_p1:.2f})")
        elif resume != "auto":
            raise FileNotFoundError(path)
        else:
            rank0_log(rank, f"--resume auto: no {path}, starting fresh")

    micro_step, loss_sum, loss_q_sum, loss_b_sum, loss_n = 0, 0.0, 0.0, 0.0, 0
    t0 = time.time()

    rank0_log(rank, "starting training")
    optimizer.zero_grad(set_to_none=True)
    while step < cfg["max_steps"]:
        if trn_sampler is not None:
            trn_sampler.set_epoch(epoch)
        epoch += 1
        for batch in trn_loader:
            if step >= cfg["max_steps"]:
                break
            model.train()
            kw = batch_to_device(batch, device)
            y = batch["labels"].to(device, non_blocking=True)

            is_boundary = ((micro_step + 1) % cfg["grad_accum"] == 0)
            sync_ctx = nullcontext() if (world_size == 1 or is_boundary) else model.no_sync()
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp_on):
                    logits_q, logits_b = model(**kw)
                loss_q = loss_fn(logits_q.float(), y)
                loss_b = loss_fn(logits_b.float(), y)
                loss = loss_q + loss_b
                (loss / cfg["grad_accum"]).backward()

            loss_sum += loss.item()
            loss_q_sum += loss_q.item()
            loss_b_sum += loss_b.item()
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
                n = max(loss_n, 1)
                rank0_log(rank,
                          f"step {step}/{cfg['max_steps']} | loss {loss_sum / n:.4f} "
                          f"| q {loss_q_sum / n:.4f} b {loss_b_sum / n:.4f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e} "
                          f"| elapsed {(time.time() - t0) / 60:.1f}m")
                loss_sum = loss_q_sum = loss_b_sum = 0.0
                loss_n = 0

            if step % cfg["eval_steps"] == 0 or step == cfg["max_steps"]:
                rank0_log(rank, f"--- eval at step {step} ---")
                # every rank participates; counts are all_reduced inside
                m = evaluate(model, tst_loader, device, alpha, cfg["topk"],
                             use_amp=cfg["use_amp"], reduce_ddp=world_size > 1)
                if is_main(rank):
                    LOG.info("\n%s", format_metrics(f"step {step}", m, cfg["topk"]))
                    if m["ens"][1] > best_p1:
                        best_p1 = m["ens"][1]
                        LOG.info("new best ENS P@1=%.2f, saving to %s", best_p1, out_dir)
                        save_checkpoint(raw, qtok, btok, out_dir, cfg, trn.num_labels,
                                        metrics=m, step=step, alpha=alpha)
                ddp_barrier(world_size)

            if save_last and is_main(rank) and step % cfg["save_steps"] == 0:
                save_last_state(out_dir / LAST_STATE_FILE, raw, optimizer, scheduler,
                                step, epoch, best_p1)
                LOG.info("wrote resumable state at step %d", step)

    rank0_log(rank, "=== final eval ===")
    m = evaluate(model, tst_loader, device, alpha, cfg["topk"],
                 use_amp=cfg["use_amp"], reduce_ddp=world_size > 1)
    if is_main(rank):
        LOG.info("\n%s", format_metrics("FINAL", m, cfg["topk"]))
        LOG.info("best ENS P@1: %.2f  (checkpoint: %s)", best_p1, out_dir)
        if save_last:
            save_last_state(out_dir / LAST_STATE_FILE, raw, optimizer, scheduler,
                            step, epoch, best_p1)
    ddp_barrier(world_size)
    ddp_cleanup(world_size)
    return best_p1


def main():
    args = build_train_parser().parse_args()
    cfg = cfg_from_args(args)
    train(cfg, resume=args.resume, save_last=args.save_last)


if __name__ == "__main__":
    main()
