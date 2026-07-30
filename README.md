# DualMLC Single-Encoder Branch

This branch provides single-encoder baselines for BERT and Qwen. Only the selected encoder is loaded, tokenized, trained, evaluated, and saved.

## Setup

```bash
conda create -n dualmlc python=3.8 -y
conda activate dualmlc
pip install -r requirements.txt
```

## BERT-only training

```bash
mkdir -p logs

torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --encoder bert \
    --data-dir xmc-base/wiki10-31k \
    --model-dir models/wiki10_bert \
    --bert-name models/bert-base-uncased \
    --max-steps 4000 \
    --bert-lr 1e-4 \
    --bert-head-lr 2e-3 \
    --bert-layers 4 \
    --bert-dropout 0.1 \
    2>&1 | tee logs/wiki10-31k-bert.log
```

## Qwen-only training

```bash
mkdir -p logs

torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --encoder qwen \
    --data-dir xmc-base/wiki10-31k \
    --model-dir models/wiki10_qwen \
    --qwen-name models/Qwen2.5-7B \
    --max-steps 4000 \
    --lr 5e-5 \
    --head-lr 5e-4 \
    --qwen-layers 4 \
    --qwen-dropout 0.3 \
    2>&1 | tee logs/wiki10-31k-qwen.log
```

## Evaluation

Multiple GPUs:

```bash
torchrun --nproc_per_node=8 --master_port=29516 test.py \
    --ckpt models/wiki10_bert \
    --data-dir xmc-base/wiki10-31k
```

Single GPU:

```bash
python test.py \
    --ckpt models/wiki10_bert \
    --data-dir xmc-base/wiki10-31k
```
