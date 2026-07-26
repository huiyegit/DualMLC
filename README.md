# DualMLC

Dual-encoder co-training for extreme multi-label text classification (XMC).

Two live, jointly-trained branches, each with its own classifier head and its own
BCE loss. There is no coupling term — the branches only meet at prediction time,
where their logits are blended:

```
                    ┌────────────────────────────────────────┐
  document ──┬─────▶│ Qwen2.5-7B (LoRA)  →  mean-pool        │──▶ head_qwen ──▶ logits_A
             │      │                    →  mean-of-N reduce │
             │      └────────────────────────────────────────┘
             │      ┌────────────────────────────────────────┐
             └─────▶│ bert-base-uncased (full FT) → CLS pool │──▶ head_bert ──▶ logits_B
                    └────────────────────────────────────────┘

  loss  = BCE(logits_A, y) + BCE(logits_B, y)
  score = α · logits_A + (1 − α) · logits_B
```

Each document is tokenized twice, once per encoder, since the two branches use
different vocabularies. Only the LoRA adapters, the BERT encoder, and the two
heads are trained; the 7B base weights stay frozen in bf16.

## Results

Wiki10-31K, best configuration (α = 0.5):

| Model          | P@1  | P@3 | P@5 |
|----------------|------|-----|-----|
| Ensemble       | 90.4 | —   | —   |
| Qwen branch    | —    | —   | —   |
| BERT branch    | —    | —   | —   |

Fill in from `models/<run>/best_metrics.json` after a run.

## Setup

These results were produced on 8 × NVIDIA GeForce RTX 4090 (24 GB each, compute
capability sm_89) with CUDA 11.8.

```bash
conda create -n dualmlc python=3.8 -y
conda activate dualmlc
pip install -r requirements.txt
```

## Data

Expects the PECOS `xmc-base` layout:

```
xmc-base/wiki10-31k/
  X.trn.txt      # one raw document per line
  Y.trn.npz      # scipy sparse CSR, shape (n_docs, n_labels)
  X.tst.txt
  Y.tst.npz
```

## Training

Reference run (8 GPUs):

```bash
torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --max-steps 4000 --lr 5e-5 --head-lr 5e-4 --bert-lr 1e-4 --bert-head-lr 2e-3 \
    --ensemble-alpha 0.5 --model-dir models/wiki10_qwen_bert_dual
```

Single GPU:

```bash
python train.py --max-steps 4000 --model-dir models/wiki10_qwen_bert_dual
```

The checkpoint with the best ensemble P@1 is written to `--model-dir` as training
proceeds. Add `--save-last` to also write a resumable `last_state.pt` (weights +
optimizer + scheduler) every `--save-steps`, and continue an interrupted run with
`--resume auto`.

Both branches are evaluated separately at every eval step alongside the ensemble,
so you can see which one is carrying the result.

## Inference

Evaluate a checkpoint:

```bash
python test.py --ckpt models/wiki10_qwen_bert_dual --data-dir xmc-base/wiki10-31k
```

Find the best ensemble weight without retraining — every α is scored in a single
pass over the data:

```bash
python test.py --ckpt models/wiki10_qwen_bert_dual --data-dir xmc-base/wiki10-31k \
    --sweep-alpha 0 0.25 0.5 0.75 1.0
```

Predict labels for raw documents:

```bash
python test.py --ckpt models/wiki10_qwen_bert_dual --no-eval \
    --input-file docs.txt --predict-topk 10 --out preds.jsonl \
    --label-file xmc-base/wiki10-31k/output-items.txt
```

## Layout

| File        | Contents                                                                 |
|-------------|--------------------------------------------------------------------------|
| `config.py` | Defaults, the arch keys persisted with each checkpoint, both CLI parsers |
| `data.py`   | Dual-tokenizer datasets, collation, dataloaders                          |
| `model.py`  | `QwenBertDual`, `evaluate()`, checkpoint save/load                       |
| `train.py`  | DDP setup, training loop, resume                                         |
| `test.py`   | Checkpoint evaluation, α sweep, prediction                               |

A saved checkpoint contains:

```
qwen_lora_adapter/   LoRA adapter only (the frozen base is not copied)
bert_encoder/        full fine-tuned BERT
head_qwen.pt         head_bert.pt
qwen_tokenizer/      bert_tokenizer/
config.json          architecture + num_labels, so test.py can rebuild it
best_metrics.json    P@k at the step this checkpoint was saved
```

## Notes on attribution

The license in this repository covers this code only. It does not relicense the
pretrained models or the dataset:

- **Qwen2.5-7B** — see the model card for its terms.
- **bert-base-uncased** — Apache 2.0.
- **Wiki10-31K** — from the
  [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html);
  link to it rather than re-hosting the preprocessed data.

A LoRA adapter trained on Qwen2.5-7B is a derivative of those weights and carries
their terms, independent of this repository's license.