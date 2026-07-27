# DualMLC

This is the official repository of the paper
**LLM-Enhanced Dual-Branch Learning for Multi-Label Text Classification**.

Paper: arXiv link coming soon.
<!-- Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) -->

## Model Zoo

| Dataset       | P@1   | P@3   | P@5   | Training cost | Checkpoint | Log |
|---------------|-------|-------|-------|---------------|------------|-----|
| eurlex-4k     | 88.82 | 75.90 | 62.91 | TBD           | TBD        | TBD |
| wiki10-31k    | 90.78 | 81.11 | 71.94 | 0.25h(8x4090) | [weight](https://drive.google.com/drive/folders/1REzsEq83Y_V2SS5f0btfweI_biTdTuDi?usp=sharing) | [log](https://drive.google.com/file/d/1N107fEG10HPIK0Cn1Ys5KZGFGWA2Fmwj/view?usp=drive_link) |
| amazoncat-13k | 96.81 | 84.31 | 69.11 | TBD           | TBD        | TBD |

<!-- Replace TBD with e.g.  4h (8x4090)  |  [gdrive](URL)  |  [log](logs/wiki10-31k.log) -->

## Setup

These results were produced on 8 × NVIDIA GeForce RTX 4090 (24 GB each, compute
capability sm_89) with CUDA 11.8.

```bash
conda create -n dualmlc python=3.8 -y
conda activate dualmlc
pip install -r requirements.txt
```

## Prepare Data

Download the XMC datasets `eurlex-4k`, `wiki10-31k`, `amazoncat-13k` from
[MatchXML](https://github.com/huiyegit/MatchXML) or
[XR-Transformer](https://github.com/amzn/pecos/tree/mainline/examples/xr-transformer-neurips21)
and put them in the folder `xmc-base`.

Each dataset directory holds:

```
xmc-base/wiki10-31k/
  X.trn.txt      # one raw document per line
  Y.trn.npz      # scipy sparse CSR, shape (n_docs, n_labels)
  X.tst.txt
  Y.tst.npz
```

## Training

Wiki10-31K, 8 GPUs. Effective batch size is 16 (2 per GPU × 8 GPUs).

```bash
mkdir -p logs
torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --data-dir xmc-base/wiki10-31k --model-dir models/wiki10_dualmlc \
    --max-steps 4000 --lr 5e-5 --head-lr 5e-4 --bert-lr 1e-4 --bert-head-lr 2e-3 \
    --qwen-layers 4 --bert-layers 4 \
    --ensemble-alpha 0.6 --qwen-dropout 0.2 --bert-dropout 0.1 \
    2>&1 | tee logs/wiki10-31k-dualmlc.log
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
python test.py --ckpt models/wiki10_dualmlc --data-dir xmc-base/wiki10-31k
```

Find the best ensemble weight without retraining — every α is scored in a single
pass over the data:

```bash
python test.py --ckpt models/wiki10_dualmlc --data-dir xmc-base/wiki10-31k \
    --sweep-alpha 0 0.25 0.5 0.75 1.0
```

Predict labels for raw documents:

```bash
python test.py --ckpt models/wiki10_dualmlc --no-eval \
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