# DualMLC

Official implementation of: **LLM-Enhanced Dual-Branch Learning for Multi-Label Text Classification**.

Paper: arXiv link coming soon.  
<!-- Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) -->


## Model Zoo

| Dataset | P@1 | P@3 | P@5 | Training Cost | Checkpoint | Log |
|---|---:|---:|---:|---|---|---|
| EUR-Lex-4K | 88.82 | 75.80 | 62.74 | 0.45 h (8 × RTX 4090) | [Weights](https://drive.google.com/drive/folders/17UoVMKI6G4sm-9cib9kjghfMmSGBDNUp?usp=drive_link) | [Log](https://drive.google.com/file/d/1MRVv9sjIFRfK9TS4j-kax51ZVESt2QB1/view?usp=drive_link) |
| Wiki10-31K | 90.78 | 81.11 | 71.94 | 0.25 h (8 × RTX 4090) | [Weights](https://drive.google.com/drive/folders/1REzsEq83Y_V2SS5f0btfweI_biTdTuDi?usp=sharing) | [Log](https://drive.google.com/file/d/1N107fEG10HPIK0Cn1Ys5KZGFGWA2Fmwj/view?usp=drive_link) |
| AmazonCat-13K | 96.81 | 84.31 | 69.11 | 23.25 h (8 × RTX 4090) | [Weights](https://drive.google.com/drive/folders/1YxLiT0XHKWYS8nSpxF65zcQU61jhu7S_?usp=drive_link) | [Log](https://drive.google.com/file/d/1PvzFA1Mo08_MS-L1Yfpb6FxcxO4VV6U0/view?usp=drive_link) |

## Setup

Results were obtained using 8 × RTX 4090 GPUs with CUDA 11.8.

```bash
conda create -n dualmlc python=3.8 -y
conda activate dualmlc
pip install -r requirements.txt
```

## Pretrained Models

Download Qwen2.5-7B and BERT:

```bash
pip install -U huggingface_hub
mkdir -p models

hf download Qwen/Qwen2.5-7B \
    --local-dir models/Qwen2.5-7B

hf download google-bert/bert-base-uncased \
    --local-dir models/bert-base-uncased
```


## Data Preparation

Download `eurlex-4k`, `wiki10-31k`, and `amazoncat-13k` from [MatchXML](https://github.com/huiyegit/MatchXML) or [XR-Transformer](https://github.com/amzn/pecos/tree/mainline/examples/xr-transformer-neurips21).

Place the datasets under `xmc-base`:

```text
xmc-base/wiki10-31k/
├── X.trn.txt
├── Y.trn.npz
├── X.tst.txt
└── Y.tst.npz
```

## Training

Train DualMLC on Wiki10-31K using eight GPUs:

```bash
mkdir -p logs

torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --data-dir xmc-base/wiki10-31k \
    --model-dir models/wiki10_dualmlc \
    --qwen-name models/Qwen2.5-7B \
    --bert-name models/bert-base-uncased \
    --max-steps 4000 \
    --lr 5e-5 \
    --bert-lr 1e-4 \
    --head-lr 5e-4 \
    --bert-head-lr 2e-3 \
    --qwen-layers 4 \
    --bert-layers 4 \
    --ensemble-alpha 0.6 \
    --qwen-dropout 0.3 \
    --bert-dropout 0.1 \
    2>&1 | tee logs/wiki10-31k-dualmlc.log
```

Train DualMLC on EUR-Lex-4K using eight GPUs:

```bash
torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --data-dir xmc-base/eurlex-4k \
    --model-dir models/eurlex_dualmlc \
    --qwen-name models/Qwen2.5-7B \
    --bert-name models/bert-base-uncased \
    --max-steps 8000 \
    --lr 5e-5 \
    --bert-lr 1e-4 \
    --head-lr 1e-3 \
    --bert-head-lr 2e-3 \
    --qwen-layers 1 \
    --bert-layers 1 \
    --qwen-dropout 0.4 \
    --bert-dropout 0.4 \
    --ensemble-alpha 0.6 \
    2>&1 | tee logs/eurlex-4k-dualmlc.log
```


Train DualMLC on AmazonCat-13K using eight GPUs:

```bash
torchrun --nproc_per_node=8 --master_port=29515 train.py \
    --data-dir xmc-base/amazoncat-13k \
    --model-dir models/amazoncat_dualmlc \
    --qwen-name models/Qwen2.5-7B \
    --bert-name models/bert-base-uncased \
    --max-steps 400000 \
    --logging-steps 500 \
    --eval-steps 50000 \
    --save-steps 100000 \
    --lr 5e-5 \
    --bert-lr 1e-4 \
    --head-lr 1e-3 \
    --bert-head-lr 2e-3 \
    --qwen-layers 4 \
    --bert-layers 4 \
    --ensemble-alpha 0.6 \
    --qwen-dropout 0.3 \
    --bert-dropout 0.1 \
    2>&1 | tee logs/amazoncat-13k-dualmlc.log
```

## Evaluation

Multiple GPUs:

```bash
torchrun --nproc_per_node=8 --master_port=29516 test.py \
    --ckpt models/wiki10_dualmlc \
    --data-dir xmc-base/wiki10-31k
```

Single GPU:

```bash
python test.py \
    --ckpt models/wiki10_dualmlc \
    --data-dir xmc-base/wiki10-31k
```

## License

The repository license covers the DualMLC code only.
