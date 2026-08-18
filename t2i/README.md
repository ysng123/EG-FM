# PixelDiT-T2I + EG-FM

This directory contains the released text-to-image training and inference path.
The model and stage recipe follow the PixelDiT Gaussian T2I implementation,
while the moving endpoint and exact target come from the repository's public
`egfm` package. Nothing under `example/` is used or modified.

## Setup

Install the base environment and the optional T2I dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r t2i/requirements.txt
```

The text encoder is Gemma-2-2B-IT. By default it is loaded only from the local
path passed through `TEXT_ENCODER_PATH`; add `--allow_text_encoder_download` if
you intentionally want Transformers to fetch it.

## Training

Launch 512 pretraining on eight GPUs:

```bash
DATA_PATH=/path/to/blip3o/pretrain-shards \
TEXT_ENCODER_PATH=/path/to/gemma-2-2b-it \
STAGE=pretrain512 \
NPROC_PER_NODE=8 \
bash t2i/scripts/train.sh
```

For 512 supervised fine-tuning, set `STAGE=sft512` and point `DATA_PATH` at the
BLIP3o-60K WebDataset shards. The default stage recipe matches the reference
launcher:

| Stage | Resolution | Batch/GPU | Steps | REPA |
|---|---:|---:|---:|---:|
| `pretrain256` | 256 | 128 | 200,000 | 0.5 |
| `pretrain512` | 512 | 48 | 100,000 | 0.0 |
| `sft512` | 512 | 48 | 100,000 | 0.0 |
| `sft1024` | 1024 | 8 | 100,000 | 0.0 |

All stages default to CAME, learning rate `1e-4`, 2,000 warmup steps,
gradient clipping `0.2`, `flow_shift=3`, `sigma0=3.5`, smootherstep release,
and text dropout `0.1`. `pretrain*` reads tar shards directly; `sft*` defaults
to the indexed Hugging Face WebDataset backend. Override with
`--dataset_backend tar|hf` when needed.

To initialize a later stage from an earlier checkpoint:

```bash
LOAD_FROM=/path/to/checkpoint-last.pth STAGE=pretrain512 \
DATA_PATH=/path/to/shards TEXT_ENCODER_PATH=/path/to/gemma \
bash t2i/scripts/train.sh
```

An existing `${OUTPUT_DIR}/checkpoint-last.pth` is resumed automatically,
including optimizer state and global step.

## Inference

```bash
TEXT_ENCODER_PATH=/path/to/gemma-2-2b-it \
bash t2i/scripts/infer.sh
```

The launcher defaults to the packaged SFT-512 step-40,000 checkpoint at
`../checkpoint/t2i/512/checkpoint-40000.pth` and its EMA2 weights. This
checkpoint reproduces the reported DPG-Bench score (83.8768, reported as
83.9); the available local GenEval summary is 0.81780 and does not reproduce
the paper's 0.85. Set `CHECKPOINT=/another/checkpoint.pth` or `WEIGHTS=ema1`
to override it. The inference entry point uses 25-step FlowDPM with CFG 4 over
the full interval and writes PNG files plus `sampling.json`. Settings stored
in the checkpoint are used unless explicitly overridden:

```bash
python t2i/inference.py \
  --checkpoint /path/to/checkpoint.pth \
  --prompt "A red panda making tea in a tiny kitchen." \
  --text_encoder_path /path/to/gemma-2-2b-it \
  --steps 25 --cfg 4 --output_dir samples/red-panda
```

## Directory layout

```text
t2i/
├── models/
│   ├── backbone.py       # Joint image/text PixelDiT
│   ├── denoiser.py       # EG-FM objective, dual EMA, FlowDPM
│   └── text_encoder.py   # Frozen Gemma encoder
├── data/
│   └── datasets.py       # Tar and Hugging Face WebDataset backends
├── training/
│   ├── engine.py         # Distributed step-based training loop
│   └── optim.py          # CAME optimizer
├── scripts/              # Shell launchers
├── tests/                # Lightweight unit tests
├── train.py              # Training entry point
└── inference.py          # Prompt-based inference entry point
```
