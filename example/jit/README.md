# JiT + EG-FM

Run the commands below from `example/jit/`.

## Installation

```bash
conda env create -f environment.yaml
conda activate jit
python -m pip install -e ../../egfm
```

## Training

The following command trains JiT-B/16 on ImageNet 256×256 with eight GPUs:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_jit.py \
  --model JiT-B/16 \
  --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 \
  --img_size 256 --noise_scale 1.0 \
  --batch_size 128 --blr 5e-5 \
  --epochs 600 --warmup_epochs 5 \
  --gen_bsz 128 --num_images 50000 \
  --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
  --egfm_sigma0 3.5 \
  --egfm_curve smootherstep \
  --egfm_release_start 0.0 \
  --egfm_release_end 1.0 \
  --output_dir /path/to/output \
  --resume /path/to/output \
  --data_path /path/to/imagenet \
  --online_eval
```

For 512×512 training, set `--img_size 512`, choose a `/32` model such as `JiT-B/32`, and set `--noise_scale 2.0`.

## Inference

`--resume` must point to a directory containing `checkpoint-last.pth`. Use the same EG-FM settings as training:

```bash
torchrun --standalone --nproc_per_node=1 \
  main_jit.py \
  --model JiT-B/16 \
  --img_size 256 --noise_scale 1.0 \
  --gen_bsz 64 --num_images 50000 \
  --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
  --egfm_sigma0 3.5 \
  --egfm_curve smootherstep \
  --egfm_release_start 0.0 \
  --egfm_release_end 1.0 \
  --output_dir /path/to/eval-output \
  --resume /path/to/checkpoint-directory \
  --data_path /path/to/imagenet \
  --evaluate_gen
```

Generated images and evaluation results are written below `--output_dir`.
