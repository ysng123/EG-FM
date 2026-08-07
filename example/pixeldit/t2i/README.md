# PixelDiT T2I + EG-FM

## Installation

Run from `example/pixeldit/`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../../egfm
```

The EG-FM settings are under `scheduler` in each T2I config:

```yaml
egfm_sigma0: 3.5
egfm_curve: smootherstep
egfm_release_start: 0.0
egfm_release_end: 1.0
```

## Training

Run from `example/pixeldit/t2i/`.

Stage 1, 512×512:

```bash
bash train.sh configs/PixelDiT_512px_pixel_diffusion_stage1.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/stage1-output \
  --name=pixeldit-t2i-512-stage1 \
  --tracker_project_name=pixeldit_t2i \
  --train.save_model_steps=10000
```

Stage 2, 512×512:

```bash
bash train.sh configs/PixelDiT_512px_pixel_diffusion_stage2.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/stage2-output \
  --name=pixeldit-t2i-512-stage2 \
  --load_from=/path/to/stage1-checkpoint.pth
```

Stage 3, 1024×1024:

```bash
bash train.sh configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/stage3-output \
  --name=pixeldit-t2i-1024-stage3 \
  --load_from=/path/to/stage2-checkpoint.pth
```

Resume training with optimizer and scheduler state:

```bash
bash train.sh configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --resume_from=/path/to/checkpoint.pth \
  --work_dir=/path/to/output \
  --name=pixeldit-t2i-1024-stage3
```

## Inference

```bash
python inference.py \
  --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --model_path /path/to/pixeldit_t2i.pth \
  --txt_file prompts.txt \
  --custom_height 1024 \
  --custom_width 1024 \
  --cfg_scale 2.75 \
  --seed 2025 \
  --work_dir /path/to/output
```

Generated images are written below `<work_dir>/vis/`.
