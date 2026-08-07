# PixelDiT + EG-FM

This directory contains both class-to-image (`c2i/`) and text-to-image (`t2i/`) training and inference. Run each command block from `example/pixeldit/`.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../../egfm
```

EG-FM is configured with the following YAML keys:

```yaml
egfm_sigma0: 3.5
egfm_curve: smootherstep
egfm_release_start: 0.0
egfm_release_end: 1.0
```

## C2I

Train on eight GPUs:

```bash
cd c2i
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl.yaml
```

Resume from a checkpoint:

```bash
cd c2i
bash train_c2i.sh \
  --num-gpus 8 \
  --config configs/pix256_xl.yaml \
  --ckpt-path /path/to/checkpoint.ckpt
```

Generate ImageNet samples:

```bash
cd c2i
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=/path/to/checkpoint.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=2.75 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=0.9 \
  --per_run_seed=false \
  --seed_everything=1600
```

Generated images and `output.npz` are written below `c2i/train_logs/`. See `c2i/README.md` for dataset preparation and other resolutions.

## T2I

Train the 512×512 first stage:

```bash
cd t2i
bash train.sh configs/PixelDiT_512px_pixel_diffusion_stage1.yaml \
  --data.data_dir="[/path/to/dataset1, /path/to/dataset2]" \
  --work_dir=/path/to/output \
  --name=pixeldit-t2i-512-stage1 \
  --tracker_project_name=pixeldit_t2i \
  --train.save_model_steps=10000
```

For later stages, use `PixelDiT_512px_pixel_diffusion_stage2.yaml` and `PixelDiT_1024px_pixel_diffusion_stage3.yaml`, passing the previous checkpoint through `--load_from`.

Generate images from text prompts:

```bash
cd t2i
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

Generated images are written below `<work_dir>/vis/`. See `t2i/README.md` for dataset fields and multi-stage training details.
