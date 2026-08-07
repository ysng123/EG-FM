# PixelDiT C2I + EG-FM

## Installation

Run from `example/pixeldit/`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../../egfm
```

The EG-FM settings are under `model.diffusion_trainer.init_args` in each C2I config:

```yaml
egfm_sigma0: 3.5
egfm_curve: smootherstep
egfm_release_start: 0.0
egfm_release_end: 1.0
```

## Training

Run from `example/pixeldit/c2i/`.

ImageNet 256×256:

```bash
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl.yaml
```

ImageNet 512×512:

```bash
bash train_c2i.sh --num-gpus 8 --config configs/pix512_xl.yaml
```

Resume training:

```bash
bash train_c2i.sh \
  --num-gpus 8 \
  --config configs/pix256_xl.yaml \
  --ckpt-path /path/to/checkpoint.ckpt
```

## Inference

```bash
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

Generated images and `output.npz` are written below `train_logs/`.
