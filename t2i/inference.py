#!/usr/bin/env python3
"""Generate images from prompts with a PixelDiT-T2I checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

from checkpoint import checkpoint_metadata, extract_model_state, load_checkpoint_file  # noqa: E402
from t2i.models import GemmaTextEncoder, T2IDenoiser  # noqa: E402
from t2i.train import STAGE_DEFAULTS, apply_stage_defaults, get_args_parser  # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt_file", type=Path)
    parser.add_argument("--samples_per_prompt", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "samples/t2i")
    parser.add_argument("--weights", choices=("ema1", "ema2", "model"), default="ema1")
    parser.add_argument("--text_encoder_path", default=None)
    parser.add_argument("--allow_text_encoder_download", action="store_true")
    parser.add_argument("--img_size", default=None, type=int)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--cfg", default=None, type=float)
    parser.add_argument("--interval_min", default=None, type=float)
    parser.add_argument("--interval_max", default=None, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_mmap", action="store_true")
    return parser


def _namespace_from_checkpoint(checkpoint):
    defaults = {
        action.dest: action.default
        for action in get_args_parser()._actions
        if action.dest != "help"
    }
    saved = checkpoint.get("args", {})
    if hasattr(saved, "__dict__"):
        saved = vars(saved)
    if not isinstance(saved, dict):
        raise TypeError("checkpoint['args'] must be a dict or argparse Namespace")
    defaults.update(saved)
    defaults.setdefault("stage", "pretrain512")
    if defaults["stage"] not in STAGE_DEFAULTS:
        defaults["stage"] = "pretrain512"
    return apply_stage_defaults(argparse.Namespace(**defaults))


def _read_prompts(args):
    prompts = list(args.prompt)
    if args.prompt_file:
        prompts.extend(
            line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not prompts:
        raise ValueError("Provide at least one --prompt or --prompt_file")
    if args.samples_per_prompt < 1 or args.batch_size < 1:
        raise ValueError("samples_per_prompt and batch_size must be positive")
    return [prompt for prompt in prompts for _ in range(args.samples_per_prompt)]


def _save_image(sample, path):
    image = sample.detach().float().add(1).div(2).clamp(0, 1)
    array = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def main():
    cli = get_parser().parse_args()
    checkpoint_path = cli.checkpoint.expanduser()
    checkpoint = load_checkpoint_file(checkpoint_path, mmap=not cli.no_mmap)
    backend = checkpoint.get("backend", "pixeldit_t2i")
    if backend != "pixeldit_t2i":
        raise ValueError(f"Expected pixeldit_t2i checkpoint, found {backend!r}")
    settings = _namespace_from_checkpoint(checkpoint)
    for source, destination in (
        (cli.img_size, "img_size"),
        (cli.steps, "num_sampling_steps"),
        (cli.cfg, "cfg"),
        (cli.interval_min, "interval_min"),
        (cli.interval_max, "interval_max"),
    ):
        if source is not None:
            setattr(settings, destination, source)
    if cli.text_encoder_path is not None:
        settings.text_encoder_path = cli.text_encoder_path
    if not settings.text_encoder_path:
        raise ValueError("Text encoder path is absent from the checkpoint; pass --text_encoder_path")
    settings.feat_loss_weight = 0.0
    if not 0 <= settings.interval_min < settings.interval_max <= 1:
        raise ValueError("CFG interval must satisfy 0 <= min < max <= 1")

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if cli.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("Use --precision float32 for CPU inference")
    dtype = torch.bfloat16 if cli.precision == "bfloat16" else torch.float32
    prompts = _read_prompts(cli)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    model = T2IDenoiser(settings)
    state = extract_model_state(checkpoint, which=cli.weights)
    model.net.load_state_dict(state)
    metadata = checkpoint_metadata(checkpoint)
    del state, checkpoint
    model = model.to(device=device, dtype=dtype).eval()
    if cli.compile:
        model.net = torch.compile(model.net, mode="max-autotune")
    encoder = GemmaTextEncoder(
        settings.text_encoder_path,
        max_length=settings.text_max_length,
        use_chi_prompt=settings.use_chi_prompt,
        local_files_only=not cli.allow_text_encoder_download,
    )
    generator = torch.Generator(device=device).manual_seed(cli.seed)
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if dtype == torch.bfloat16
        else nullcontext()
    )

    image_index = 0
    with torch.inference_mode(), autocast:
        for start in range(0, len(prompts), cli.batch_size):
            batch_prompts = prompts[start:start + cli.batch_size]
            text, text_mask = encoder.encode(batch_prompts, device)
            null_text, null_mask = encoder.encode_null(len(batch_prompts), device)
            samples = model.generate(
                text.to(dtype), null_text.to(dtype), text_mask, null_mask, generator
            )
            for sample in samples:
                _save_image(sample, cli.output_dir / f"{image_index:05d}.png")
                image_index += 1

    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metadata": metadata,
        "weights": cli.weights,
        "prompts": prompts,
        "seed": cli.seed,
        "resolution": settings.img_size,
        "sampler": "FlowDPM",
        "steps": settings.num_sampling_steps,
        "cfg": settings.cfg,
        "cfg_interval": [settings.interval_min, settings.interval_max],
        "flow_shift": settings.flow_shift,
        "prediction_target": "sigma_velocity",
    }
    (cli.output_dir / "sampling.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {image_index} image(s) and sampling.json to {cli.output_dir}")


if __name__ == "__main__":
    main()
