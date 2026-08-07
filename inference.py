#!/usr/bin/env python3
"""Class-conditional inference from a user-specified checkpoint."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image

from checkpoint import load_inference_weights
from models import build_model
from sampling import Sampler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints/pixeldit600/checkpoint-600.pth"
DEFAULT_CFG = 2.55
DEFAULT_INTERVAL_MIN = 0.11
DEFAULT_INTERVAL_MAX = 0.975
DEFAULT_SEED = 99985


def parse_class_ids(value: str) -> list[int]:
    """Parse a comma-separated list of ImageNet class indices."""

    try:
        labels = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("class IDs must be comma-separated integers") from exc
    if not labels:
        raise argparse.ArgumentTypeError("at least one class ID is required")
    invalid = [label for label in labels if not 0 <= label < 1000]
    if invalid:
        raise argparse.ArgumentTypeError(f"class IDs must be in [0, 999], got {invalid}")
    return labels


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a complete PixelDiT checkpoint (default: checkpoint 600)",
    )
    parser.add_argument("--class-ids", type=parse_class_ids, default=parse_class_ids("207"))
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    parser.add_argument("--interval-min", type=float, default=DEFAULT_INTERVAL_MIN)
    parser.add_argument("--interval-max", type=float, default=DEFAULT_INTERVAL_MAX)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    return parser


def resolve_settings(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    settings = {
        "checkpoint": args.checkpoint,
        "cfg": args.cfg,
        "interval_min": args.interval_min,
        "interval_max": args.interval_max,
        "seed": args.seed,
    }
    if args.samples_per_class < 1 or args.batch_size < 1 or args.steps < 1:
        raise ValueError("samples-per-class, batch-size, and steps must be positive")
    if not 0.0 <= settings["interval_min"] < settings["interval_max"] <= 1.0:
        raise ValueError("CFG interval must satisfy 0 <= min < max <= 1")
    return settings


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().float().add(1.0).div(2.0).clamp(0.0, 1.0)
    array = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def main() -> None:
    args = get_parser().parse_args()
    settings = resolve_settings(args)
    checkpoint_path = Path(settings["checkpoint"]).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass an existing full checkpoint with --checkpoint."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if args.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 inference is supported only with --device cuda")

    labels = [
        class_id
        for class_id in args.class_ids
        for _ in range(args.samples_per_class)
    ]
    output_dir = args.output_dir or PROJECT_DIR / "samples" / "pixeldit600"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building PixelDiT-XL/16 ...")
    model = build_model(
        "PixDiT-XL/16", input_size=256, in_channels=3, num_classes=1000
    ).to(device)
    metadata = load_inference_weights(
        model,
        checkpoint_path,
        which="ema1",
        mmap=not args.no_mmap,
    )
    model.eval()
    if args.compile:
        model = torch.compile(model, mode="max-autotune")

    sampler = Sampler(
        method="flowdpm",
        steps=args.steps,
        cfg=float(settings["cfg"]),
        interval_min=float(settings["interval_min"]),
        interval_max=float(settings["interval_max"]),
        timeshift=1.0,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(settings["seed"]))
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bfloat16"
        else nullcontext()
    )

    image_index = 0
    with torch.inference_mode(), autocast:
        for start in range(0, len(labels), args.batch_size):
            batch_labels = torch.tensor(
                labels[start : start + args.batch_size], dtype=torch.long, device=device
            )
            noise = torch.randn(
                batch_labels.shape[0],
                3,
                256,
                256,
                device=device,
                generator=generator,
            )
            samples = sampler(model, noise, batch_labels, null_label=1000)
            for sample, class_id in zip(samples, batch_labels.tolist()):
                filename = f"{image_index:05d}_class{class_id:04d}.png"
                save_image(sample, output_dir / filename)
                image_index += 1

    run_metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metadata": metadata,
        "weights": "ema1",
        "class_ids": args.class_ids,
        "samples_per_class": args.samples_per_class,
        "seed": int(settings["seed"]),
        "sampler": "flowdpm",
        "steps": args.steps,
        "cfg": float(settings["cfg"]),
        "cfg_interval": [
            float(settings["interval_min"]),
            float(settings["interval_max"]),
        ],
        "resolution": 256,
        "prediction_target": "velocity",
    }
    with (output_dir / "sampling.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)
    print(f"Saved {image_index} image(s) and sampling.json to {output_dir}")


if __name__ == "__main__":
    main()
