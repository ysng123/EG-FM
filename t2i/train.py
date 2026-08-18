#!/usr/bin/env python3
"""Train PixelDiT-T2I with Energy-Guided Flow Matching."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

from checkpoint import load_checkpoint_file  # noqa: E402
from t2i.data import (  # noqa: E402
    BLIP3oHFDataset,
    BLIP3oTarDataset,
    load_blip3o_hf_webdataset,
    t2i_collate,
)
from t2i.models import GemmaTextEncoder, T2IDenoiser  # noqa: E402
from t2i.training import CAME, train_steps  # noqa: E402
from utils import misc  # noqa: E402


STAGE_DEFAULTS = {
    "pretrain256": dict(img_size=256, batch_size=128, max_train_steps=200_000,
                        gradient_accumulation_steps=1, feat_loss_weight=0.5),
    "pretrain512": dict(img_size=512, batch_size=48, max_train_steps=100_000,
                        gradient_accumulation_steps=1, feat_loss_weight=0.0),
    "sft512": dict(img_size=512, batch_size=48, max_train_steps=100_000,
                   gradient_accumulation_steps=1, feat_loss_weight=0.0),
    "sft1024": dict(img_size=1024, batch_size=8, max_train_steps=100_000,
                    gradient_accumulation_steps=1, feat_loss_weight=0.0),
}


def get_args_parser():
    parser = argparse.ArgumentParser("PixelDiT-T2I + EG-FM")
    data = parser.add_argument_group("data and stages")
    data.add_argument("--stage", default="pretrain256", choices=tuple(STAGE_DEFAULTS))
    data.add_argument("--data_path", required=True)
    data.add_argument("--dataset_backend", default="auto", choices=("auto", "tar", "hf"))
    data.add_argument("--data_cache_dir", default="./data_cache/blip3o")
    data.add_argument("--data_num_proc", default=64, type=int)
    data.add_argument("--max_shards", default=0, type=int)
    data.add_argument("--img_size", default=None, type=int)
    data.add_argument("--batch_size", default=None, type=int)
    data.add_argument("--num_workers", default=8, type=int)
    data.add_argument("--random_crop", action="store_true")
    data.add_argument("--no_random_flip", action="store_false", dest="random_flip")
    data.add_argument("--pin_mem", action="store_true")
    data.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    data.set_defaults(random_flip=True, pin_mem=True)

    text = parser.add_argument_group("text encoder")
    text.add_argument("--text_encoder_path", required=True)
    text.add_argument("--text_dim", default=2304, type=int)
    text.add_argument("--text_max_length", default=300, type=int)
    text.add_argument("--use_chi_prompt", action="store_true")
    text.add_argument("--no_chi_prompt", action="store_false", dest="use_chi_prompt")
    text.add_argument("--allow_text_encoder_download", action="store_true")
    text.set_defaults(use_chi_prompt=True)

    model = parser.add_argument_group("PixelDiT-T2I")
    model.add_argument("--patch_size", default=16, type=int)
    model.add_argument("--num_groups", default=24, type=int)
    model.add_argument("--hidden_size", default=1536, type=int)
    model.add_argument("--pixel_hidden_size", default=16, type=int)
    model.add_argument("--pixel_attn_hidden_size", default=1152, type=int)
    model.add_argument("--pixel_num_groups", default=16, type=int)
    model.add_argument("--patch_depth", default=14, type=int)
    model.add_argument("--pixel_depth", default=2, type=int)
    model.add_argument("--num_text_blocks", default=4, type=int)
    model.add_argument("--use_text_rope", action="store_true")
    model.add_argument("--no_text_rope", action="store_false", dest="use_text_rope")
    model.add_argument("--text_rope_theta", default=10_000.0, type=float)
    model.add_argument("--pit_adaln_post_modulation", action="store_true")
    model.set_defaults(use_text_rope=True)

    objective = parser.add_argument_group("EG-FM objective")
    objective.add_argument("--noise_scale", default=1.0, type=float)
    objective.add_argument("--flow_shift", default=3.0, type=float)
    objective.add_argument("--train_sampling_steps", default=1000, type=int)
    objective.add_argument("--text_drop_prob", default=0.1, type=float)
    objective.add_argument("--freq_sigma_release", action="store_true")
    objective.add_argument("--no_freq_sigma_release", action="store_false", dest="freq_sigma_release")
    objective.add_argument("--freq_ab_sigma0", default=3.5, type=float)
    objective.add_argument("--freq_ab_curve", default="smootherstep",
                           choices=("smootherstep", "smoothstep", "cosine", "linear"))
    objective.add_argument("--freq_release_start_time", default=0.0, type=float)
    objective.add_argument("--freq_release_time", default=1.0, type=float)
    objective.add_argument("--feat_loss_weight", default=None, type=float)
    objective.add_argument("--repa_encoder", default="dinov2_vitb14")
    objective.add_argument("--repa_encoder_index", default=6, type=int)
    objective.add_argument("--torch_hub_dir", default="")
    objective.set_defaults(freq_sigma_release=True)

    training = parser.add_argument_group("training")
    training.add_argument("--max_train_steps", default=None, type=int)
    training.add_argument("--gradient_accumulation_steps", default=None, type=int)
    training.add_argument("--gradient_clip", default=0.2, type=float)
    training.add_argument("--lr", default=1e-4, type=float)
    training.add_argument("--min_lr", default=1e-6, type=float)
    training.add_argument("--lr_schedule", default="constant", choices=("constant", "cosine"))
    training.add_argument("--warmup_steps", default=2000, type=int)
    training.add_argument("--weight_decay", default=0.0, type=float)
    training.add_argument("--optimizer", default="came", choices=("came", "adamw"))
    training.add_argument("--opt_beta1", default=0.9, type=float)
    training.add_argument("--opt_beta2", default=0.999, type=float)
    training.add_argument("--opt_beta3", default=0.9999, type=float)
    training.add_argument("--opt_eps1", default=1e-30, type=float)
    training.add_argument("--opt_eps2", default=1e-16, type=float)
    training.add_argument("--came_clip_threshold", default=1.0, type=float)
    training.add_argument("--ema_decay1", default=0.9999, type=float)
    training.add_argument("--ema_decay2", default=0.9996, type=float)
    training.add_argument("--seed", default=1, type=int)
    training.add_argument("--device", default="cuda")

    sampling = parser.add_argument_group("validation sampling")
    sampling.add_argument("--num_sampling_steps", default=25, type=int)
    sampling.add_argument("--cfg", default=4.0, type=float)
    sampling.add_argument("--interval_min", default=0.0, type=float)
    sampling.add_argument("--interval_max", default=1.0, type=float)
    sampling.add_argument("--eval_steps", default=0, type=int)
    sampling.add_argument("--validation_prompts", nargs="*", default=[
        "A small dog running through a field of flowers.",
        "An astronaut in a jungle, detailed, muted colors, 8k.",
    ])

    io = parser.add_argument_group("output and checkpoints")
    io.add_argument("--output_dir", default="./outputs/t2i")
    io.add_argument("--resume", default="", help="Directory containing checkpoint-last.pth")
    io.add_argument("--resume_checkpoint", default="")
    io.add_argument("--load_from", default="", help="Initialize model weights from a checkpoint")
    io.add_argument("--no_resume_optimizer", action="store_true")
    io.add_argument("--save_steps", default=10_000, type=int)
    io.add_argument("--keep_checkpoint_steps", default=50_000, type=int)
    io.add_argument("--log_freq", default=20, type=int)

    compile_group = parser.add_argument_group("torch.compile")
    compile_group.add_argument("--compile", action="store_true")
    compile_group.add_argument("--compile_backend", default="inductor")
    compile_group.add_argument("--compile_mode", default="max-autotune")
    compile_group.add_argument("--compile_fullgraph", action="store_true")
    compile_group.add_argument("--compile_dynamic", action="store_true")
    compile_group.add_argument("--compile_cache_size_limit", default=128, type=int)
    compile_group.add_argument("--compile_optimize_ddp", action="store_true")

    distributed = parser.add_argument_group("distributed")
    distributed.add_argument("--world_size", default=1, type=int)
    distributed.add_argument("--local_rank", default=-1, type=int)
    distributed.add_argument("--dist_on_itp", action="store_true")
    distributed.add_argument("--dist_url", default="env://")
    distributed.add_argument("--ddp_find_unused_parameters", action="store_true")
    distributed.add_argument("--no_ddp_find_unused_parameters", action="store_false", dest="ddp_find_unused_parameters")
    distributed.set_defaults(ddp_find_unused_parameters=True)
    return parser


def apply_stage_defaults(args):
    for name, value in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.dataset_backend == "auto":
        args.dataset_backend = "hf" if args.stage.startswith("sft") else "tar"
    return args


def _load_dataset(args):
    if args.dataset_backend == "tar":
        return BLIP3oTarDataset(
            args.data_path, args.img_size, args.random_crop, args.random_flip,
            repeat=True, shuffle_shards=True, max_shards=args.max_shards
        ), None
    dataset, shards = load_blip3o_hf_webdataset(
        args.data_path, args.data_cache_dir, args.data_num_proc, args.max_shards
    )
    return BLIP3oHFDataset(
        dataset, args.img_size, args.random_crop, args.random_flip
    ), shards


def _weights_from_checkpoint(checkpoint):
    return checkpoint.get("model", checkpoint)


def main(args):
    args = apply_stage_defaults(args)
    misc.init_distributed_mode(args)
    args.effective_total_batch_size = (
        args.batch_size * misc.get_world_size() * args.gradient_accumulation_steps
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.torch_hub_dir:
        torch.hub.set_dir(args.torch_hub_dir)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit
    torch._dynamo.config.optimize_ddp = args.compile_optimize_ddp

    log_writer = None
    if misc.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(args.output_dir)
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    dataset, shards = _load_dataset(args)
    print(f"Dataset backend: {args.dataset_backend}; shards: {len(shards) if shards else len(dataset.tar_files)}")
    if isinstance(dataset, BLIP3oHFDataset):
        sampler = DistributedSampler(
            dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(),
            shuffle=True, seed=args.seed, drop_last=True
        ) if args.distributed else RandomSampler(dataset)
    else:
        sampler = None
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=t2i_collate, persistent_workers=args.num_workers > 0
    )

    text_encoder = GemmaTextEncoder(
        args.text_encoder_path, max_length=args.text_max_length,
        use_chi_prompt=args.use_chi_prompt,
        local_files_only=not args.allow_text_encoder_download,
    )
    model_without_ddp = T2IDenoiser(args).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1e6:.3f}M")
    if args.optimizer == "came":
        optimizer = CAME(
            model_without_ddp.parameters(), lr=args.lr,
            betas=(args.opt_beta1, args.opt_beta2, args.opt_beta3),
            eps=(args.opt_eps1, args.opt_eps2),
            clip_threshold=args.came_clip_threshold, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(), lr=args.lr,
            betas=(args.opt_beta1, args.opt_beta2), eps=args.opt_eps2,
            weight_decay=args.weight_decay
        )

    checkpoint_path = args.resume_checkpoint or (
        os.path.join(args.resume, "checkpoint-last.pth") if args.resume else ""
    )
    start_step = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = load_checkpoint_file(checkpoint_path)
        model_without_ddp.load_training_checkpoint(checkpoint, device)
        if "optimizer" in checkpoint and not args.no_resume_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("global_step", checkpoint.get("epoch", -1))) + 1
        print(f"Resumed {checkpoint_path} at step {start_step}")
    else:
        if args.resume_checkpoint:
            raise FileNotFoundError(args.resume_checkpoint)
        if args.load_from:
            checkpoint = load_checkpoint_file(args.load_from)
            model_without_ddp.net.load_state_dict(_weights_from_checkpoint(checkpoint))
            if "repa_proj" in checkpoint:
                model_without_ddp.repa_proj.load_state_dict(checkpoint["repa_proj"])
            print("Initialized weights from", args.load_from)
        model_without_ddp.init_ema_from_current()

    if args.compile:
        model_without_ddp.net.compile(
            backend=args.compile_backend, mode=args.compile_mode,
            fullgraph=args.compile_fullgraph, dynamic=args.compile_dynamic
        )
    model = model_without_ddp
    if args.distributed:
        device_ids = [args.gpu] if device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp, device_ids=device_ids,
            find_unused_parameters=args.ddp_find_unused_parameters
        )

    started = time.time()
    train_steps(
        model, model_without_ddp, text_encoder, loader, optimizer,
        device, start_step, log_writer, args
    )
    print("Training time:", datetime.timedelta(seconds=int(time.time() - started)))
    if log_writer is not None:
        log_writer.flush()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
