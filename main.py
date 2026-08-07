"""Training and distributed evaluation entry point for PixelDiT + EG-FM."""

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from checkpoint import load_checkpoint_file
from denoiser import Denoiser
from engine import evaluate, train_one_epoch
from models import MODEL_REGISTRY
from utils import misc
from utils.crop import center_crop_arr


def get_args_parser():
    parser = argparse.ArgumentParser("PixelDiT + Energy-Guided Flow Matching")

    model = parser.add_argument_group("model and objective")
    model.add_argument("--model", default="PixDiT-XL/16", choices=sorted(MODEL_REGISTRY))
    model.add_argument("--prediction_target", default="velocity", choices=["velocity", "x"])
    model.add_argument("--img_size", default=256, type=int)
    model.add_argument("--class_num", default=1000, type=int)
    model.add_argument("--attn_dropout", default=0.0, type=float)
    model.add_argument("--proj_dropout", default=0.0, type=float)

    path = parser.add_argument_group("EG-FM path")
    path.add_argument("--freq_ab_sigma0", default=3.5, type=float)
    path.add_argument(
        "--freq_ab_curve",
        default="smootherstep",
        choices=["smootherstep", "smoothstep", "cosine", "linear"],
    )
    path.add_argument("--freq_release_start_time", default=0.0, type=float)
    path.add_argument("--freq_release_time", default=1.0, type=float)
    path.add_argument("--t_sampler", default="lognormal", choices=["lognormal", "uniform"])
    path.add_argument("--P_mean", default=0.0, type=float)
    path.add_argument("--P_std", default=1.0, type=float)
    path.add_argument("--timeshift", default=1.0, type=float)
    path.add_argument("--endpoint_low_prob", default=0.0, type=float)
    path.add_argument("--endpoint_high_prob", default=0.0, type=float)
    path.add_argument("--endpoint_width", default=0.1, type=float)
    path.add_argument("--noise_scale", default=1.0, type=float)
    path.add_argument("--t_eps", default=5e-2, type=float)
    path.add_argument("--label_drop_prob", default=0.1, type=float)

    repa = parser.add_argument_group("optional REPA")
    repa.add_argument("--repa_weight", default=0.0, type=float)
    repa.add_argument("--repa_layer", default=8, type=int)
    repa.add_argument("--repa_encoder", default="dinov2_vitb14")
    repa.add_argument("--torch_hub_dir", default="")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", default=200, type=int)
    training.add_argument("--warmup_epochs", default=5, type=int)
    training.add_argument("--batch_size", default=128, type=int)
    training.add_argument("--lr", default=None, type=float)
    training.add_argument("--blr", default=5e-5, type=float)
    training.add_argument("--min_lr", default=0.0, type=float)
    training.add_argument(
        "--lr_schedule",
        default="constant",
        choices=["constant", "cosine", "constant_restart", "cosine_restart", "flat_linear_cosine"],
    )
    training.add_argument("--lr_restart_epoch", default=-1, type=int)
    training.add_argument("--lr_linear_start_epoch", default=160.0, type=float)
    training.add_argument("--lr_linear_end_epoch", default=162.0, type=float)
    training.add_argument("--lr_linear_end_lr", default=2e-5, type=float)
    training.add_argument("--weight_decay", default=0.0, type=float)
    training.add_argument("--grad_clip", default=0.0, type=float)
    training.add_argument("--ema_decay1", default=0.9999, type=float)
    training.add_argument("--ema_decay2", default=0.9996, type=float)
    training.add_argument("--seed", default=0, type=int)
    training.add_argument("--start_epoch", default=0, type=int)
    training.add_argument("--num_workers", default=12, type=int)
    training.add_argument("--pin_mem", action="store_true")
    training.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    training.set_defaults(pin_mem=True)

    sampling = parser.add_argument_group("sampling and evaluation")
    sampling.add_argument("--sampling_method", default="flowdpm", choices=["flowdpm", "heun", "euler"])
    sampling.add_argument("--num_sampling_steps", default=50, type=int)
    sampling.add_argument("--cfg", default=1.0, type=float)
    sampling.add_argument("--interval_min", default=0.0, type=float)
    sampling.add_argument("--interval_max", default=1.0, type=float)
    sampling.add_argument("--num_images", default=50000, type=int)
    sampling.add_argument("--gen_bsz", default=256, type=int)
    sampling.add_argument("--eval_freq", default=40, type=int)
    sampling.add_argument("--online_eval", action="store_true")
    sampling.add_argument("--evaluate_gen", action="store_true")
    sampling.add_argument("--eval_save_root", default="")
    sampling.add_argument("--fid_stats", default="")
    sampling.add_argument("--keep_eval_images", action="store_true")

    io = parser.add_argument_group("data and checkpoints")
    io.add_argument("--data_path", default="./data/imagenet")
    io.add_argument("--output_dir", default="./outputs/default")
    io.add_argument("--resume", default="")
    io.add_argument("--resume_checkpoint", default="")
    io.add_argument("--no_resume_optimizer", action="store_true")
    io.add_argument("--save_last_freq", default=5, type=int)
    io.add_argument("--save_epoch_freq", default=20, type=int)
    io.add_argument("--log_freq", default=100, type=int)
    io.add_argument("--device", default="cuda")

    compile_group = parser.add_argument_group("torch.compile")
    compile_group.add_argument("--compile", action="store_true")
    compile_group.add_argument("--no_compile", action="store_false", dest="compile")
    compile_group.add_argument("--compile_backend", default="inductor")
    compile_group.add_argument("--compile_mode", default="max-autotune")
    compile_group.add_argument("--compile_fullgraph", action="store_true")
    compile_group.add_argument("--compile_dynamic", action="store_true")
    compile_group.add_argument("--compile_cache_size_limit", default=128, type=int)
    compile_group.add_argument("--compile_optimize_ddp", action="store_true")
    compile_group.add_argument("--no_compile_optimize_ddp", action="store_false", dest="compile_optimize_ddp")
    compile_group.set_defaults(compile=False, compile_optimize_ddp=False)

    distributed = parser.add_argument_group("distributed")
    distributed.add_argument("--world_size", default=1, type=int)
    distributed.add_argument("--local_rank", default=-1, type=int)
    distributed.add_argument("--dist_on_itp", action="store_true")
    distributed.add_argument("--dist_url", default="env://")
    return parser


def _checkpoint_path(args):
    return args.resume_checkpoint or (
        os.path.join(args.resume, "checkpoint-last.pth") if args.resume else ""
    )


def main(args):
    misc.init_distributed_mode(args)
    if args.torch_hub_dir:
        torch.hub.set_dir(args.torch_hub_dir)
    if not args.eval_save_root:
        args.eval_save_root = os.path.join(args.output_dir, "eval_samples")

    checkpoint_path = _checkpoint_path(args)
    if args.resume_checkpoint and not os.path.isfile(args.resume_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
    if args.evaluate_gen and not (checkpoint_path and os.path.isfile(checkpoint_path)):
        raise FileNotFoundError("--evaluate_gen requires a valid checkpoint")

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    rank = misc.get_rank()
    world_size = misc.get_world_size()
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    log_writer = None
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)

    data_loader = None
    if not args.evaluate_gen:
        train_dir = os.path.join(args.data_path, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"ImageNet training directory not found: {train_dir}")
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: center_crop_arr(image, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        print(dataset)

    torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit
    torch._dynamo.config.optimize_ddp = args.compile_optimize_ddp
    model_without_ddp = Denoiser(args).to(device)
    trainable = sum(parameter.numel() for parameter in model_without_ddp.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.3f}M")

    effective_batch = args.batch_size * world_size
    if args.lr is None:
        args.lr = args.blr * effective_batch / 256
    parameter_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, betas=(0.9, 0.95))

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = load_checkpoint_file(checkpoint_path, mmap=True)
        model_without_ddp.load_training_checkpoint(checkpoint, device)
        saved_target = checkpoint.get("prediction_target")
        if saved_target and saved_target != args.prediction_target:
            print(f"Warning: changing prediction target from {saved_target} to {args.prediction_target}")
        if not args.evaluate_gen and not args.no_resume_optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        print("Resumed checkpoint from", checkpoint_path)
        del checkpoint
    else:
        model_without_ddp.init_ema_from_current()
        print("Training from scratch")

    if args.compile:
        # Training calls Denoiser.forward, while generation calls the backbone
        # through predict_velocity. Compile the function that is actually on
        # the hot path in each mode.
        compile_target = model_without_ddp.net if args.evaluate_gen else model_without_ddp
        compile_target.compile(
            backend=args.compile_backend,
            mode=args.compile_mode,
            fullgraph=args.compile_fullgraph,
            dynamic=args.compile_dynamic,
        )
    model = (
        torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu])
        if args.distributed
        else model_without_ddp
    )

    if args.evaluate_gen:
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(seed)
            evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    print(f"Start training at epoch {args.start_epoch}, stop before epoch {args.epochs}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer, args
        )
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args, model_without_ddp, optimizer, epoch, epoch_name="last")
        if args.save_epoch_freq > 0 and epoch > 0 and epoch % args.save_epoch_freq == 0:
            misc.save_model(args, model_without_ddp, optimizer, epoch)
        if args.online_eval and epoch > 0 and (
            epoch % args.eval_freq == 0 or epoch + 1 == args.epochs
        ):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, args, epoch, args.gen_bsz, log_writer)
            torch.cuda.empty_cache()
        if log_writer is not None:
            log_writer.flush()
    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("Training time:", elapsed)


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
