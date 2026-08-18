"""Step-based distributed training loop for PixelDiT-T2I."""

import contextlib
import gc
import math
import os
import time

import torch
from torchvision.utils import save_image

from utils import misc


def adjust_learning_rate(optimizer, step, args):
    if step < args.warmup_steps:
        lr = args.lr * (step + 1) / max(1, args.warmup_steps)
    elif args.lr_schedule == "constant":
        lr = args.lr
    elif args.lr_schedule == "cosine":
        progress = (step - args.warmup_steps) / max(1, args.max_train_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported lr schedule: {args.lr_schedule}")
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr


def train_steps(model, model_without_ddp, text_encoder, loader, optimizer,
                device, start_step, log_writer, args):
    model.train()
    sampler = getattr(loader, "sampler", None)
    sampler_epoch = 0
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(sampler_epoch)
    iterator = iter(loader)
    metrics = misc.MetricLogger(delimiter="  ")
    metrics.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metrics.add_meter("grad_norm", misc.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    last_log = time.time()

    for global_step in range(start_step, args.max_train_steps):
        lr = adjust_learning_rate(optimizer, global_step, args)
        optimizer.zero_grad(set_to_none=True)
        loss_sums = {}
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                images, captions, raw_images = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(sampler_epoch)
                iterator = iter(loader)
                images, captions, raw_images = next(iterator)
            images = images.to(device, non_blocking=True)
            raw_images = raw_images.to(device, non_blocking=True)
            with torch.no_grad():
                text, text_mask = text_encoder.encode(captions, device)
                null_text, null_mask = text_encoder.encode_null(images.shape[0], device)
            text, null_text = text.to(images.dtype), null_text.to(images.dtype)
            synchronize = micro_step + 1 == args.gradient_accumulation_steps
            no_sync = model.no_sync if hasattr(model, "no_sync") and not synchronize else contextlib.nullcontext
            with no_sync():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    losses = model(images, text, text_mask, null_text, null_mask, raw_images)
                    loss = losses["loss"] / args.gradient_accumulation_steps
                if not math.isfinite(float(loss.detach())):
                    raise FloatingPointError(f"Non-finite loss: {float(loss.detach())}")
                loss.backward()
            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach()) / args.gradient_accumulation_steps

        grad_norm = None
        if args.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), args.gradient_clip)
        optimizer.step()
        model_without_ddp.update_ema()
        reduced = {name: misc.all_reduce_mean(value) for name, value in loss_sums.items()}
        metrics.update(**reduced, lr=lr, grad_norm=float(grad_norm) if grad_norm is not None else None)
        if global_step % args.log_freq == 0:
            print(f"Step [{global_step}/{args.max_train_steps}]  {metrics}  time: {time.time() - last_log:.2f}s")
            last_log = time.time()
            if log_writer is not None and misc.is_main_process():
                for name, value in reduced.items():
                    log_writer.add_scalar(f"train_{name}", value, global_step)
                log_writer.add_scalar("lr", lr, global_step)

        should_save = global_step + 1 == args.max_train_steps or (
            global_step > start_step and global_step % args.save_steps == 0
        )
        if args.output_dir and should_save:
            args.global_step = global_step
            misc.save_model(args, model_without_ddp, optimizer, global_step, epoch_name="last")
            if args.keep_checkpoint_steps > 0 and global_step % args.keep_checkpoint_steps == 0:
                misc.save_model(args, model_without_ddp, optimizer, global_step)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        if args.eval_steps > 0 and global_step > start_step and global_step % args.eval_steps == 0:
            save_validation(model_without_ddp, text_encoder, device, global_step, args)
            model.train()


@torch.no_grad()
def save_validation(model, text_encoder, device, step, args):
    if not misc.is_main_process():
        return
    prompts = [prompt for prompt in args.validation_prompts if prompt]
    if not prompts:
        return
    output = os.path.join(args.output_dir, "vis")
    os.makedirs(output, exist_ok=True)
    current = model.capture_model_state()
    model.load_ema_model_state(1)
    model.eval()
    text, mask = text_encoder.encode(prompts, device)
    null_text, null_mask = text_encoder.encode_null(len(prompts), device)
    dtype = next(model.net.parameters()).dtype
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        samples = model.generate(text.to(dtype), null_text.to(dtype), mask, null_mask)
    save_image(samples.add(1).div(2).clamp(0, 1), os.path.join(output, f"step_{step:08d}.png"), nrow=min(4, len(prompts)))
    model.load_model_state(current)


__all__ = ["adjust_learning_rate", "save_validation", "train_steps"]
