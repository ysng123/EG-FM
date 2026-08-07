"""Shared training and evaluation loops."""

import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch_fidelity

from utils import lr_sched, misc


def _barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def train_one_epoch(
    model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None
):
    model.train(True)
    metrics = misc.MetricLogger(delimiter="  ")
    metrics.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(metrics.log_every(data_loader, 20, f"Epoch: [{epoch}]")):
        lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)
        raw_images = images.to(device, non_blocking=True).float().div_(255)
        images = raw_images.mul(2.0).sub(1.0)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            losses = model(images, labels, raw_images=raw_images)
        loss = losses["loss"]
        if not math.isfinite(loss.item()):
            print(f"Non-finite loss {loss.item()}, stopping")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        model_without_ddp.update_ema()

        scalar_losses = {name: value.item() for name, value in losses.items()}
        metrics.update(**scalar_losses, lr=optimizer.param_groups[0]["lr"])
        if step % args.log_freq == 0:
            reduced_losses = {
                name: misc.all_reduce_mean(value) for name, value in scalar_losses.items()
            }
        if log_writer is not None and step % args.log_freq == 0:
            x_axis = int((step / len(data_loader) + epoch) * 1000)
            for name, value in reduced_losses.items():
                log_writer.add_scalar(f"train_{name}", value, x_axis)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], x_axis)


def _run_name(model, args):
    model_name = args.model.replace("/", "-")
    return (
        f"{model_name}-{args.prediction_target}-{model.method}-steps{model.steps}"
        f"-cfg{model.cfg_scale}-interval{model.cfg_interval[0]}-{model.cfg_interval[1]}"
        f"-image{args.num_images}-res{args.img_size}"
    )


def _write_metadata(model, args, folder, metrics=None):
    payload = {
        "checkpoint": args.resume_checkpoint
        or (os.path.join(args.resume, "checkpoint-last.pth") if args.resume else ""),
        "model": args.model,
        "prediction_target": args.prediction_target,
        "resolution": args.img_size,
        "sampling_method": model.method,
        "sampling_steps": model.steps,
        "cfg": model.cfg_scale,
        "cfg_interval": list(model.cfg_interval),
        "timeshift": args.timeshift,
        "sigma0": args.freq_ab_sigma0,
        "release_curve": args.freq_ab_curve,
        "num_images": args.num_images,
        "seed": args.seed,
        "timesteps": [float(value) for value in model.sampler.timesteps.tolist()],
        "metrics": metrics or {},
    }
    with open(os.path.join(folder, "sampling.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


@torch.no_grad()
def evaluate(model, args, epoch, batch_size=64, log_writer=None):
    model.eval()
    world_size = misc.get_world_size()
    rank = misc.get_rank()
    samples_per_step = batch_size * world_size
    num_steps = (args.num_images + samples_per_step - 1) // samples_per_step
    save_folder = os.path.join(args.eval_save_root, _run_name(model, args))
    if rank == 0:
        os.makedirs(save_folder, exist_ok=True)
    _barrier()

    original_state = model.capture_model_state()
    model.load_ema_model_state(which=1)
    class_count = args.class_num
    if args.num_images % class_count == 0:
        labels_world = np.arange(class_count).repeat(args.num_images // class_count)
    else:
        labels_world = np.arange(args.num_images) % class_count
    padding = num_steps * samples_per_step - args.num_images
    if padding:
        labels_world = np.pad(labels_world, (0, padding), mode="wrap")
    device = next(model.parameters()).device

    for step in range(num_steps):
        print(f"Generation step {step + 1}/{num_steps}")
        start = world_size * batch_size * step + rank * batch_size
        labels = torch.as_tensor(
            labels_world[start : start + batch_size], dtype=torch.long, device=device
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            samples = model.generate(labels)
        samples = samples.add(1.0).div(2.0).cpu()
        for local_index, sample in enumerate(samples):
            image_id = step * samples.shape[0] * world_size + rank * samples.shape[0] + local_index
            if image_id >= args.num_images:
                break
            image = np.clip(np.round(sample.numpy().transpose(1, 2, 0) * 255), 0, 255)
            cv2.imwrite(
                os.path.join(save_folder, f"{image_id:05d}.png"),
                image.astype(np.uint8)[:, :, ::-1],
            )
    _barrier()
    model.load_model_state(original_state)

    result = None
    if rank == 0 and args.fid_stats:
        fid_stats = os.path.abspath(args.fid_stats)
        if not os.path.isfile(fid_stats):
            raise FileNotFoundError(f"FID statistics not found: {fid_stats}")
        result = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_stats,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = result["frechet_inception_distance"]
        inception_score = result["inception_score_mean"]
        print(f"FID: {fid:.4f}, Inception Score: {inception_score:.4f}")
        if log_writer is not None:
            suffix = f"_{args.model.replace('/', '-')}_{args.prediction_target}_{args.img_size}"
            log_writer.add_scalar(f"fid{suffix}", fid, epoch)
            log_writer.add_scalar(f"is{suffix}", inception_score, epoch)

    if rank == 0:
        serializable = {name: float(value) for name, value in (result or {}).items()}
        if args.fid_stats and not args.keep_eval_images:
            shutil.rmtree(save_folder)
            os.makedirs(save_folder, exist_ok=True)
        else:
            print("Generated images kept at:", save_folder)
        _write_metadata(model, args, save_folder, serializable)
        print("Sampling metadata written to:", os.path.join(save_folder, "sampling.json"))
    _barrier()
