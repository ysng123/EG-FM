# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from https://github.com/NVlabs/Sana

import datetime
import gc
import getpass
import hashlib
import json
import os, sys
import os.path as osp
import time
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")  # ignore warning

import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches, DistributedDataParallelKwargs
from PIL import Image
from termcolor import colored

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion import DPMS, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import PixDiTConfig, model_init_config
from diffusion.data.builder import AspectRatioBatchSampler
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import init_random_seed, set_random_seed, save_data_batch_visualization
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def set_fsdp_env():
    # Basic FSDP settings
    os.environ["ACCELERATE_USE_FSDP"] = "true"

    # Auto wrapping policy
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    # Align FSDP auto-wrap target with PixDiT transformer block
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "MMDiTBlockT2I"

    # Performance optimization settings
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"

    # State dict settings
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"


def ema_update(model_dest, model_src, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


@torch.no_grad()
def log_validation(accelerator, config, model, logger, step, device, init_noise=None):
    torch.cuda.empty_cache()

    vis_sampler = config.scheduler.vis_sampler
    if vis_sampler != "flow_dpm-solver":
        raise ValueError(f"Unsupported vis_sampler {vis_sampler}; only flow_dpm-solver is supported.")
    model = accelerator.unwrap_model(model).eval()
    try:
        model_dtype = next(model.parameters()).dtype
    except Exception:
        model_dtype = torch.float32
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device).to(dtype=model_dtype)

    logger.info("Running validation... ")
    image_logs = []

    def run_sampling(init_z=None, label_suffix="", sampler="flow_dpm-solver"):
        if sampler != "flow_dpm-solver":
            raise ValueError(f"{sampler} not implemented; use flow_dpm-solver.")
        latents = []
        current_image_logs = []
        for prompt in validation_prompts:
            z = (
                torch.randn(1, 3, image_size, image_size, device=device)
                if init_z is None
                else init_z
            )
            embed = torch.load(
                osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
                map_location="cpu",
            )
            caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
            caption_embs = caption_embs.to(dtype=model_dtype)
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=caption_embs,
                uncondition=null_y,
                cfg_scale=4.5,
                model_type="flow",
                model_kwargs=model_kwargs,
                schedule="FLOW",
            )
            denoised = dpm_solver.sample(
                z,
                steps=20,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=config.scheduler.flow_shift,
            )

            latents.append(denoised)
        torch.cuda.empty_cache()

        for prompt, latent in zip(validation_prompts, latents):
            samples = (
                torch.clamp(127.5 * latent + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()[0]
            )
            image = Image.fromarray(samples)
            current_image_logs.append({"validation_prompt": prompt + label_suffix, "images": [image]})

        return current_image_logs

    image_logs += run_sampling(init_z=None, label_suffix="", sampler=vis_sampler)

    if init_noise is not None:
        torch.cuda.empty_cache()
        gc.collect()
        init_noise = torch.clone(init_noise).to(device)
        image_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", sampler=vis_sampler)

    formatted_images = []
    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        for image in images:
            formatted_images.append((validation_prompt, np.asarray(image)))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for validation_prompt, image in formatted_images:
                tracker.writer.add_images(validation_prompt, image[None, ...], step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb

            wandb_images = []
            for validation_prompt, image in formatted_images:
                wandb_images.append(wandb.Image(image, caption=validation_prompt, file_type="jpg"))
            tracker.log({"validation": wandb_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io

        images = [log["images"][0] for log in image_caption]
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))

        new_im = Image.new("RGB", (max_width * images_per_row, total_height))

        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)

        return new_im

    if config.train.local_save_vis:
        file_format = "webp"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        concatenated_image = concatenate_images(image_logs, images_per_row=5, image_format=file_format)
        save_path = (
            osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
            if init_noise is None
            else osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
        )
        concatenated_image.save(save_path)

    model.train()
    flush()
    return image_logs


def train(
    config, args, accelerator, model, model_ema, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger, dino=None
):
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0
    model_instance.to(accelerator.device)

    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))

        # Determine this rank's chunk size for clearer progress logs
        try:
            worker_chunk = (
                train_dataloader.batch_sampler.sampler.worker_end
                - train_dataloader.batch_sampler.sampler.worker_start
            )
        except Exception:
            worker_chunk = None

        for index, _ in enumerate(train_dataloader):
            if index % 2000 == 0:
                cached = len(train_dataloader.batch_sampler.cached_idx)
                if worker_chunk is not None:
                    logger.info(f"rank: {rank}, Cached samples: {cached} / {worker_chunk}")
                    print(f"rank: {rank}, Cached samples: {cached} / {worker_chunk}")
                else:
                    logger.info(f"rank: {rank}, Cached samples: {cached}")
                    print(f"rank: {rank}, Cached samples: {cached}")

            # Periodic save to avoid losing progress; DO NOT barrier here
            if (time.time() - caching_start) / 3600 > 3.5:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                break

            # Opportunistic save near completion of this rank's share
            if worker_chunk is not None and len(train_dataloader.batch_sampler.cached_idx) >= max(worker_chunk - 1000, 0):
                logger.info(
                    f"Saving rank: {rank}, Cached samples: {len(train_dataloader.batch_sampler.cached_idx)} / {worker_chunk}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            continue

        # Final save and a single global sync before exit
        print(f"Saving rank-{rank} Cached samples: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        accelerator.wait_for_everyone()
        return

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # image, json_info, key = batch
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            # Best-effort data visualization for early iterations (per-rank)
            save_data_batch_visualization(
                config=config,
                global_step=global_step,
                rank=rank,
                epoch=epoch,
                local_step=step,
                batch=batch,
                img_batch=batch[0],
            )
            vae_time_start = time.time()

            z = batch[0].to(accelerator.device)

            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            clean_images = z
            data_info = batch[3]

            lm_time_start = time.time()
            bs = clean_images.shape[0]

            if load_text_feat:
                y = batch[1]  # bs, 1, N, C
                y_mask = batch[2]  # bs, 1, 1, N
            else:
                if (
                    "gemma" in config.text_encoder.text_encoder_name
                    or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    with torch.no_grad():
                        if not config.text_encoder.chi_prompt:
                            max_length_all = config.text_encoder.model_max_length
                            prompt = batch[1]
                        else:
                            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                            prompt = [chi_prompt + i for i in batch[1]]
                            num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
                            max_length_all = (
                                num_sys_prompt_tokens + config.text_encoder.model_max_length - 2
                            )  # magic number 2: [bos], [_]
                        txt_tokens = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=max_length_all,
                            truncation=True,
                            return_tensors="pt",
                        ).to(accelerator.device)
                        select_index = [0] + list(
                            range(-config.text_encoder.model_max_length + 1, 0)
                        )  # first bos and end N-1
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                            :, :, select_index
                        ]
                        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
                else:
                    raise ValueError("Unsupported text encoder configuration.")

            # classifier-free dropout during training
            p_drop = float(getattr(config.model, "class_dropout_prob", 0.0) or 0.0)
            if p_drop > 0 and isinstance(y, torch.Tensor):
                if null_y_train is not None and null_y_mask is not None:
                    null_y_local = null_y_train.to(device=y.device, dtype=y.dtype)
                    null_y_mask_local = null_y_mask.to(device=y_mask.device, dtype=y_mask.dtype)
                else:
                    null_y_local = torch.zeros_like(y)
                    null_y_mask_local = torch.zeros_like(y_mask)

                drop_mask = (torch.rand((bs,), device=y.device) < p_drop).view(bs, 1, 1, 1)
                y = torch.where(drop_mask, null_y_local.expand_as(y), y)
                y_mask = torch.where(drop_mask, null_y_mask_local.expand_as(y_mask), y_mask)

            # Ensure text embedding dtype matches model/image dtype to avoid matmul dtype mismatch
            if 'y' in locals() and isinstance(y, torch.Tensor):
                y = y.to(dtype=clean_images.dtype)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_images.device)
            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                # Prepare REPA external tokens up-front if requested
                repa_tokens = None
                prepare_repa = (
                    getattr(config.train, "repa_loss_weight", 0.0) > 0.0
                    and dino is not None
                    and getattr(config.model, "model", "") in ["PixDiTTrainer"]
                )
                if prepare_repa:
                    with torch.no_grad():
                        imgs01 = (clean_images.float() + 1.0) / 2.0
                        imgs224 = F.interpolate(imgs01, size=(224, 224), mode="bicubic", align_corners=False)
                        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN, device=imgs224.device, dtype=imgs224.dtype).view(1, 3, 1, 1)
                        std = torch.as_tensor(IMAGENET_DEFAULT_STD, device=imgs224.device, dtype=imgs224.dtype).view(1, 3, 1, 1)
                        imgs224 = (imgs224 - mean) / std
                        feats = dino.forward_features(imgs224)
                        if isinstance(feats, dict):
                            if "x_norm_patchtokens" in feats:
                                repa_tokens = feats["x_norm_patchtokens"]  # [B, 256, 768]
                            elif "tokens" in feats:
                                repa_tokens = feats["tokens"][:, 1:]
                        elif isinstance(feats, torch.Tensor):
                            repa_tokens = feats[:, 1:]
                loss_term = train_diffusion.training_losses(
                    model,
                    clean_images,
                    timesteps,
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, repa_tokens=repa_tokens),
                )
                loss = loss_term["loss"].mean()

                if isinstance(loss_term, dict) and (loss_term.get("extra", None) is not None):
                    repa_loss = loss_term["extra"].get("repa_loss", None)
                    if repa_loss is not None:
                        loss = loss + float(getattr(config.train, "repa_loss_weight", 0.1)) * repa_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if not config.train.use_fsdp and config.train.ema_update and model_ema is not None:
                        ema_update(model_ema, model, config.train.ema_rate)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            logs.update(opt_step=int(accelerator.sync_gradients))
            if 'repa_loss' in locals() and repa_loss is not None:
                logs.update(repa_loss=accelerator.gather(repa_loss).mean().item())
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_preproc = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step

                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, pre:{t_preproc:.3f}, lr:{lr:.3e}, Cap: {batch[5][0]}, "
                )
                # Handle models that may not have h and w attributes
                if hasattr(model, "module"):
                    if hasattr(model.module, "h") and hasattr(model.module, "w"):
                        info += f"s:({model.module.h}, {model.module.w}), "
                    else:
                        info += f"s:(N/A), "
                else:
                    if hasattr(model, "h") and hasattr(model, "w"):
                        info += f"s:({model.h}, {model.w}), "
                    else:
                        info += f"s:(N/A), "

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
            ):
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

                # Choose different saving methods based on whether FSDP is used
                if config.train.use_fsdp:
                    # FSDP mode
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        model=model,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        step=global_step,
                        add_symlink=True,
                    )
                else:
                    # DDP mode
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        ckpt_saved_path = save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            model=accelerator.unwrap_model(model),
                            model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            step=global_step,
                            generator=generator,
                            add_symlink=True,
                        )

                if accelerator.is_main_process:
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.early_stop_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return

            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                if config.train.use_fsdp:
                    merged_state_dict = accelerator.get_state_dict(model)

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if config.train.use_fsdp:
                        model_instance.load_state_dict(merged_state_dict)
                    if validation_noise is not None:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            init_noise=validation_noise,
                        )
                    else:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                        )
                

            # avoid dead-lock of multiscale data batch sampler
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                global_step = (
                    (global_step + train_dataloader_len - 1) // train_dataloader_len
                ) * train_dataloader_len + 1
                logger.info("Early stop current iteration")
                skip_first_batches(train_dataloader, True)
                break

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()

            # Choose different saving methods based on whether FSDP is used
            if config.train.use_fsdp:
                # FSDP mode
                os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    work_dir=osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    model=model,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=global_step,
                    add_symlink=True,
                )
            else:
                # DDP mode
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )

            if accelerator.is_main_process:
                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
            
            # no-op hook removed


@pyrallis.wrap()
def main(cfg: PixDiTConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, generator, num_replicas, rank, training_start_time
    global load_text_feat, validation_noise, text_encoder, tokenizer
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global null_y_train, null_y_mask
    global image_size, cache_file, total_steps, model_instance

    config = cfg
    args = cfg

    # 1.Initialize training mode
    if config.train.use_fsdp:
        set_fsdp_env()
        init_train = "FSDP"
    else:
        init_train = "DDP"

    training_start_time = time.time()
    load_from = True

    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=config.train.resume_lr_scheduler,
        )

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler, ddp_kwargs],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    # Support time-based seed when config.train.seed < 0; otherwise preserve existing behavior
    if getattr(config.train, "seed", None) is None:
        config.train.seed = init_random_seed(None)
    elif int(config.train.seed) < 0:
        config.train.seed = int(time.time())
    else:
        config.train.seed = init_random_seed(config.train.seed)
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    logger.info(f"Set seed: {config.train.seed}, local_rank: {int(os.environ['LOCAL_RANK'])}")
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project="pixeldit", name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")

    image_size = config.model.image_size
    latent_size = image_size  # PixDiTTrainer operates directly in pixel space
    
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length

    validation_noise = (
        torch.randn(
            1, 
            3, 
            latent_size, 
            latent_size, 
            device="cpu", 
            generator=generator
        )
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        # Robustly infer text embedding dimension
        try:
            text_embed_dim = int(getattr(text_encoder.config, "hidden_size"))
        except Exception:
            try:
                text_embed_dim = int(getattr(getattr(text_encoder, "config", object()), "hidden_size"))
            except Exception:
                text_embed_dim = int(getattr(config.text_encoder, "caption_channels", 4096))
    else:
        text_embed_dim = config.text_encoder.caption_channels

    if config.text_encoder.chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        logger.info(f"Complex Human Instruct: {chi_prompt}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    safe_text_encoder_name = str(config.text_encoder.text_encoder_name).replace("/", "-")
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{safe_text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )

    # training-time null prompt embedding for CFG dropout (when not loading text feat)
    null_y_train = None
    null_y_mask = None
    if not config.data.load_text_feat and tokenizer is not None and text_encoder is not None:
        with torch.no_grad():
            null_tokens_train = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb_train = text_encoder(
                null_tokens_train.input_ids, attention_mask=null_tokens_train.attention_mask
            )[0].detach()
            null_attention_mask_train = null_tokens_train.attention_mask
        null_y_train = null_token_emb_train[:, None]  # [1, 1, L, C]
        null_y_mask = null_attention_mask_train[:, None, None, :]  # [1, 1, 1, L]

    # 2.preparing embeddings for visualization. We put it here for saving GPU memory
    if config.train.visualize and len(config.train.validation_prompts):
        valid_prompt_embed_suffix = f"{max_length}token_{safe_text_encoder_name}_{text_embed_dim}.pth"
        validation_prompts = config.train.validation_prompts
        skip = True
        if config.text_encoder.chi_prompt:
            uuid_sys_prompt = hashlib.sha256(chi_prompt.encode()).hexdigest()
        else:
            uuid_sys_prompt = hashlib.sha256(b"").hexdigest()
        config.train.valid_prompt_embed_root = osp.join(config.train.valid_prompt_embed_root, uuid_sys_prompt)
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)

        if config.text_encoder.chi_prompt:
            # Save system prompt to a file
            system_prompt_file = osp.join(config.train.valid_prompt_embed_root, "system_prompt.txt")
            with open(system_prompt_file, "w", encoding="utf-8") as f:
                f.write(chi_prompt)

        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_text_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
                tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name)

            for prompt in validation_prompts:
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                if (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    if not config.text_encoder.chi_prompt:
                        max_length_all = config.text_encoder.model_max_length
                    else:
                        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                        prompt = chi_prompt + prompt
                        num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
                        max_length_all = (
                            num_sys_prompt_tokens + config.text_encoder.model_max_length - 2
                        )  # magic number 2: [bos], [_]

                    txt_tokens = tokenizer(
                        prompt,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][
                        :, select_index
                    ]
                    caption_emb_mask = txt_tokens.attention_mask[:, select_index]
                else:
                    raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")

                torch.save({"caption_embeds": caption_emb, "emb_mask": caption_emb_mask}, prompt_embed_path)

            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            if "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
                null_attention_mask = null_tokens.attention_mask
            else:
                raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_attention_mask},
                null_embed_path,
            )
            if null_y_train is None:
                null_y_train = null_token_emb.detach()[:, None]
                null_y_mask = null_attention_mask[:, None, None, :]
            if config.data.load_text_feat:
                del tokenizer
                del text_encoder
            del null_token_emb
            del null_tokens
            flush()

    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 3. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
        egfm_sigma0=config.scheduler.egfm_sigma0,
        egfm_curve=config.scheduler.egfm_curve,
        egfm_release_start=config.scheduler.egfm_release_start,
        egfm_release_end=config.scheduler.egfm_release_end,
    )
    predict_info = (
        f"flow-prediction: {config.scheduler.predict_flow_v}, noise schedule: {config.scheduler.noise_schedule}"
    )
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 4. build models
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        null_embed_path=null_embed_path,
        **model_kwargs,
    ).train()

    if (not config.train.use_fsdp) and config.train.ema_update:
        model_ema = deepcopy(model).eval()
        logger.info("Creating EMA model for DDP mode")
    elif config.train.use_fsdp and config.train.ema_update:
        logger.warning("EMA update is not supported in FSDP mode. Setting model_ema to None.")
        model_ema = None
    else:
        model_ema = None

    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )

    if config.train.use_fsdp:
        model_instance = deepcopy(model)
    elif model_ema is not None:
        model_instance = deepcopy(model_ema)
    else:
        model_instance = model

    # 4-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        _, missing, unexpected, _ = load_checkpoint(
            checkpoint=config.model.load_from,
            model=model,
            model_ema=model_ema,
            FSDP=config.train.use_fsdp,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")

    if config.train.ema_update and not config.train.use_fsdp and model_ema is not None:
        ema_update(model_ema, model, 0.0)

    # 5. build dataloader
    config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        real_prompt_ratio=config.train.real_prompt_ratio,
        max_length=max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_downsample_rate,
    )
    accelerator.wait_for_everyone()
    if config.model.multi_scale:
        drop_last = True
        uuid = hashlib.sha256("-".join(config.data.data_dir).encode()).hexdigest()[:8]
        # Estimate cache file size: entries * avg bytes (tunable via env)
        avg_bytes_per_item = int(os.environ.get("WIDS_CACHE_EST_BYTES_PER_ITEM", "1100"))
        size_threshold_bytes = int(os.environ.get("WIDS_CACHE_SIZE_THRESHOLD_BYTES", str(1024 ** 3)))  # 1GB
        expected_cache_bytes = len(dataset) * avg_bytes_per_item

        local_cache_root = "./data_wds_caching"
        home_cache_root = osp.expanduser("~/.cache/_wids_batchsampler_cache")
        cache_root = local_cache_root if expected_cache_bytes > size_threshold_bytes else home_cache_root
        os.makedirs(cache_root, exist_ok=True)

        # Use a stable dataset key to avoid slashes/very long names
        dataset_key = hashlib.sha256("\n".join(sorted(config.data.data_dir)).encode()).hexdigest()[:16]
        base_pattern = (
            f"{cache_root}/{getpass.getuser()}-{uuid}-sort_dataset{config.data.sort_dataset}"
            f"-hq_only{config.data.hq_only}-valid_num{config.data.valid_num}"
            f"-aspect_ratio{len(dataset.aspect_ratio)}-droplast{drop_last}"
            f"dataset_len{len(dataset)}"
        )
        cache_file = f"{base_pattern}-num_replicas{num_replicas}-rank{rank}-{dataset_key}.json"

        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=drop_last,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
            hq_only=config.data.hq_only,
            cache_file=cache_file,
            caching=args.caching,
            clipscore_filter_thres=args.data.del_img_clip_thr,
        )
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers)
        train_dataloader_len = len(train_dataloader)
        logger.info(f"rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
    else:
        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            sampler=sampler,
        )
        train_dataloader_len = len(train_dataloader)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)
    # 6. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except Exception as e:
            logger.warning(f"accelerator.init_trackers failed for project '{args.tracker_project_name}', falling back to TensorBoard only: {e}")
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # 7. Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0

        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                config.model.resume_from["resume_optimizer"] = True
                config.model.resume_from["resume_lr_scheduler"] = True
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["resume_optimizer"] = config.train.load_from_optimizer
                config.model.resume_from["resume_lr_scheduler"] = config.train.load_from_lr_scheduler
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, _ = load_checkpoint(
                **config.model.resume_from,
                model=model,
                model_ema=model_ema if not config.train.use_fsdp else None,
                FSDP=config.train.use_fsdp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                null_embed_path=null_embed_path,
            )

            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            # If requested, force the learning rate to follow YAML after resuming
            if getattr(config.train, "override_lr_on_resume", False):
                try:
                    new_lr = float(config.train.optimizer["lr"])
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                    if hasattr(lr_scheduler, "base_lrs") and isinstance(lr_scheduler.base_lrs, list):
                        lr_scheduler.base_lrs = [new_lr for _ in lr_scheduler.base_lrs]
                    logger.info(f"Override LR on resume: set lr to {new_lr} from YAML.")
                except Exception as e:
                    logger.warning(f"Failed to override LR on resume: {e}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass
    # load checkpoint hooks removed

    # 8. Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    if model_ema is not None and not config.train.use_fsdp:
        model_ema = accelerator.prepare(model_ema)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # load everything except model when resume
    if (
        config.train.use_fsdp
        and config.model.resume_from is not None
        and config.model.resume_from["checkpoint"] is not None
        and config.model.resume_from["resume_optimizer"]
        and config.model.resume_from["resume_lr_scheduler"]
    ):
        logger.info(f"FSDP resume: Loading optimizer, scheduler, scaler, random_states...")
        accelerator.load_state(
            os.path.join(config.model.resume_from["checkpoint"], "model"),
            state_dict_key=["optimizer", "scheduler", "scaler", "random_states"],
        )

        # Ensure LR override also applies after accelerator state load (FSDP path)
        if getattr(config.train, "override_lr_on_resume", False):
            try:
                new_lr = float(config.train.optimizer["lr"])
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                if hasattr(lr_scheduler, "base_lrs") and isinstance(lr_scheduler.base_lrs, list):
                    lr_scheduler.base_lrs = [new_lr for _ in lr_scheduler.base_lrs]
                logger.info(f"Override LR on resume (FSDP): set lr to {new_lr} from YAML.")
            except Exception as e:
                logger.warning(f"Failed to override LR on resume (FSDP): {e}")

    set_random_seed((start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
    logger.info(f'Set seed: {(start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')

    # Initialize DINOv2 for REPA if requested
    dino = None
    try:
        repa_w = float(getattr(config.train, "repa_loss_weight", 0.0))
    except Exception as e:
        logger.warning(f"Invalid repa_loss_weight in config; defaulting to 0.0. Error: {e}")
        repa_w = 0.0
    if repa_w > 0.0 and config.model.model in ['PixDiTTrainer']:
        try:
            import timm
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(accelerator.device)
            dino.eval()
            for p in dino.parameters():
                p.requires_grad = False
        except Exception as e:
            dino = None
            logger.warning(f"Failed to load DINOv2 model for REPA; REPA will be disabled. Error: {e}")

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
        dino=dino,
    )


if __name__ == "__main__":
    main()
