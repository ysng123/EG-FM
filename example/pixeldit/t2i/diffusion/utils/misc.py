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

import json
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from diffusion.utils.dist_utils import get_dist_info

os.environ["MOX_SILENT_MODE"] = "1"  # mute moxing log

# Track how many batch visualizations have been saved in this run per rank
RUN_LOCAL_VIS_COUNT = {}
RUN_LOCAL_VIS_MAX = 30


def init_random_seed(seed=None, device="cuda"):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_data_batch_visualization(
    config,
    global_step,
    rank,
    epoch,
    local_step,
    batch,
    img_batch,
):
    """Save per-rank batch visualization for early iterations.

    - Saves images, captions, and extra metadata (JSON)
    - Best-effort, errors are swallowed to avoid impacting training

    Returns:
        bool: True if attempted and directory created, False otherwise
    """
    try:
        # Guard: only early steps and when enabled
        if not getattr(config.train, "local_save_vis", False):
            return False
        # Limit to first RUN_LOCAL_VIS_MAX iterations of the current run per rank (independent of resume global_step)
        rank_int = int(rank)
        count = RUN_LOCAL_VIS_COUNT.get(rank_int, 0)
        if count >= RUN_LOCAL_VIS_MAX:
            return False

        # Prepare destination directory: <work_dir>/data_vis/rankXX/step_XXXXXX
        vis_dir = osp.join(
            config.work_dir,
            "data_vis",
            f"rank{int(rank):02d}",
            f"step_{int(global_step):06d}",
        )
        os.umask(0o000)
        os.makedirs(vis_dir, exist_ok=True)
        # Increment saved counter for this rank as soon as we create a folder for this step
        RUN_LOCAL_VIS_COUNT[rank_int] = count + 1

        # Unpack batch fields
        captions = batch[1] if isinstance(batch[1], (list, tuple)) else [str(batch[1])]
        cap_types = batch[5] if isinstance(batch[5], (list, tuple)) else [str(batch[5])]
        clip_scores = batch[7] if isinstance(batch[7], (list, tuple)) else [str(batch[7])]
        if torch.is_tensor(batch[4]):
            sample_indices = batch[4].tolist()
        elif isinstance(batch[4], (list, tuple)):
            sample_indices = list(batch[4])
        else:
            sample_indices = list(range(len(captions)))

        # Convert images to uint8 HWC
        imgs_uint8 = (
            torch.clamp(127.5 * img_batch + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        # Save each sample (image + text + extra metadata)
        bs = img_batch.shape[0] if hasattr(img_batch, "shape") else len(captions)
        for i in range(bs):
            idx_id = sample_indices[i] if i < len(sample_indices) else i

            # Save caption and metadata (flat text)
            try:
                txt_path = osp.join(vis_dir, f"{i:04d}_idx{idx_id}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(str(captions[i]) if captions[i] is not None else "")
                    f.write("\n")
                    f.write(f"[caption_type]: {cap_types[i] if i < len(cap_types) else ''}\n")
                    f.write(f"[clipscore]: {clip_scores[i] if i < len(clip_scores) else ''}\n")
            except Exception:
                pass

            # Save image (if available)
            try:
                if imgs_uint8 is not None and i < len(imgs_uint8):
                    img = Image.fromarray(imgs_uint8[i])
                    img_path = osp.join(vis_dir, f"{i:04d}_idx{idx_id}.jpg")
                    img.save(img_path)
            except Exception:
                pass

            # Save additional per-sample information to JSON
            try:
                meta = {
                    "rank": int(rank),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "local_step": int(local_step),
                    "dataset_index": int(idx_id),
                    "caption_type": (cap_types[i] if i < len(cap_types) else None),
                    "clipscore": (clip_scores[i] if i < len(clip_scores) else None),
                }

                # data_info (original h/w and aspect ratio)
                try:
                    di = batch[3]
                    sample_di = {}
                    if isinstance(di, dict):
                        if "img_hw" in di:
                            v = di["img_hw"]
                            if torch.is_tensor(v):
                                v_i = v[i]
                                sample_di["img_hw"] = [int(v_i[0].item()), int(v_i[1].item())]
                            elif isinstance(v, (list, tuple)):
                                v_i = v[i] if len(v) > i else v
                                sample_di["img_hw"] = [int(v_i[0]), int(v_i[1])]
                        if "aspect_ratio" in di:
                            ar = di["aspect_ratio"]
                            if torch.is_tensor(ar):
                                ar_i = ar[i].item() if ar.ndim > 0 else float(ar.item())
                                sample_di["aspect_ratio"] = float(ar_i)
                            elif isinstance(ar, (list, tuple)):
                                ar_i = ar[i] if len(ar) > i else ar
                                sample_di["aspect_ratio"] = float(ar_i)
                            else:
                                sample_di["aspect_ratio"] = float(ar)
                    meta["data_info"] = sample_di
                except Exception:
                    pass

                # dataindex_info
                try:
                    dii = batch[6]
                    if isinstance(dii, dict):
                        sample_dii = {}
                        if "index" in dii:
                            v = dii["index"]
                            if torch.is_tensor(v):
                                sample_dii["index"] = int(v[i].item())
                            elif isinstance(v, (list, tuple)):
                                sample_dii["index"] = int(v[i]) if len(v) > i else int(v)
                            else:
                                sample_dii["index"] = int(v)
                        if "shard" in dii:
                            v = dii["shard"]
                            if isinstance(v, (list, tuple)):
                                sample_dii["shard"] = str(v[i]) if len(v) > i else str(v)
                            else:
                                sample_dii["shard"] = str(v)
                        if "shardindex" in dii:
                            v = dii["shardindex"]
                            if torch.is_tensor(v):
                                sample_dii["shardindex"] = int(v[i].item())
                            elif isinstance(v, (list, tuple)):
                                sample_dii["shardindex"] = int(v[i]) if len(v) > i else int(v)
                            else:
                                sample_dii["shardindex"] = int(v)
                        meta["dataindex_info"] = sample_dii
                except Exception:
                    pass

                # attention mask summary (if available)
                try:
                    attn = batch[2]
                    if torch.is_tensor(attn):
                        attn_i = attn[i]
                        meta["attention_mask_sum"] = int(attn_i.sum().item())
                        meta["attention_mask_shape"] = list(attn_i.shape)
                except Exception:
                    pass

                json_path = osp.join(vis_dir, f"{i:04d}_idx{idx_id}.json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(meta, jf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return True
    except Exception:
        # Best-effort; never crash training due to visualization
        return False
