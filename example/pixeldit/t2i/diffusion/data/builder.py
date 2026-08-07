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
import random
import time
from typing import Sequence

from termcolor import colored
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger


class DatasetRegistry:
    """Minimal registry to decouple dataset construction from mmcv."""

    def __init__(self):
        self._module_dict = {}

    def register_module(self, name=None):
        def _register(cls):
            key = name or cls.__name__
            if key in self._module_dict:
                raise KeyError(f"Dataset '{key}' already registered.")
            self._module_dict[key] = cls
            return cls

        return _register

    def build(self, cfg, default_args=None):
        if isinstance(cfg, str):
            cfg = dict(type=cfg)
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise ValueError("Dataset config must be a string or dict with key 'type'.")

        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        if obj_type not in self._module_dict:
            raise KeyError(f"Dataset '{obj_type}' is not registered.")

        kwargs = {}
        if default_args:
            kwargs.update(default_args)
        kwargs.update(cfg)
        return self._module_dict[obj_type](**kwargs)


DATASETS = DatasetRegistry()

def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get("type")
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    cfg = cfg.copy()
    transform = cfg.pop("transform", "default_train")
    transform = get_transform(transform, resolution)
    dataset = DATASETS.build(cfg, default_args=dict(transform=transform, resolution=resolution, **kwargs))
    logger.info(
        f"{colored(f'Dataset {dataset_type} constructed: ', 'green', attrs=['bold'])}"
        f"time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}"
    )
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if "batch_sampler" in kwargs:
        dataloader = DataLoader(
            dataset, batch_sampler=kwargs["batch_sampler"], num_workers=num_workers, pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, **kwargs
        )
    return dataloader


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios: dict,
        drop_last: bool = False,
        config=None,
        valid_num=0,  # take as valid aspect-ratio when sample number >= valid_num
        hq_only=False,
        cache_file=None,
        caching=False,
        **kwargs,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler should be an instance of ``Sampler``, but got {sampler}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.hq_only = hq_only
        self.config = config
        self.caching = caching
        self.cache_file = cache_file
        self.order_check_pass = False

        self.ratio_nums_gt = kwargs.get("ratio_nums", None)
        assert self.ratio_nums_gt, "ratio_nums_gt must be provided."
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios.keys()}
        self.current_available_bucket_keys = [str(k) for k, v in self.ratio_nums_gt.items() if v >= valid_num]

        logger = (
            get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, "train_log.log"))
        )
        logger.warning(
            f"Using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}"
        )

        self.data_all = {} if caching else None
        if os.path.exists(cache_file):
            logger.info(f"Loading cached file for multi-scale training: {cache_file}")
            try:
                self.cached_idx = json.load(open(cache_file))
            except:
                logger.info(f"Failed loading: {cache_file}")
                self.cached_idx = {}
        else:
            logger.info(f"No cached file is found, dataloader is slow: {cache_file}")
            self.cached_idx = {}
        self.exist_ids = len(self.cached_idx)

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info, closest_ratio = self._get_data_info_and_ratio(idx)
            if not data_info:
                continue

            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                self._update_cache(bucket)
                yield bucket[:]
                del bucket[:]

        for bucket in self._aspect_ratio_buckets.values():
            while bucket:
                if not self.drop_last or len(bucket) == self.batch_size:
                    yield bucket[:]
                del bucket[:]

    def _get_data_info_and_ratio(self, idx):
        str_idx = str(idx)
        if self.caching:
            if str_idx in self.cached_idx:
                return self.cached_idx[str_idx], self.cached_idx[str_idx]["closest_ratio"]
            data_info = self.dataset.get_data_info(int(idx))
            if data_info is None or (
                self.hq_only and "version" in data_info and data_info["version"] not in ["high_quality"]
            ):
                return None, None
            closest_ratio = self._get_closest_ratio(data_info["height"], data_info["width"])
            self.data_all[str_idx] = {
                "height": data_info["height"],
                "width": data_info["width"],
                "closest_ratio": closest_ratio,
                "key": data_info["key"],
            }
            return data_info, closest_ratio
        else:
            if self.cached_idx:
                if self.cached_idx.get(str_idx):
                    if not self.order_check_pass or random.random() < 0.01:
                        # Ensure the cached dataset is in the same order as the original tar file
                        self._order_check(str_idx)
                    closest_ratio = self.cached_idx[str_idx]["closest_ratio"]
                    return self.cached_idx[str_idx], closest_ratio

            data_info = self.dataset.get_data_info(int(idx))
            if data_info is None or (
                self.hq_only and "version" in data_info and data_info["version"] not in ["high_quality"]
            ):
                return None, None
            closest_ratio = self._get_closest_ratio(data_info["height"], data_info["width"])

            return data_info, closest_ratio

    def _get_closest_ratio(self, height, width):
        ratio = height / width
        return min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))

    def _order_check(self, str_idx):
        ori_data = self.cached_idx[str_idx]
        real_key = self.dataset.get_data_info(int(str_idx))["key"]
        assert real_key and ori_data["key"] == real_key, ValueError(
            f"index: {str_idx}, real key: {real_key} ori key: {ori_data['key']}"
        )
        self.order_check_pass = True

    def _update_cache(self, bucket):
        if self.caching:
            for idx in bucket:
                if str(idx) in self.cached_idx:
                    continue
                self.cached_idx[str(idx)] = self.data_all.pop(str(idx))
