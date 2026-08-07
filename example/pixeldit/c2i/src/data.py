# Modified from https://github.com/MCG-NJU/PixNerd and https://github.com/End2End-Diffusion/REPA-E/

import copy
import io
import json
import os
import random
import re
import time
import unicodedata
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import PIL.Image
import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir: str):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)

        labels = self._load_h5_file("dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _load_h5_file(self, path):
        if path.endswith('.png'):
            rtn = np.array(PIL.Image.open(io.BytesIO(np.array(self.h5f[path]))))
            rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
        elif path.endswith('.json'):
            rtn = json.loads(np.array(self.h5f[path]).tobytes().decode('utf-8'))
        elif path.endswith('.npy'):
            rtn = np.array(self.h5f[path])
        else:
            raise ValueError(f'Unknown file type: {path}')
        return rtn

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        image_fname = self.filelist[index]
        image = self._load_h5_file(image_fname)

        image_tensor = torch.from_numpy(image).float() / 255.0
        normalized_image = (image_tensor - 0.5) / 0.5

        target = int(self.labels[index])
        metadata = {
            "raw_image": image_tensor,
            "class": target,
        }
        return normalized_image, target, metadata


def _clean_filename(s: str) -> str:
    s = s.strip().strip('.')
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    illegal_chars = r'[/]'
    s = re.sub(illegal_chars, '_', s)
    s = re.sub(r'_{2,}', '_', s)
    s = s.lower()
    max_length = 200
    s = s[:max_length]
    if not s:
        return 'untitled'
    return s


def _save_fn(image, metadata, root_path):
    image_path = os.path.join(root_path, f"{metadata['filename']}.png")
    PIL.Image.fromarray(image).save(image_path)


class RandomNDataset(Dataset):
    def __init__(self, latent_shape=(4, 64, 64), conditions: Union[int, List, str] = None,
                 seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if isinstance(conditions, int):
            conditions = list(range(conditions))
        elif isinstance(conditions, str):
            if os.path.exists(conditions):
                conditions = open(conditions, "r").read().splitlines()
            else:
                raise FileNotFoundError(conditions)
        elif isinstance(conditions, list):
            conditions = conditions
        self.conditions = conditions
        self.num_conditons = len(conditions)
        self.seeds = seeds

        if num_samples_per_instance > 0:
            max_num_instances = num_samples_per_instance * self.num_conditons
        else:
            max_num_instances = max_num_instances

        if seeds is not None:
            self.max_num_instances = len(seeds) * self.num_conditons
            self.num_seeds = len(seeds)
        else:
            self.num_seeds = (max_num_instances + self.num_conditons - 1) // self.num_conditons
            self.max_num_instances = self.num_seeds * self.num_conditons
        self.latent_shape = latent_shape

    def __getitem__(self, idx):
        condition = self.conditions[idx // self.num_seeds]

        seed = random.randint(0, 1 << 31)
        if self.seeds is not None:
            seed = self.seeds[idx % self.num_seeds]

        filename = f"{_clean_filename(str(condition))}_{seed}"
        generator = torch.Generator().manual_seed(seed)
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        metadata = dict(
            filename=filename,
            seed=seed,
            condition=condition,
            save_fn=_save_fn,
        )
        return latent, condition, metadata

    def __len__(self):
        return self.max_num_instances


class ClassLabelRandomNDataset(RandomNDataset):
    def __init__(self, latent_shape=(4, 64, 64), num_classes=1000, conditions: Union[int, List, str] = None,
                 seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if conditions is None:
            conditions = list(range(num_classes))
        super().__init__(latent_shape, conditions, seeds, max_num_instances, num_samples_per_instance)


def mirco_batch_collate_fn(batch):
    batch = copy.deepcopy(batch)
    new_batch = []
    for micro_batch in batch:
        new_batch.extend(micro_batch)
    x, y, metadata = list(zip(*new_batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except Exception:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata


def collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except Exception:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata


def eval_collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    x = torch.stack(x, dim=0)
    return x, y, metadata


class DataModule(LightningDataModule):
    def __init__(self,
                 train_dataset: Dataset = None,
                 eval_dataset: Dataset = None,
                 pred_dataset: Dataset = None,
                 train_batch_size: int = 64,
                 train_num_workers: int = 16,
                 train_prefetch_factor: int = 8,
                 eval_batch_size: int = 32,
                 eval_num_workers: int = 4,
                 pred_batch_size: int = 32,
                 pred_num_workers: int = 4,
                 seed: int = None):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pred_dataset = pred_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size
        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers
        self.seed = seed if seed is not None else int(time.time())
        self._train_dataloader: Optional[DataLoader] = None

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        micro_batch_size = getattr(self.train_dataset, "micro_batch_size", None)
        if micro_batch_size is not None:
            assert self.train_batch_size % micro_batch_size == 0
            dataloader_batch_size = self.train_batch_size // micro_batch_size
            train_collate_fn = mirco_batch_collate_fn
        else:
            dataloader_batch_size = self.train_batch_size
            train_collate_fn = collate_fn

        if not isinstance(self.train_dataset, IterableDataset):
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                seed=int(self.seed) if self.seed is not None else 0
            )
        else:
            sampler = None

        self._train_dataloader = DataLoader(
            self.train_dataset,
            dataloader_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(
            self.eval_dataset,
            self.eval_batch_size,
            num_workers=self.eval_num_workers,
            prefetch_factor=2,
            sampler=sampler,
            collate_fn=eval_collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(
            self.pred_dataset,
            batch_size=self.pred_batch_size,
            num_workers=self.pred_num_workers,
            prefetch_factor=4,
            sampler=sampler,
            collate_fn=eval_collate_fn,
        )

