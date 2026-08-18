"""BLIP3o tar/Hugging Face WebDataset readers for T2I training."""

from __future__ import annotations

import glob
import io
import os
import random
import tarfile

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _resize_and_crop(image, resolution, random_crop, random_flip):
    image = image.convert("RGB")
    if min(image.size) < resolution:
        return None
    while min(image.size) >= 2 * resolution:
        image = image.resize(tuple(side // 2 for side in image.size), Image.Resampling.BOX)
    scale = resolution / min(image.size)
    image = image.resize(
        tuple(round(side * scale) for side in image.size), Image.Resampling.BICUBIC
    )
    if random_crop:
        left = random.randint(0, image.width - resolution)
        top = random.randint(0, image.height - resolution)
    else:
        left = (image.width - resolution) // 2
        top = (image.height - resolution) // 2
    image = image.crop((left, top, left + resolution, top + resolution))
    if random_flip and random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    raw = pil_to_tensor(image).float().div_(255.0)
    return raw.mul(2.0).sub(1.0), raw


def _tar_files(roots, max_shards=0):
    files = []
    for root in str(roots).split(":"):
        root = os.path.expanduser(root)
        files.extend(glob.glob(os.path.join(root, "**", "*.tar"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", "*.tar.gz"), recursive=True))
    files = sorted(set(files))
    return files[:max_shards] if max_shards else files


class BLIP3oTarDataset(IterableDataset):
    def __init__(self, root, resolution=512, random_crop=False, random_flip=True,
                 repeat=True, shuffle_shards=True, max_shards=0):
        super().__init__()
        self.resolution = int(resolution)
        self.random_crop = bool(random_crop)
        self.random_flip = bool(random_flip)
        self.repeat = bool(repeat)
        self.shuffle_shards = bool(shuffle_shards)
        self.tar_files = _tar_files(root, int(max_shards))
        if not self.tar_files:
            raise FileNotFoundError(f"No .tar/.tar.gz files found under {root}")

    def _process(self, image_bytes, caption):
        transformed = _resize_and_crop(
            Image.open(io.BytesIO(image_bytes)), self.resolution,
            self.random_crop, self.random_flip
        )
        if transformed is None or not caption.strip():
            return None
        image, raw = transformed
        return image, caption.strip(), raw

    def _iter_tar(self, path):
        pending = {}
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                stem, extension = os.path.splitext(os.path.basename(member.name))
                extension = extension.lower()
                if extension not in IMAGE_EXTENSIONS | {".txt"}:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                slot = pending.setdefault(stem, {})
                value = extracted.read()
                if extension == ".txt":
                    slot["caption"] = value.decode("utf-8", errors="ignore")
                else:
                    slot["image"] = value
                if "image" in slot and "caption" in slot:
                    try:
                        sample = self._process(slot["image"], slot["caption"])
                    except Exception:
                        sample = None
                    pending.pop(stem, None)
                    if sample is not None:
                        yield sample

    def __iter__(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        shards = self.tar_files[rank::world_size][worker_id::worker_count]
        if not shards:
            return
        while True:
            if self.shuffle_shards:
                random.shuffle(shards)
            for path in shards:
                yield from self._iter_tar(path)
            if not self.repeat:
                return


class BLIP3oHFDataset(Dataset):
    def __init__(self, dataset, resolution=512, random_crop=False,
                 random_flip=True, max_sample_retries=32):
        self.dataset = dataset
        self.resolution = int(resolution)
        self.random_crop = bool(random_crop)
        self.random_flip = bool(random_flip)
        self.max_sample_retries = int(max_sample_retries)
        image_candidates = [name for name in ("jpg", "jpeg", "png", "webp", "image") if name in dataset.column_names]
        caption_candidates = [name for name in ("txt", "text", "caption", "prompt") if name in dataset.column_names]
        if not image_candidates or not caption_candidates:
            raise ValueError(f"Could not resolve image/caption fields from {dataset.column_names}")
        self.image_key, self.caption_key = image_candidates[0], caption_candidates[0]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_pil(value):
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict) and value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value))
        return Image.open(value)

    def __getitem__(self, index):
        last_error = None
        for offset in range(min(self.max_sample_retries, len(self.dataset))):
            try:
                item = self.dataset[(int(index) + offset) % len(self.dataset)]
                transformed = _resize_and_crop(
                    self._to_pil(item[self.image_key]), self.resolution,
                    self.random_crop, self.random_flip
                )
                caption = str(item[self.caption_key]).strip()
                if transformed is not None and caption:
                    image, raw = transformed
                    return image, caption, raw
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to load a valid sample near index {index}") from last_error


def load_blip3o_hf_webdataset(root, cache_dir, num_proc=64, max_shards=0):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face data backend requires datasets; install t2i/requirements.txt"
        ) from exc
    tar_files = _tar_files(root, int(max_shards))
    if not tar_files:
        raise FileNotFoundError(f"No .tar/.tar.gz files found under {root}")
    os.makedirs(cache_dir, exist_ok=True)
    return load_dataset(
        "webdataset", data_files=tar_files, cache_dir=cache_dir,
        split="train", num_proc=int(num_proc)
    ), tar_files


def t2i_collate(batch):
    images, captions, raw_images = zip(*batch)
    return torch.stack(images), list(captions), torch.stack(raw_images)


__all__ = [
    "BLIP3oHFDataset", "BLIP3oTarDataset", "load_blip3o_hf_webdataset", "t2i_collate"
]
