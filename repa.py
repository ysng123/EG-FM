"""Optional frozen representation encoder used by REPA."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2(nn.Module):
    """Lazily load the frozen DINOv2 encoder used by REPA."""

    def __init__(self, model_name: str = "dinov2_vitb14", base_patch_size: int = 16):
        super().__init__()
        self.model_name = model_name
        self.base_patch_size = base_patch_size
        self.encoder = None

    def _load(self):
        if self.encoder is None:
            try:
                encoder = torch.hub.load(
                    "facebookresearch/dinov2", self.model_name, trust_repo=True
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not load DINOv2; provide network access or a warm torch hub cache"
                ) from exc
            encoder.head = nn.Identity()
            encoder.eval()
            encoder.requires_grad_(False)
            self.encoder = encoder
        return self.encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder = self._load().to(images.device)
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std
        height, width = images.shape[-2:]
        images = F.interpolate(
            images,
            (14 * height // self.base_patch_size, 14 * width // self.base_patch_size),
            mode="bicubic",
            align_corners=False,
        )
        autocast = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if images.is_cuda
            else nullcontext()
        )
        with autocast:
            features = encoder.forward_features(images)["x_norm_patchtokens"]
        return features
