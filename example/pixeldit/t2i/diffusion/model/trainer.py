import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS

# Ensure the top-level pixdit_core repo (shared components) is importable when running inside job sandboxes
try:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from pixdit_core.pixeldit_t2i import PixDiT_T2I  # type: ignore
except Exception:
    PixDiT_T2I = None  # type: ignore



@MODELS.register_module()
class PixDiTTrainer(nn.Module):
    """
    Pixel-space PixDiT trainer/wrapper for text-to-image.

    Expects:
    - x: [B, 3, H, W] in [-1, 1]
    - timestep: [B]
    - y: [B, 1, L, C] text embeddings (Gemma/Qwen), will be squeezed to [B, L, C]
    Returns dict with key "x": predicted image in pixel space.
    """

    def __init__(
        self,
        input_size=32,
        in_channels=3,
        image_size=512,
        class_dropout_prob: float = 0.0,
        pred_sigma: bool = False,
        learn_sigma: bool = False,
        config=None,
        caption_channels: int = 2304,
        model_max_length: int = 300,
        extra=None,
        **kwargs,
    ):
        super().__init__()

        extra = extra or {}

        patch_size = int(extra.get("patch_size", 32))
        num_groups = int(extra.get("num_groups", 24))
        hidden_size = int(extra.get("hidden_size", 1920))
        pixel_hidden_size = int(extra.get("pixel_hidden_size", extra.get("hidden_size_x", 32)))
        pixel_attn_hidden_size = int(extra.get("pixel_attn_hidden_size", hidden_size))
        pixel_num_groups = int(extra.get("pixel_num_groups", num_groups))
        total_depth = int(extra.get("depth", extra.get("num_blocks", 18)))
        patch_depth = int(extra.get("patch_depth", extra.get("num_cond_blocks", total_depth)))
        pixel_depth = int(extra.get("pixel_depth", max(total_depth - patch_depth, 1)))
        num_text_blocks = int(extra.get("num_text_blocks", 4))
        txt_embed_dim = int(extra.get("txt_embed_dim", caption_channels))
        txt_max_length = int(extra.get("txt_max_length", model_max_length))
        use_text_rope = bool(extra.get("use_text_rope", True))
        text_rope_theta = float(extra.get("text_rope_theta", 10000.0))
        repa_encoder_index = int(extra.get("repa_encoder_index", -1))
        use_pixel_abs_pos = bool(extra.get("use_pixel_abs_pos", True))
        pit_adaln_post_modulation = bool(extra.get("pit_adaln_post_modulation", False))

        if PixDiT_T2I is None:
            raise ImportError("Failed to import PixDiT_T2I from pixdit_core.pixeldit_t2i. Check repo layout and PYTHONPATH.")
        self.core = PixDiT_T2I(
            in_channels=in_channels,
            num_groups=num_groups,
            hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size,
            pixel_attn_hidden_size=pixel_attn_hidden_size,
            pixel_num_groups=pixel_num_groups,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_text_blocks=num_text_blocks,
            patch_size=patch_size,
            txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length,
            use_text_rope=use_text_rope,
            text_rope_theta=text_rope_theta,
            repa_encoder_index=repa_encoder_index,
            use_pixel_abs_pos=use_pixel_abs_pos,
            pit_adaln_post_modulation=pit_adaln_post_modulation,
        )

        self.image_size = int(image_size)
        self.patch_size = patch_size
        self.pred_sigma = bool(pred_sigma)
        self.config = config
        self._txt_embed_dim = int(txt_embed_dim)
        if int(caption_channels) != self._txt_embed_dim:
            raise ValueError(f"caption_channels {caption_channels} != txt_embed_dim {self._txt_embed_dim}")

        projector_dim = 2048
        self._repa_projector = nn.Sequential(
            nn.Linear(self.core.hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, 768),
        )

    def forward(self, x, timestep, y, mask=None, data_info=None, repa_tokens=None, **kwargs):
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        if y.dim() == 4:
            y_proc = y.squeeze(1)
        elif y.dim() == 3:
            y_proc = y
        else:
            raise ValueError("PixDiTTrainer expects y of shape [B,1,L,C] or [B,L,C]")
        y_proc = y_proc.to(self.dtype)

        if y_proc.shape[-1] != self._txt_embed_dim:
            raise RuntimeError(
                f"PixDiTTrainer: text embedding dim {y_proc.shape[-1]} != expected {self._txt_embed_dim}. "
                f"Please set config.text_encoder.caption_channels to {self._txt_embed_dim} or set extra.txt_embed_dim to {y_proc.shape[-1]}."
            )

        if hasattr(self.core, "last_repa_tokens"):
            self.core.last_repa_tokens = None

        out = self.core(x, timestep, y_proc, s=None, mask=None)
        repa_loss = None
        if repa_tokens is not None:
            repa_tokens = repa_tokens.to(self.dtype)
            
        if repa_tokens is not None and getattr(self.core, "last_repa_tokens", None) is not None and self.core.last_repa_tokens is not None:
            proj_tokens = self._repa_projector(self.core.last_repa_tokens)  # [B, L, 768]
            proj_tokens = F.normalize(proj_tokens, dim=-1)
            B, Td, C = repa_tokens.shape
            Bu, Tu, Cu = proj_tokens.shape
            h_u = int(Tu ** 0.5)
            h_d = int(Td ** 0.5)
            if h_u * h_u == Tu and h_d * h_d == Td:
                if Td > Tu:
                    dino_2d = repa_tokens.permute(0, 2, 1).reshape(B, C, h_d, h_d)
                    dino_resized = F.interpolate(dino_2d, size=(h_u, h_u), mode="bilinear", align_corners=False)
                    dino_resized = dino_resized.flatten(2).permute(0, 2, 1)
                    dino_resized = F.normalize(dino_resized, dim=-1)
                    repa_loss = -((proj_tokens * dino_resized).sum(dim=-1)).mean()
                elif Td < Tu:
                    usit_2d = proj_tokens.permute(0, 2, 1).reshape(Bu, Cu, h_u, h_u)
                    usit_resized = F.interpolate(usit_2d, size=(h_d, h_d), mode="bilinear", align_corners=False)
                    usit_resized = usit_resized.flatten(2).permute(0, 2, 1)
                    usit_resized = F.normalize(usit_resized, dim=-1)
                    repa_tokens = F.normalize(repa_tokens, dim=-1)
                    repa_loss = -((usit_resized * repa_tokens).sum(dim=-1)).mean()
                else:
                    repa_tokens = F.normalize(repa_tokens, dim=-1)
                    repa_loss = -((proj_tokens * repa_tokens).sum(dim=-1)).mean()
        return {"x": out, "repa_loss": repa_loss}

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        out = self.forward(x, timestep, y, mask, **kwargs)
        if isinstance(out, dict):
            return out["x"]
        return out

    @property
    def dtype(self):
        return next(self.parameters()).dtype


