from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .modules import (
    ClassEmbedder,
    FeedForward,
    FinalLayer,
    MLP,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    precompute_freqs_cis_2d,
)


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class AugmentedDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache = {}

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        key = ("image", height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device=device, dtype=dtype)
        if height == width:
            pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, height)
        else:
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)
        pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
        self._pos_cache[key] = pos
        return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None):
        if inputs.dim() != 4:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B,C,H,W]")
        if img_height is None or img_width is None or patch_size is None:
            raise ValueError("img_height, img_width, and patch_size must be provided")
        bsz, _channels, height, width = inputs.shape
        assert height == img_height and width == img_width
        assert (height % patch_size == 0) and (width % patch_size == 0)
        hs, ws = height // patch_size, width // patch_size
        pixels_per_patch = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(height, width, inputs.device, inputs.dtype)
            x = x + pos_full.view(height, width, self.hidden_size_output).unsqueeze(0)
        x = x.view(bsz, hs, patch_size, ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(bsz * hs * ws, pixels_per_patch, self.hidden_size_output)


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_fn=None,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        assert self.attn_dim % self.num_heads == 0
        pixels_per_patch = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(pixels_per_patch * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, pixels_per_patch * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, 6 * self.pixel_dim * pixels_per_patch, bias=True))
        self._pos_cache = {}
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = self._rope_fn(self.attn_dim // self.num_heads, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(self, x: torch.Tensor, s_cond: torch.Tensor, image_height: int, image_width: int, patch_size: int, mask=None):
        bl, pixels_per_patch, channels = x.shape
        if channels != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {channels}")
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0)
        hs, ws = image_height // patch_size, image_width // patch_size
        length = hs * ws
        batch_size = bl // length
        cond_params = self.adaLN_modulation(s_cond).view(bl, pixels_per_patch, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(bl, pixels_per_patch * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(batch_size, length, self.attn_dim)
        pos_comp = self._fetch_pos(hs, ws, x.device)
        attn_out = self.attn(x_comp, pos_comp, mask)
        attn_flat = self.expand_from_attn(attn_out.view(batch_size * length, self.attn_dim))
        x = x + gate_msa * attn_flat.view(bl, pixels_per_patch, self.pixel_dim)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PixDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_groups=12,
        hidden_size=1152,
        pixel_hidden_size=64,
        patch_depth=18,
        pixel_depth=4,
        patch_size=2,
        num_classes=1000,
        use_pixel_abs_pos=True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT expects pixel_depth > 0")

        self.pixel_embedder = PixelTokenEmbedder(self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size ** 2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)
        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)
        self.patch_blocks = nn.ModuleList([AugmentedDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.patch_depth)])
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(
                self.pixel_hidden_size,
                self.hidden_size,
                patch_size=self.patch_size,
                num_heads=self.num_groups,
                mlp_ratio=4.0,
            )
            for _ in range(self.pixel_depth)
        ])
        self.precompute_pos = {}
        self.initialize_weights()

    def fetch_pos(self, height, width, device):
        key = (height, width)
        if key in self.precompute_pos:
            return self.precompute_pos[key].to(device)
        pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
        self.precompute_pos[key] = pos
        return pos

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    def forward(self, x, t, y, s=None, mask=None):
        batch_size, _channels, height, width = x.shape
        pos = self.fetch_pos(height // self.patch_size, width // self.patch_size, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(t.view(-1)).view(batch_size, -1, self.hidden_size)
        y_emb = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        c = nn.functional.silu(t_emb + y_emb)
        if s is None:
            s = self.s_embedder(x_patches)
            for block in self.patch_blocks:
                s = block(s, c, pos, mask)
            s = nn.functional.silu(t_emb + s)
        batch_size, length, _ = s.shape
        s_cond = s.view(batch_size * length, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=height, img_width=width, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, height, width, self.patch_size, mask)
        x_pixels = self.final_layer(x_pixels)
        channels_out = self.out_channels
        pixels_per_patch = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(batch_size, length, pixels_per_patch, channels_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(batch_size, channels_out * pixels_per_patch, length)
        return torch.nn.functional.fold(x_pixels, (height, width), kernel_size=self.patch_size, stride=self.patch_size)
