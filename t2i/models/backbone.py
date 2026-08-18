"""PixelDiT text-to-image backbone with joint image-text attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pixeldit_core.modules import (
    FeedForward,
    FinalLayer,
    MLP,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)
from models.pixeldit_core.pixeldit_c2i import PatchTokenEmbedder, PixelTokenEmbedder


class JointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.qkv_x = nn.Linear(dim, 3 * dim, bias=False)
        self.qkv_y = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

    def _qkv(self, projection, tokens, q_norm, k_norm):
        batch, length, channels = tokens.shape
        qkv = projection(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 1, 3, 4)
        return q_norm(qkv[0]), k_norm(qkv[1]), qkv[2]

    def forward(self, x, y, pos_img, pos_txt=None, attn_mask=None):
        batch, image_length, channels = x.shape
        if y.shape[0] != batch or y.shape[2] != channels:
            raise ValueError("image and text tokens must share batch and channel dimensions")
        text_length = y.shape[1]
        qx, kx, vx = self._qkv(self.qkv_x, x, self.q_norm_x, self.k_norm_x)
        qy, ky, vy = self._qkv(self.qkv_y, y, self.q_norm_y, self.k_norm_y)
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        query = torch.cat([qy.transpose(1, 2), qx.transpose(1, 2)], dim=2)
        key = torch.cat([ky.transpose(1, 2), kx.transpose(1, 2)], dim=2)
        value = torch.cat([vy.transpose(1, 2), vx.transpose(1, 2)], dim=2)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0
        )
        output_y = output[:, :, :text_length].transpose(1, 2).reshape(
            batch, text_length, channels
        )
        output_x = output[:, :, text_length:].transpose(1, 2).reshape(
            batch, image_length, channels
        )
        return self.proj_x(output_x), self.proj_y(output_y)


class JointBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = JointAttention(hidden_size, num_heads)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden)
        self.adaLN_modulation_img = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_txt = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, y, condition, pos_img, pos_txt=None, attn_mask=None):
        x_params = self.adaLN_modulation_img(condition).chunk(6, dim=-1)
        y_params = self.adaLN_modulation_txt(condition).chunk(6, dim=-1)
        shift_xa, scale_xa, gate_xa, shift_xm, scale_xm, gate_xm = x_params
        shift_ya, scale_ya, gate_ya, shift_ym, scale_ym, gate_ym = y_params
        attn_x, attn_y = self.attn(
            apply_adaln(self.norm_x1(x), shift_xa, scale_xa),
            apply_adaln(self.norm_y1(y), shift_ya, scale_ya),
            pos_img,
            pos_txt,
            attn_mask,
        )
        x = x + gate_xa * attn_x
        y = y + gate_ya * attn_y
        x = x + gate_xm * self.mlp_x(
            apply_adaln(self.norm_x2(x), shift_xm, scale_xm)
        )
        y = y + gate_ym * self.mlp_y(
            apply_adaln(self.norm_y2(y), shift_ym, scale_ym)
        )
        return x, y


class PixelBlock(nn.Module):
    """Pixel branch used by the released T2I checkpoints."""

    def __init__(
        self,
        pixel_dim: int,
        context_dim: int,
        patch_size: int,
        attn_dim: int,
        num_heads: int,
        post_modulation: bool = False,
    ):
        super().__init__()
        if attn_dim % num_heads:
            raise ValueError("pixel attention size must be divisible by its heads")
        self.pixel_dim = int(pixel_dim)
        self.context_dim = int(context_dim)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.post_modulation = bool(post_modulation)
        pixels_per_patch = self.patch_size**2
        self.compress_to_attn = nn.Linear(
            pixels_per_patch * self.pixel_dim, self.attn_dim, bias=True
        )
        self.expand_from_attn = nn.Linear(
            self.attn_dim, pixels_per_patch * self.pixel_dim, bias=True
        )
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=4.0, drop=0.0)
        chunks = 4 if self.post_modulation else 6
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.context_dim, chunks * self.pixel_dim * pixels_per_patch)
        )
        self._pos_cache = {}

    def _position(self, height, width, device):
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = precompute_freqs_cis_2d(
                self.attn_dim // self.num_heads, height, width
            )
        return self._pos_cache[key].to(device)

    def forward(self, x, condition, image_height, image_width):
        patch_h = image_height // self.patch_size
        patch_w = image_width // self.patch_size
        patch_count = patch_h * patch_w
        batch = x.shape[0] // patch_count
        pixels_per_patch = self.patch_size**2
        chunks = 4 if self.post_modulation else 6
        params = self.adaLN_modulation(condition).view(
            batch * patch_count, pixels_per_patch, chunks * self.pixel_dim
        )
        if self.post_modulation:
            scale_attn, shift_attn, scale_mlp, shift_mlp = params.chunk(4, dim=-1)
            normalized = self.norm1(x)
        else:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                params.chunk(6, dim=-1)
            )
            normalized = apply_adaln(self.norm1(x), shift_attn, scale_attn)
        compressed = self.compress_to_attn(
            normalized.reshape(batch * patch_count, -1)
        ).view(batch, patch_count, self.attn_dim)
        attended = self.attn(compressed, self._position(patch_h, patch_w, x.device))
        attended = self.expand_from_attn(attended.reshape(batch * patch_count, -1)).view_as(x)
        if self.post_modulation:
            x = x + attended * (1.0 + scale_attn) + shift_attn
            return x + self.mlp(self.norm2(x)) * (1.0 + scale_mlp) + shift_mlp
        x = x + gate_attn * attended
        return x + gate_mlp * self.mlp(
            apply_adaln(self.norm2(x), shift_mlp, scale_mlp)
        )


class PixelDiTT2I(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        num_groups: int = 24,
        hidden_size: int = 1536,
        pixel_hidden_size: int = 16,
        pixel_attn_hidden_size: int = 1152,
        pixel_num_groups: int = 16,
        patch_depth: int = 14,
        pixel_depth: int = 2,
        num_text_blocks: int = 4,
        text_dim: int = 2304,
        text_max_length: int = 300,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_layer: int = 6,
        pixel_post_modulation: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.num_groups = int(num_groups)
        self.hidden_size = int(hidden_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.text_dim = int(text_dim)
        self.text_max_length = int(text_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_layer = int(repa_layer)

        self.pixel_embedder = PixelTokenEmbedder(
            self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=True
        )
        self.s_embedder = PatchTokenEmbedder(
            self.in_channels * self.patch_size**2, self.hidden_size, bias=True
        )
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = PatchTokenEmbedder(
            self.text_dim, self.hidden_size, norm_layer=RMSNorm, bias=True
        )
        self.y_pos_embedding = nn.Parameter(
            torch.randn(1, self.text_max_length, self.hidden_size)
        )
        self.patch_blocks = nn.ModuleList(
            [JointBlock(self.hidden_size, self.num_groups) for _ in range(self.patch_depth)]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                PixelBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    self.patch_size,
                    pixel_attn_hidden_size,
                    pixel_num_groups,
                    post_modulation=pixel_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)
        self._image_pos_cache = {}
        self._text_pos_cache = {}
        self.last_repa_tokens = None
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.s_embedder.proj.weight.view(self.hidden_size, -1))
        nn.init.zeros_(self.s_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _image_position(self, height, width, device):
        key = (height, width)
        if key not in self._image_pos_cache:
            self._image_pos_cache[key] = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, height, width
            )
        return self._image_pos_cache[key].to(device)

    def _text_position(self, length, device):
        if length not in self._text_pos_cache:
            head_dim = self.hidden_size // self.num_groups
            frequencies = 1.0 / (
                self.text_rope_theta
                ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
            )
            angles = torch.arange(length, device=device).float().unsqueeze(1) * frequencies
            self._text_pos_cache[length] = torch.polar(torch.ones_like(angles), angles)
        return self._text_pos_cache[length].to(device)

    @staticmethod
    def _joint_attention_mask(mask, batch, text_tokens, image_tokens, device):
        if mask is None:
            return None
        mask = mask.to(device=device)
        while mask.dim() > 2 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.dim() != 2:
            raise ValueError("text mask must have shape [B, L]")
        allowed_text = mask[:, :text_tokens].bool()
        allowed_image = torch.ones(batch, image_tokens, dtype=torch.bool, device=device)
        return torch.cat([allowed_text, allowed_image], dim=1).view(
            batch, 1, 1, text_tokens + image_tokens
        )

    def forward(self, x, t, text, mask=None):
        batch, _channels, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if text.dim() != 3 or text.shape[0] != batch or text.shape[2] != self.text_dim:
            raise ValueError(f"text must have shape [B, L, {self.text_dim}]")
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        image_tokens = patch_h * patch_w
        text_tokens = min(text.shape[1], self.text_max_length)
        position_image = self._image_position(patch_h, patch_w, x.device)
        position_text = (
            self._text_position(text_tokens, x.device) if self.use_text_rope else None
        )
        patches = F.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2)
        time = self.t_embedder(t.reshape(-1)).view(batch, 1, self.hidden_size)
        condition = F.silu(time)
        image_state = self.s_embedder(patches)
        text_state = self.y_embedder(text[:, :text_tokens]).view(
            batch, text_tokens, self.hidden_size
        )
        text_state = text_state + self.y_pos_embedding[:, :text_tokens].to(text_state.dtype)
        attention_mask = self._joint_attention_mask(
            mask, batch, text_tokens, image_tokens, x.device
        )
        self.last_repa_tokens = None
        for index, block in enumerate(self.patch_blocks, start=1):
            image_state, text_state = block(
                image_state,
                text_state,
                condition,
                position_image,
                position_text,
                attention_mask,
            )
            if index == self.repa_layer:
                self.last_repa_tokens = image_state
        if self.last_repa_tokens is None:
            self.last_repa_tokens = image_state
        image_state = F.silu(time + image_state)
        pixel_condition = image_state.reshape(batch * image_tokens, self.hidden_size)
        pixels = self.pixel_embedder(
            x, img_height=height, img_width=width, patch_size=self.patch_size
        )
        for block in self.pixel_blocks:
            pixels = block(pixels, pixel_condition, height, width)
        pixels = self.final_layer(pixels)
        pixels_per_patch = self.patch_size**2
        pixels = pixels.view(batch, image_tokens, pixels_per_patch, self.out_channels)
        pixels = pixels.permute(0, 3, 2, 1).reshape(
            batch, self.out_channels * pixels_per_patch, image_tokens
        )
        return F.fold(pixels, (height, width), self.patch_size, stride=self.patch_size)


# Reference checkpoint/config spelling.
PixDiT_T2I = PixelDiTT2I

__all__ = ["PixelDiTT2I", "PixDiT_T2I"]
