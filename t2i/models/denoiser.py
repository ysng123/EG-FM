"""EG-FM training objective, EMA handling, and FlowDPM sampling for T2I."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_EGFM = PROJECT_ROOT / "egfm" / "src"
if LOCAL_EGFM.is_dir() and str(LOCAL_EGFM) not in sys.path:
    sys.path.insert(0, str(LOCAL_EGFM))

from egfm import EnergyGuidedPath, make_training_batch  # noqa: E402
from repa import DINOv2  # noqa: E402
from t2i.models.backbone import PixelDiTT2I  # noqa: E402


class T2IDenoiser(nn.Module):
    """Checkpoint-compatible PixelDiT-T2I wrapper.

    The network predicts derivatives with respect to ``sigma`` (noise at 1,
    data at 0). The bundled EG-FM library uses forward time (noise at 0, data
    at 1), so its exact velocity target is negated during training.
    """

    def __init__(self, args):
        super().__init__()
        self.img_size = int(args.img_size)
        self.patch_size = int(args.patch_size)
        self.text_dim = int(args.text_dim)
        self.text_max_length = int(args.text_max_length)
        self.text_drop_prob = float(args.text_drop_prob)
        self.noise_scale = float(args.noise_scale)
        self.flow_shift = float(args.flow_shift)
        self.train_sampling_steps = int(args.train_sampling_steps)
        self.feat_loss_weight = float(args.feat_loss_weight)
        self.energy_guided = bool(args.freq_sigma_release)
        if self.flow_shift <= 0:
            raise ValueError("flow_shift must be positive")
        if self.train_sampling_steps < 2:
            raise ValueError("train_sampling_steps must be at least 2")

        self.ema_decay1 = float(args.ema_decay1)
        self.ema_decay2 = float(args.ema_decay2)
        self.cfg_scale = float(args.cfg)
        self.cfg_interval = (float(args.interval_min), float(args.interval_max))
        if not 0 <= self.cfg_interval[0] < self.cfg_interval[1] <= 1:
            raise ValueError("CFG interval must satisfy 0 <= min < max <= 1")
        self.steps = int(args.num_sampling_steps)
        if self.steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        self.ema_params1 = None
        self.ema_params2 = None

        self.net = PixelDiTT2I(
            in_channels=3,
            patch_size=self.patch_size,
            num_groups=int(args.num_groups),
            hidden_size=int(args.hidden_size),
            pixel_hidden_size=int(args.pixel_hidden_size),
            pixel_attn_hidden_size=int(args.pixel_attn_hidden_size),
            pixel_num_groups=int(args.pixel_num_groups),
            patch_depth=int(args.patch_depth),
            pixel_depth=int(args.pixel_depth),
            num_text_blocks=int(args.num_text_blocks),
            text_dim=self.text_dim,
            text_max_length=self.text_max_length,
            use_text_rope=bool(args.use_text_rope),
            text_rope_theta=float(args.text_rope_theta),
            repa_layer=int(args.repa_encoder_index),
            pixel_post_modulation=bool(args.pit_adaln_post_modulation),
        )
        hidden_size = int(args.hidden_size)
        # Always construct this projection to preserve reference checkpoint keys.
        self.repa_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 768),
        )
        self.repa_encoder = (
            DINOv2(model_name=args.repa_encoder, base_patch_size=self.patch_size)
            if self.feat_loss_weight > 0
            else None
        )
        self.path = EnergyGuidedPath(
            sigma0=float(args.freq_ab_sigma0),
            curve=str(args.freq_ab_curve),
            release_start=float(args.freq_release_start_time),
            release_end=float(args.freq_release_time),
        )

        # Match the discrete timestep convention of the released T2I code.
        sigma = torch.linspace(0.0, 0.999, self.train_sampling_steps)
        self.register_buffer("train_sigmas", self._shift_sigma(sigma), persistent=False)

    def _shift_sigma(self, sigma):
        return self.flow_shift * sigma / (1.0 + (self.flow_shift - 1.0) * sigma)

    def _net_named_parameters(self):
        return list(self.net.named_parameters())

    def _sample_sigma(self, batch_size, device):
        u = torch.randn(batch_size, device=device, dtype=torch.float32).sigmoid()
        index = (u * self.train_sampling_steps).long().clamp_(
            0, self.train_sampling_steps - 1
        )
        sigma = self.train_sigmas[index]
        return sigma, sigma * self.train_sampling_steps

    def _drop_text(self, text, mask, null_text, null_mask):
        if not self.training or self.text_drop_prob <= 0:
            return text, mask
        dropped = (torch.rand(text.shape[0], device=text.device) < self.text_drop_prob)
        text = torch.where(
            dropped[:, None, None], null_text.to(text).expand_as(text), text
        )
        if mask is not None and null_mask is not None:
            mask = torch.where(
                dropped[:, None], null_mask.to(mask).expand_as(mask), mask
            )
        return text, mask

    def _repa_loss(self, raw_images):
        if self.repa_encoder is None:
            return raw_images.new_zeros(())
        source = self.net.last_repa_tokens
        if source is None:
            raise RuntimeError("REPA layer did not expose image tokens")
        source = self.repa_proj(source)
        with torch.no_grad():
            target = self.repa_encoder(raw_images).to(source.dtype)
        if source.shape[1] != target.shape[1]:
            batch, source_len, channels = source.shape
            source_hw = math.isqrt(source_len)
            target_hw = math.isqrt(target.shape[1])
            if source_hw**2 != source_len or target_hw**2 != target.shape[1]:
                raise ValueError("REPA features must form square token grids")
            source = source.view(batch, source_hw, source_hw, channels).permute(0, 3, 1, 2)
            source = F.interpolate(
                source, size=(target_hw, target_hw), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).reshape(batch, -1, channels)
        return 1.0 - F.cosine_similarity(source, target, dim=-1).mean()

    def forward(
        self,
        clean,
        text_embeds,
        text_mask,
        null_text_embeds,
        null_text_mask,
        raw_images=None,
    ):
        text_embeds, text_mask = self._drop_text(
            text_embeds, text_mask, null_text_embeds, null_text_mask
        )
        sigma, model_time = self._sample_sigma(clean.shape[0], clean.device)
        sigma = sigma.to(clean.dtype)
        model_time = model_time.to(clean.dtype)
        noise = torch.randn_like(clean) * self.noise_scale
        if self.energy_guided:
            forward_time = 1.0 - sigma
            flow = make_training_batch(clean, forward_time, self.path, noise=noise)
            state, target = flow.state, -flow.velocity
        else:
            sigma_view = sigma[:, None, None, None]
            state = (1.0 - sigma_view) * clean + sigma_view * noise
            target = noise - clean
        prediction = self.net(state, model_time, text_embeds, mask=text_mask)
        fm_loss = (prediction - target).square().mean()
        if raw_images is None:
            raw_images = ((clean + 1.0) / 2.0).clamp(0.0, 1.0)
        repa_loss = self._repa_loss(raw_images)
        return {
            "fm_loss": fm_loss,
            "repa_loss": repa_loss,
            "loss": fm_loss + self.feat_loss_weight * repa_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        text_embeds,
        null_text_embeds,
        text_mask=None,
        null_text_mask=None,
        generator=None,
    ):
        device = text_embeds.device
        batch_size = text_embeds.shape[0]
        current = torch.randn(
            batch_size,
            3,
            self.img_size,
            self.img_size,
            device=device,
            dtype=text_embeds.dtype,
            generator=generator,
        ) * self.noise_scale
        sigmas = self._shift_sigma(
            torch.linspace(1.0, 0.0, self.steps + 1, device=device, dtype=current.dtype)
        )
        previous_velocity = previous_dt = None
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            dt = sigma_next - sigma
            model_time = (sigma * self.train_sampling_steps).expand(batch_size)
            cfg_state = torch.cat([current, current])
            cfg_time = torch.cat([model_time, model_time])
            cfg_text = torch.cat([null_text_embeds.to(text_embeds), text_embeds])
            cfg_mask = None
            if text_mask is not None and null_text_mask is not None:
                cfg_mask = torch.cat([null_text_mask, text_mask])
            unconditional, conditional = self.net(
                cfg_state, cfg_time, cfg_text, mask=cfg_mask
            ).chunk(2)
            guidance = (
                self.cfg_scale
                if self.cfg_interval[0] < float(sigma) < self.cfg_interval[1]
                else 1.0
            )
            velocity = unconditional + guidance * (conditional - unconditional)
            if previous_velocity is None:
                current = current + dt * velocity
            else:
                safe_previous_dt = previous_dt.sign() * previous_dt.abs().clamp_min(
                    torch.finfo(previous_dt.dtype).eps
                )
                ratio = dt / safe_previous_dt
                current = current + dt * (
                    (1.0 + 0.5 * ratio) * velocity
                    - 0.5 * ratio * previous_velocity
                )
            previous_velocity, previous_dt = velocity, dt
        return current

    def init_ema_from_current(self):
        parameters = [parameter for _, parameter in self._net_named_parameters()]
        self.ema_params1 = [parameter.detach().clone() for parameter in parameters]
        self.ema_params2 = [parameter.detach().clone() for parameter in parameters]

    def _ema_state_dict(self, parameters):
        state = copy.deepcopy(self.net.state_dict())
        for (name, _), parameter in zip(self._net_named_parameters(), parameters):
            state[name] = parameter.detach().clone()
        return state

    def capture_model_state(self):
        return copy.deepcopy(self.net.state_dict())

    def load_model_state(self, state):
        self.net.load_state_dict(state)

    def load_ema_model_state(self, which=1):
        parameters = self.ema_params1 if which == 1 else self.ema_params2
        if parameters is None:
            raise RuntimeError("EMA parameters have not been initialized")
        self.load_model_state(self._ema_state_dict(parameters))

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params1 is None or self.ema_params2 is None:
            return
        source = [parameter for _, parameter in self._net_named_parameters()]
        for target, parameter in zip(self.ema_params1, source):
            target.mul_(self.ema_decay1).add_(parameter, alpha=1.0 - self.ema_decay1)
        for target, parameter in zip(self.ema_params2, source):
            target.mul_(self.ema_decay2).add_(parameter, alpha=1.0 - self.ema_decay2)

    def build_training_checkpoint(self, args, optimizer, epoch):
        return {
            "format_version": 2,
            "backend": "pixeldit_t2i",
            "model_name": "PixDiT-T2I",
            "prediction_target": "sigma_velocity",
            "model": self.net.state_dict(),
            "repa_proj": self.repa_proj.state_dict(),
            "ema_model": self._ema_state_dict(self.ema_params1),
            "ema_model_2": self._ema_state_dict(self.ema_params2),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": int(getattr(args, "global_step", epoch)),
            "args": dict(vars(args)),
        }

    def load_training_checkpoint(self, checkpoint, device):
        backend = checkpoint.get("backend", "pixeldit_t2i")
        if backend != "pixeldit_t2i":
            raise ValueError(f"Checkpoint backend {backend!r} does not match pixeldit_t2i")
        self.net.load_state_dict(checkpoint["model"])
        if "repa_proj" in checkpoint:
            self.repa_proj.load_state_dict(checkpoint["repa_proj"])
        ema1 = checkpoint.get("ema_model") or checkpoint.get("model_ema1") or checkpoint["model"]
        ema2 = checkpoint.get("ema_model_2") or checkpoint.get("model_ema2") or ema1
        self.ema_params1 = [ema1[name].to(device) for name, _ in self._net_named_parameters()]
        self.ema_params2 = [ema2[name].to(device) for name, _ in self._net_named_parameters()]


__all__ = ["T2IDenoiser"]
