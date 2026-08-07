"""One EG-FM wrapper for every model and prediction parameterization."""

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# A source checkout runs against its bundled library without an editable
# install. If this file is moved elsewhere, an installed egfm remains usable.
local_egfm_src = Path(__file__).resolve().parent / "egfm" / "src"
if local_egfm_src.is_dir():
    sys.path.insert(0, str(local_egfm_src))

from egfm import (  # noqa: E402
    EnergyGuidedPath,
    make_training_batch,
    prediction_to_velocity,
)
from models import build_model
from repa import DINOv2
from sampling import Sampler


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model
        self.prediction_target = args.prediction_target
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.noise_scale = args.noise_scale
        self.t_eps = args.t_eps
        self.label_drop_prob = args.label_drop_prob
        self.t_sampler = args.t_sampler
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.timeshift = args.timeshift
        self.endpoint_low_prob = args.endpoint_low_prob
        self.endpoint_high_prob = args.endpoint_high_prob
        self.endpoint_width = args.endpoint_width
        if self.P_std < 0:
            raise ValueError("P_std must be non-negative")
        if self.timeshift <= 0:
            raise ValueError("timeshift must be positive")
        if self.endpoint_low_prob < 0 or self.endpoint_high_prob < 0:
            raise ValueError("endpoint probabilities must be non-negative")
        if self.endpoint_low_prob + self.endpoint_high_prob >= 1:
            raise ValueError("endpoint probabilities must sum to less than one")
        if not 0 < self.endpoint_width <= 1:
            raise ValueError("endpoint_width must be in (0, 1]")
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.net = build_model(
            args.model,
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.path = EnergyGuidedPath(
            sigma0=args.freq_ab_sigma0,
            curve=args.freq_ab_curve,
            release_start=args.freq_release_start_time,
            release_end=args.freq_release_time,
        )
        self.sampler = Sampler(
            method=args.sampling_method,
            steps=args.num_sampling_steps,
            cfg=args.cfg,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            timeshift=args.timeshift,
        )
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        self.repa_weight = float(args.repa_weight)
        self.repa_layer = int(args.repa_layer)
        if self.repa_weight > 0:
            feature_dim = int(getattr(self.net, "hidden_size"))
            self.repa_encoder = DINOv2(args.repa_encoder)
            self.repa_proj = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, 768),
            )
        else:
            self.repa_encoder = None
            self.repa_proj = None

    def _sample_t(self, batch_size, device):
        if self.t_sampler == "lognormal":
            t = (
                torch.randn(batch_size, device=device) * self.P_std + self.P_mean
            ).sigmoid()
        elif self.t_sampler == "uniform":
            t = torch.rand(batch_size, device=device)
        else:
            raise ValueError(f"Unsupported timestep distribution: {self.t_sampler}")

        if self.endpoint_low_prob or self.endpoint_high_prob:
            selector = torch.rand(batch_size, device=device)
            endpoint_u = torch.rand(batch_size, device=device)
            t = torch.where(
                selector < self.endpoint_low_prob,
                self.endpoint_width * endpoint_u,
                t,
            )
            high = (selector >= self.endpoint_low_prob) & (
                selector < self.endpoint_low_prob + self.endpoint_high_prob
            )
            t = torch.where(high, 1.0 - self.endpoint_width * endpoint_u, t)
        return t / (t + (1.0 - t) * self.timeshift)

    def _drop_labels(self, labels):
        if not self.training or self.label_drop_prob <= 0:
            return labels
        dropped = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(dropped, torch.full_like(labels, self.num_classes), labels)

    def _prediction_to_velocity(self, prediction, current, t):
        return prediction_to_velocity(
            prediction,
            current,
            t,
            self.path,
            target=self.prediction_target,
            eps=self.t_eps,
        )

    def predict_velocity(self, current, t, labels):
        prediction = self.net(current, t, labels)
        return self._prediction_to_velocity(prediction, current, t)

    def _feature_blocks(self):
        if hasattr(self.net, "patch_blocks"):
            return self.net.patch_blocks
        if hasattr(self.net, "blocks"):
            return self.net.blocks
        raise ValueError(f"Model {self.model_name} does not expose feature blocks for REPA")

    def _repa_loss(self, feature, raw_images):
        patch_size = int(getattr(self.net, "patch_size", 16))
        expected_tokens = (self.img_size // patch_size) ** 2
        if feature.shape[1] > expected_tokens:
            feature = feature[:, -expected_tokens:]
        feature = self.repa_proj(feature)
        with torch.no_grad():
            target = self.repa_encoder(raw_images)
        if feature.shape[1] != target.shape[1]:
            batch_size, source_tokens, channels = feature.shape
            source_hw = int(math.sqrt(source_tokens))
            target_hw = int(math.sqrt(target.shape[1]))
            if source_hw * source_hw != source_tokens or target_hw * target_hw != target.shape[1]:
                raise ValueError("REPA features must form square token grids")
            feature = feature.view(batch_size, source_hw, source_hw, channels).permute(0, 3, 1, 2)
            feature = F.interpolate(
                feature, size=(target_hw, target_hw), mode="bilinear", align_corners=False
            )
            feature = feature.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        return (1.0 - F.cosine_similarity(feature, target, dim=-1)).mean()

    def forward(self, clean, labels, raw_images=None):
        t = self._sample_t(clean.shape[0], clean.device).to(clean.dtype)
        flow = make_training_batch(
            clean, t, self.path, noise_scale=self.noise_scale
        )
        train_labels = self._drop_labels(labels)

        captured = []
        handle = None
        if self.repa_weight > 0:
            blocks = self._feature_blocks()
            if not 1 <= self.repa_layer <= len(blocks):
                raise ValueError(f"repa_layer must be in [1, {len(blocks)}]")
            handle = blocks[self.repa_layer - 1].register_forward_hook(
                lambda _module, _inputs, output: captured.append(
                    output[0] if isinstance(output, tuple) else output
                )
            )
        try:
            prediction = self.net(flow.state, flow.time, train_labels)
        finally:
            if handle is not None:
                handle.remove()

        velocity_prediction = self._prediction_to_velocity(
            prediction, flow.state, flow.time
        )
        fm_loss = (velocity_prediction - flow.velocity).square().mean()
        output = {"fm_loss": fm_loss, "loss": fm_loss}
        if self.repa_weight > 0:
            if not captured:
                raise RuntimeError("REPA feature hook did not capture an activation")
            if raw_images is None:
                raw_images = ((clean + 1.0) / 2.0).clamp(0.0, 1.0)
            repa_loss = self._repa_loss(captured[0], raw_images)
            output["repa_loss"] = repa_loss
            output["loss"] = fm_loss + self.repa_weight * repa_loss
        return output

    @torch.no_grad()
    def generate(self, labels):
        noise = self.noise_scale * torch.randn(
            labels.shape[0], 3, self.img_size, self.img_size, device=labels.device
        )
        return self.sampler(self.predict_velocity, noise, labels, self.num_classes)

    def _net_named_parameters(self):
        return list(self.net.named_parameters())

    def init_ema_from_current(self):
        params = [parameter for _, parameter in self._net_named_parameters()]
        self.ema_params1 = [parameter.detach().clone() for parameter in params]
        self.ema_params2 = [parameter.detach().clone() for parameter in params]

    def _ema_state_dict(self, ema_params):
        state = copy.deepcopy(self.net.state_dict())
        for (name, _), parameter in zip(self._net_named_parameters(), ema_params):
            state[name] = parameter.detach().clone()
        return state

    def capture_model_state(self):
        return copy.deepcopy(self.net.state_dict())

    def load_model_state(self, state):
        self.net.load_state_dict(state)

    def load_ema_model_state(self, which=1):
        params = self.ema_params1 if which == 1 else self.ema_params2
        self.load_model_state(self._ema_state_dict(params))

    @torch.no_grad()
    def update_ema(self):
        source = [parameter for _, parameter in self._net_named_parameters()]
        for target, parameter in zip(self.ema_params1, source):
            target.mul_(self.ema_decay1).add_(parameter, alpha=1.0 - self.ema_decay1)
        for target, parameter in zip(self.ema_params2, source):
            target.mul_(self.ema_decay2).add_(parameter, alpha=1.0 - self.ema_decay2)

    def build_training_checkpoint(self, args, optimizer, epoch):
        checkpoint = {
            "format_version": 2,
            "model_name": self.model_name,
            "prediction_target": self.prediction_target,
            "model": self.net.state_dict(),
            "ema_model": self._ema_state_dict(self.ema_params1),
            "ema_model_2": self._ema_state_dict(self.ema_params2),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": dict(vars(args)),
        }
        if self.repa_proj is not None:
            checkpoint["repa_proj"] = self.repa_proj.state_dict()
        return checkpoint

    @staticmethod
    def _normalize_state(state):
        if state and all(name.startswith("net.") for name in state):
            return {name[4:]: value for name, value in state.items()}
        return state

    def load_training_checkpoint(self, checkpoint, device):
        model_state = self._normalize_state(checkpoint["model"])
        self.net.load_state_dict(model_state)
        if self.repa_proj is not None and "repa_proj" in checkpoint:
            self.repa_proj.load_state_dict(checkpoint["repa_proj"])

        ema1 = checkpoint.get("ema_model") or checkpoint.get("model_ema1")
        ema2 = checkpoint.get("ema_model_2") or checkpoint.get("model_ema2") or ema1
        if ema1 is None:
            raise KeyError("Checkpoint does not contain EMA weights")
        ema1 = self._normalize_state(ema1)
        ema2 = self._normalize_state(ema2)
        self.ema_params1 = [ema1[name].to(device) for name, _ in self._net_named_parameters()]
        self.ema_params2 = [ema2[name].to(device) for name, _ in self._net_named_parameters()]
