from argparse import Namespace

import torch

import denoiser as denoiser_module
from egfm import EnergyGuidedPath


class TinyVelocityNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, state, time, labels):
        del time, labels
        return self.projection(state)


def test_training_wrapper_uses_public_egfm_path(monkeypatch):
    monkeypatch.setattr(denoiser_module, "build_model", lambda *args, **kwargs: TinyVelocityNet())
    args = Namespace(
        model="PixDiT-XL/16",
        prediction_target="velocity",
        img_size=8,
        class_num=10,
        noise_scale=1.0,
        t_eps=0.05,
        label_drop_prob=0.1,
        t_sampler="lognormal",
        P_mean=0.0,
        P_std=1.0,
        timeshift=1.0,
        endpoint_low_prob=0.0,
        endpoint_high_prob=0.0,
        endpoint_width=0.1,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        attn_dropout=0.0,
        proj_dropout=0.0,
        freq_ab_sigma0=3.5,
        freq_ab_curve="smootherstep",
        freq_release_start_time=0.0,
        freq_release_time=1.0,
        sampling_method="flowdpm",
        num_sampling_steps=4,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
        repa_weight=0.0,
        repa_layer=1,
        repa_encoder="dinov2_vitb14",
    )
    wrapper = denoiser_module.Denoiser(args)
    assert isinstance(wrapper.path, EnergyGuidedPath)

    clean = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([1, 2])
    losses = wrapper(clean, labels)
    assert set(losses) == {"fm_loss", "loss"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert torch.isfinite(wrapper.net.projection.weight.grad).all()
