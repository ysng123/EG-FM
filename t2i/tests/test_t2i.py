from argparse import Namespace

import torch

from t2i.models import PixelDiTT2I, T2IDenoiser
from t2i.train import apply_stage_defaults


def tiny_args(**overrides):
    values = dict(
        img_size=8,
        patch_size=2,
        text_dim=12,
        text_max_length=5,
        text_drop_prob=0.0,
        noise_scale=1.0,
        flow_shift=3.0,
        train_sampling_steps=16,
        feat_loss_weight=0.0,
        freq_sigma_release=True,
        ema_decay1=0.9,
        ema_decay2=0.8,
        cfg=2.0,
        interval_min=0.0,
        interval_max=1.0,
        num_sampling_steps=2,
        num_groups=4,
        hidden_size=32,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=32,
        pixel_num_groups=4,
        patch_depth=2,
        pixel_depth=1,
        num_text_blocks=1,
        use_text_rope=True,
        text_rope_theta=10_000.0,
        repa_encoder_index=1,
        pit_adaln_post_modulation=False,
        repa_encoder="unused",
        freq_ab_sigma0=3.5,
        freq_ab_curve="smootherstep",
        freq_release_start_time=0.0,
        freq_release_time=1.0,
    )
    values.update(overrides)
    return Namespace(**values)


def test_t2i_backbone_preserves_image_shape_and_uses_mask():
    model = PixelDiTT2I(
        patch_size=2,
        num_groups=4,
        hidden_size=32,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=32,
        pixel_num_groups=4,
        patch_depth=2,
        pixel_depth=1,
        text_dim=12,
        text_max_length=5,
        repa_layer=1,
    )
    output = model(
        torch.randn(2, 3, 8, 8),
        torch.tensor([10.0, 20.0]),
        torch.randn(2, 5, 12),
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
    )
    assert output.shape == (2, 3, 8, 8)
    assert model.last_repa_tokens.shape == (2, 16, 32)


def test_egfm_t2i_forward_and_sampling_are_finite():
    denoiser = T2IDenoiser(tiny_args())
    clean = torch.randn(2, 3, 8, 8).clamp(-1, 1)
    text = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5, dtype=torch.long)
    losses = denoiser(clean, text, mask, torch.zeros_like(text), mask)
    assert set(losses) == {"fm_loss", "repa_loss", "loss"}
    assert torch.isfinite(losses["loss"])
    sample = denoiser.generate(text, torch.zeros_like(text), mask, mask)
    assert sample.shape == clean.shape
    assert torch.isfinite(sample).all()


def test_reference_512_stage_defaults():
    args = Namespace(
        stage="pretrain512",
        img_size=None,
        batch_size=None,
        max_train_steps=None,
        gradient_accumulation_steps=None,
        feat_loss_weight=None,
        dataset_backend="auto",
    )
    args = apply_stage_defaults(args)
    assert args.img_size == 512
    assert args.batch_size == 48
    assert args.max_train_steps == 100_000
    assert args.gradient_accumulation_steps == 1
    assert args.dataset_backend == "tar"
