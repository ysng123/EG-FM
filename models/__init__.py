"""PixelDiT model registry used by the EG-FM implementation."""

from .pixeldit_core.pixeldit_c2i import PixDiT


def _pixeldit_xl_16(**kwargs):
    # These arguments belong to the JiT-compatible factory interface.  The
    # released PixelDiT architecture has fixed image-channel and dropout
    # choices, so accepting and discarding them preserves checkpoint tooling.
    kwargs.pop("input_size", None)
    kwargs.pop("attn_drop", None)
    kwargs.pop("proj_drop", None)
    kwargs.pop("in_channels", None)
    num_classes = kwargs.pop("num_classes", 1000)
    return PixDiT(
        in_channels=3,
        patch_size=16,
        num_groups=16,
        hidden_size=1152,
        patch_depth=26,
        pixel_depth=4,
        pixel_hidden_size=16,
        num_classes=num_classes,
        **kwargs,
    )


MODEL_REGISTRY = {"PixDiT-XL/16": _pixeldit_xl_16}


def build_model(name, **kwargs):
    """Build a registered PixelDiT model."""
    if name not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available models: {choices}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = ["MODEL_REGISTRY", "PixDiT", "build_model"]
