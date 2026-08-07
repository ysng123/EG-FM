"""Checkpoint loading helpers for PixelDiT models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch


WeightSet = Literal["ema1", "ema2", "model"]

_WEIGHT_KEYS: dict[WeightSet, tuple[str, ...]] = {
    "ema1": ("ema_model", "model_ema1"),
    "ema2": ("ema_model_2", "model_ema2", "ema_model", "model_ema1"),
    "model": ("model", "state_dict"),
}


def load_checkpoint_file(path: str | Path, *, mmap: bool = True):
    """Load a trusted full training checkpoint on CPU.

    Full training checkpoints may contain Python metadata and therefore
    require ``weights_only=False`` on PyTorch versions where weights-only
    loading is the default.  Only call this function for a trusted artifact.
    ``mmap`` substantially lowers peak host RAM for large training files.
    """

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(checkpoint_path, **kwargs)
    except TypeError:
        # PyTorch < 2.1 has no mmap argument; PyTorch < 2.0 has no
        # weights_only argument.  Keep compatibility without masking real
        # deserialization failures.
        kwargs.pop("mmap", None)
        try:
            return torch.load(checkpoint_path, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return torch.load(checkpoint_path, **kwargs)


def _is_state_dict(value) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(name, str) for name in value)
        and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
    )


def _strip_wrapper_prefixes(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(state)
    for prefix in ("module.net.", "module.", "net."):
        if normalized and all(name.startswith(prefix) for name in normalized):
            normalized = {name[len(prefix) :]: value for name, value in normalized.items()}
            break
    return normalized


def extract_model_state(checkpoint, which: WeightSet = "ema1") -> dict[str, torch.Tensor]:
    """Extract and normalize one model state from a full checkpoint."""

    if which not in _WEIGHT_KEYS:
        raise ValueError(f"Unsupported weight set: {which}")
    if _is_state_dict(checkpoint):
        raise ValueError("Expected a full PixelDiT checkpoint, not a bare state dict")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping or a raw model state dict")

    for key in _WEIGHT_KEYS[which]:
        state = checkpoint.get(key)
        if _is_state_dict(state):
            return _strip_wrapper_prefixes(state)
    available = ", ".join(str(key) for key in checkpoint.keys())
    expected = ", ".join(_WEIGHT_KEYS[which])
    raise KeyError(
        f"Checkpoint has no usable {which} weights (expected one of: {expected}). "
        f"Available top-level keys: {available}"
    )


def checkpoint_metadata(checkpoint) -> dict[str, object]:
    """Return small, JSON-friendly metadata without retaining tensor state."""

    if not isinstance(checkpoint, Mapping):
        return {}
    metadata: dict[str, object] = {}
    for key in (
        "backend",
        "model_name",
        "prediction_target",
        "epoch",
    ):
        value = checkpoint.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                metadata[key] = value
    return metadata


def load_inference_weights(
    model: torch.nn.Module,
    path: str | Path,
    *,
    which: WeightSet = "ema1",
    strict: bool = True,
    mmap: bool = True,
) -> dict[str, object]:
    """Load selected weights into ``model`` and release the checkpoint mapping."""

    checkpoint = load_checkpoint_file(path, mmap=mmap)
    metadata = checkpoint_metadata(checkpoint)
    state = extract_model_state(checkpoint, which=which)
    model.load_state_dict(state, strict=strict)
    del state
    del checkpoint
    return metadata


__all__ = [
    "WeightSet",
    "checkpoint_metadata",
    "extract_model_state",
    "load_checkpoint_file",
    "load_inference_weights",
]
