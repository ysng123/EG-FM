"""Internal validation helpers shared by the public EG-FM API."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch


def require_floating_tensor(value: Any, name: str) -> torch.Tensor:
    """Return ``value`` after checking the common tensor contract."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype")
    return value


def require_finite(value: torch.Tensor, name: str) -> None:
    """Reject NaN and infinity without synchronizing normal CUDA execution."""
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda" and hasattr(torch, "_assert_async"):
        # ``bool(finite)`` copies the reduction result to the host, introducing
        # a synchronization and a torch.compile graph break on every training
        # call. The device-side assertion reports the same invalid input while
        # allowing valid CUDA workloads to remain asynchronous.
        torch._assert_async(finite, f"{name} must contain only finite values")
        return
    if not bool(finite):
        raise ValueError(f"{name} must contain only finite values")


def validate_image(value: Any, name: str) -> torch.Tensor:
    """Validate a non-empty floating image batch in BCHW layout."""
    tensor = require_floating_tensor(value, name)
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [batch, channels, height, width]"
        )
    if any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must not have an empty dimension")
    require_finite(tensor, name)
    return tensor


def normalize_time(
    value: Any,
    reference: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Validate time as ``[batch]`` and place it alongside ``reference``."""
    time = require_floating_tensor(value, "t")
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError("t must have shape [batch]")
    require_finite(time, "t")
    normalized = time.to(
        device=reference.device,
        dtype=reference.dtype if dtype is None else dtype,
    )
    if normalized is not time:
        # A finite high-precision value can overflow when normalized to a
        # lower-precision reference dtype.
        require_finite(normalized, "t")
    return normalized


def validate_matching_image(
    value: Any,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate an image tensor that must exactly match ``reference``."""
    tensor = validate_image(value, name)
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} and reference must have identical shapes")
    if tensor.device != reference.device:
        raise ValueError(f"{name} and reference must be on the same device")
    if tensor.dtype != reference.dtype:
        raise TypeError(f"{name} and reference must have the same dtype")
    return tensor


def validate_path(path: Any) -> Callable[..., Any]:
    """Validate the structural path interface before invoking it."""
    if not callable(path):
        raise TypeError("path must be callable")
    return path


def validate_path_result(
    result: Any,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the two image tensors returned by an endpoint path."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("path must return (endpoint, endpoint_velocity)")
    endpoint = validate_matching_image(result[0], reference, "endpoint")
    endpoint_velocity = validate_matching_image(
        result[1], reference, "endpoint_velocity"
    )
    return endpoint, endpoint_velocity


def finite_nonnegative_scalar(value: Any, name: str) -> float:
    """Normalize a public real-valued option and validate its domain."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized
