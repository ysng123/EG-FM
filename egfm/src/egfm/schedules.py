"""Release schedules used by energy-guided endpoint paths."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from ._validation import require_finite, require_floating_tensor


ReleaseSchedule = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

_SCHEDULES: dict[str, ReleaseSchedule] = {}


def register_release_schedule(
    name: str,
    schedule: ReleaseSchedule,
    *,
    overwrite: bool = False,
) -> None:
    """Register a schedule returning its value and derivative on ``[0, 1]``.

    A schedule receives a floating tensor and must return ``(value,
    derivative)`` tensors with the same shape, device, and dtype.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("schedule name must be a non-empty string")
    if not callable(schedule):
        raise TypeError("schedule must be callable")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    if name in _SCHEDULES and not overwrite:
        raise ValueError(f"Release schedule already registered: {name}")
    _SCHEDULES[name] = schedule


def get_release_schedule(name: str) -> ReleaseSchedule:
    """Return a registered release schedule by name."""
    if not isinstance(name, str):
        raise TypeError("schedule name must be a string")
    try:
        return _SCHEDULES[name]
    except KeyError as exc:
        available = ", ".join(list_release_schedules())
        raise ValueError(
            f"Unsupported release schedule: {name}. Available: {available}"
        ) from exc


def list_release_schedules() -> tuple[str, ...]:
    """List all available release schedules in deterministic order."""
    return tuple(sorted(_SCHEDULES))


def _validate_schedule_result(
    result: Any,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("schedule must return (value, derivative)")

    checked: list[torch.Tensor] = []
    for item, name in zip(result, ("schedule value", "schedule derivative")):
        tensor = require_floating_tensor(item, name)
        if tensor.shape != reference.shape:
            raise ValueError(f"{name} must have the same shape as its input")
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on the same device as its input")
        if tensor.dtype != reference.dtype:
            raise TypeError(f"{name} must have the same dtype as its input")
        require_finite(tensor, name)
        checked.append(tensor)
    return checked[0], checked[1]


def evaluate_release_schedule(
    u: torch.Tensor,
    schedule: str = "smootherstep",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a named schedule and derivative with boundary clamping.

    Values outside ``[0, 1]`` are clamped to the nearest endpoint and have a
    zero derivative. Inputs must be finite floating-point tensors.
    """
    u = require_floating_tensor(u, "u")
    require_finite(u, "u")
    clamped = u.clamp(0.0, 1.0)
    value, derivative = _validate_schedule_result(
        get_release_schedule(schedule)(clamped), clamped
    )
    inside = (u > 0.0) & (u < 1.0)
    return value, torch.where(inside, derivative, torch.zeros_like(derivative))


def _linear(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return u, torch.ones_like(u)


def _cosine(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.sin(0.5 * math.pi * u).square()
    derivative = 0.5 * math.pi * torch.sin(math.pi * u)
    return value, derivative


def _smoothstep(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = u.square() * (3.0 - 2.0 * u)
    derivative = 6.0 * u * (1.0 - u)
    return value, derivative


def _smootherstep(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = u.pow(3) * (u * (u * 6.0 - 15.0) + 10.0)
    derivative = 30.0 * u.square() * (1.0 - u).square()
    return value, derivative


register_release_schedule("linear", _linear)
register_release_schedule("cosine", _cosine)
register_release_schedule("smoothstep", _smoothstep)
register_release_schedule("smootherstep", _smootherstep)
