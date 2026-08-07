"""Training-pair construction and prediction-target conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from ._validation import (
    finite_nonnegative_scalar,
    normalize_time,
    require_finite,
    validate_image,
    validate_matching_image,
    validate_path,
    validate_path_result,
)


PredictionTarget = Literal["velocity", "x"]


class EndpointPath(Protocol):
    """Structural interface implemented by endpoint path callables."""

    def __call__(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class EnergyFlowBatch:
    """All path quantities needed by a flow-matching training step."""

    time: torch.Tensor
    state: torch.Tensor
    velocity: torch.Tensor
    endpoint: torch.Tensor
    endpoint_velocity: torch.Tensor
    noise: torch.Tensor


def make_training_batch(
    clean: torch.Tensor,
    t: torch.Tensor,
    path: EndpointPath,
    *,
    noise: torch.Tensor | None = None,
    noise_scale: float = 1.0,
) -> EnergyFlowBatch:
    """Build noisy states and exact velocity targets for an endpoint path.

    A supplied ``noise`` tensor must exactly match ``clean`` in shape, device,
    and dtype. ``noise_scale`` applies only when noise is generated internally.
    """
    clean = validate_image(clean, "clean")
    time = normalize_time(t, clean)
    path = validate_path(path)
    scale = finite_nonnegative_scalar(noise_scale, "noise_scale")

    if noise is None:
        scale_in_clean_dtype = torch.as_tensor(
            scale, dtype=clean.dtype, device=clean.device
        )
        require_finite(scale_in_clean_dtype, "noise_scale")
        noise = torch.randn_like(clean) * scale
    noise = validate_matching_image(noise, clean, "noise")

    endpoint, endpoint_velocity = validate_path_result(
        path(clean, time), clean
    )
    t_view = time.reshape(-1, 1, 1, 1)
    state = t_view * endpoint + (1.0 - t_view) * noise
    velocity = endpoint + t_view * endpoint_velocity - noise
    return EnergyFlowBatch(
        time=time,
        state=state,
        velocity=velocity,
        endpoint=endpoint,
        endpoint_velocity=endpoint_velocity,
        noise=noise,
    )


def prediction_to_velocity(
    prediction: torch.Tensor,
    state: torch.Tensor,
    t: torch.Tensor,
    path: EndpointPath,
    *,
    target: PredictionTarget = "velocity",
    eps: float = 5e-2,
) -> torch.Tensor:
    """Convert a velocity or x-prediction network output to path velocity."""
    prediction = validate_image(prediction, "prediction")
    validate_matching_image(state, prediction, "state")

    if target == "velocity":
        return prediction
    if target != "x":
        raise ValueError(f"Unsupported prediction target: {target}")

    eps_value = finite_nonnegative_scalar(eps, "eps")
    if eps_value == 0:
        raise ValueError("eps must be positive")
    time = normalize_time(t, prediction)
    path = validate_path(path)
    endpoint, endpoint_velocity = validate_path_result(
        path(prediction, time), prediction
    )
    t_view = time.reshape(-1, 1, 1, 1)
    return t_view * endpoint_velocity + (endpoint - state) / (
        1.0 - t_view
    ).clamp_min(eps_value)
