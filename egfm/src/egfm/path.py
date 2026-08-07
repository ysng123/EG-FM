"""Energy-guided moving endpoint construction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ._validation import finite_nonnegative_scalar, normalize_time, validate_image
from .schedules import evaluate_release_schedule, get_release_schedule


class EnergyGuidedPath(nn.Module):
    """Construct a sample-adaptive heat endpoint and its endpoint velocity.

    Args:
        sigma0: Initial blur scale. Must be finite and non-negative.
        curve: Registered release-schedule name.
        release_start: Start of the release interval in normalized time.
        release_end: End of the release interval in normalized time.
        bisection_steps: Number of heat-time inversion iterations.
    """

    def __init__(
        self,
        sigma0: float = 3.5,
        curve: str = "smootherstep",
        release_start: float = 0.0,
        release_end: float = 1.0,
        bisection_steps: int = 16,
    ) -> None:
        super().__init__()
        self.sigma0 = finite_nonnegative_scalar(sigma0, "sigma0")
        get_release_schedule(curve)

        start = finite_nonnegative_scalar(release_start, "release_start")
        end = finite_nonnegative_scalar(release_end, "release_end")
        if not 0 <= start < end <= 1:
            raise ValueError("release times must satisfy 0 <= start < end <= 1")
        if isinstance(bisection_steps, bool) or not isinstance(bisection_steps, int):
            raise TypeError("bisection_steps must be an integer")
        if bisection_steps < 1:
            raise ValueError("bisection_steps must be positive")

        self.curve = curve
        self.release_start = start
        self.release_end = end
        self.bisection_steps = bisection_steps

    @staticmethod
    def _frequency_radius(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(0.5)
        return radius.view(1, 1, height, width)

    def forward(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the moving endpoint and its endpoint velocity.

        ``clean`` must be a finite floating BCHW tensor and ``t`` must be a
        finite floating vector of length ``batch``. Time is normalized to the
        image device and the float32 compute dtype before path evaluation.
        """
        clean = validate_image(clean, "clean")
        t_compute = normalize_time(t, clean, dtype=torch.float32)

        height, width = clean.shape[-2:]
        radius2 = self._frequency_radius(
            height, width, clean.device, torch.float32
        ).square()
        t_view = t_compute.reshape(-1, 1, 1, 1)
        duration = self.release_end - self.release_start
        release_u = (t_view - self.release_start) / duration
        release, release_speed = evaluate_release_schedule(release_u, self.curve)
        release_speed = release_speed / duration

        # Keep this operation order aligned with the original research release.
        blur_scale = (math.pi * self.sigma0) ** 2
        clean_hat = torch.fft.fft2(clean.float(), dim=(-2, -1))
        energy = clean_hat.abs().square().sum(dim=1, keepdim=True)
        initial_response = torch.exp(-blur_scale * radius2)
        total_info = (energy * (1.0 - initial_response).square()).sum(
            dim=(-2, -1), keepdim=True
        )
        target_info = release * total_info

        lo = torch.zeros_like(t_view)
        hi = torch.ones_like(t_view)
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            response_mid = torch.exp(-blur_scale * mid * radius2)
            info_mid = (energy * (response_mid - initial_response).square()).sum(
                dim=(-2, -1), keepdim=True
            )
            too_much = info_mid > target_info
            lo = torch.where(too_much, mid, lo)
            hi = torch.where(too_much, hi, mid)

        heat_time = 0.5 * (lo + hi)
        heat_time = torch.where(release <= 1e-6, 1.0, heat_time)
        heat_time = torch.where(release >= 1.0 - 1e-6, 0.0, heat_time)
        response = torch.exp(-blur_scale * heat_time * radius2)
        endpoint = torch.fft.ifft2(response * clean_hat, dim=(-2, -1)).real

        info_derivative = (
            -2.0
            * blur_scale
            * (energy * (response - initial_response) * radius2 * response).sum(
                dim=(-2, -1), keepdim=True
            )
        )
        valid = (
            (release_speed > 0)
            & (total_info > 1e-12)
            & (info_derivative.abs() > 1e-12)
        )
        safe_derivative = torch.where(valid, info_derivative, 1.0)
        heat_speed = torch.where(
            valid, release_speed * total_info / safe_derivative, 0.0
        )
        response_speed = -blur_scale * radius2 * response * heat_speed
        endpoint_velocity = torch.fft.ifft2(
            response_speed * clean_hat, dim=(-2, -1)
        ).real
        return endpoint.to(clean.dtype), endpoint_velocity.to(clean.dtype)

    def extra_repr(self) -> str:
        return (
            f"sigma0={self.sigma0}, curve={self.curve}, "
            f"release=[{self.release_start}, {self.release_end}], "
            f"bisection_steps={self.bisection_steps}"
        )
