import math

import pytest
import torch

from egfm import EnergyGuidedPath


def _legacy_smootherstep(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clamped = u.clamp(0.0, 1.0)
    value = clamped.pow(3) * (
        clamped * (clamped * 6.0 - 15.0) + 10.0
    )
    derivative = 30.0 * clamped.square() * (1.0 - clamped).square()
    inside = (u > 0.0) & (u < 1.0)
    return value, torch.where(inside, derivative, torch.zeros_like(derivative))


def _legacy_forward(
    clean: torch.Tensor,
    t: torch.Tensor,
    *,
    sigma0: float,
    release_start: float,
    release_end: float,
    bisection_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = clean.shape[-2:]
    fy = torch.fft.fftfreq(height, device=clean.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=clean.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius2 = (
        torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(0.5)
    ).view(1, 1, height, width).square()

    t_view = t.float().reshape(-1, 1, 1, 1)
    duration = release_end - release_start
    release, release_speed = _legacy_smootherstep(
        (t_view - release_start) / duration
    )
    release_speed = release_speed / duration
    blur_scale = (math.pi * sigma0) ** 2
    clean_hat = torch.fft.fft2(clean.float(), dim=(-2, -1))
    energy = clean_hat.abs().square().sum(dim=1, keepdim=True)
    initial_response = torch.exp(-blur_scale * radius2)
    total_info = (energy * (1.0 - initial_response).square()).sum(
        dim=(-2, -1), keepdim=True
    )
    target_info = release * total_info

    lo = torch.zeros_like(t_view)
    hi = torch.ones_like(t_view)
    for _ in range(bisection_steps):
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


def test_path_is_bitwise_equal_to_legacy_algorithm_for_valid_inputs() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn(2, 3, 7, 6, generator=generator)
    time = torch.tensor([0.2, 0.8])
    kwargs = dict(
        sigma0=2.75,
        release_start=0.1,
        release_end=0.9,
        bisection_steps=9,
    )
    expected = _legacy_forward(clean, time, **kwargs)
    actual = EnergyGuidedPath(curve="smootherstep", **kwargs)(clean, time)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_endpoint_velocity_matches_central_finite_difference() -> None:
    generator = torch.Generator().manual_seed(9)
    clean = torch.randn(2, 3, 12, 10, generator=generator)
    time = torch.tensor([0.37, 0.68])
    delta = 1e-3
    path = EnergyGuidedPath(bisection_steps=24)

    _endpoint, analytical_speed = path(clean, time)
    endpoint_before, _ = path(clean, time - delta)
    endpoint_after, _ = path(clean, time + delta)
    numerical_speed = (endpoint_after - endpoint_before) / (2.0 * delta)

    torch.testing.assert_close(
        analytical_speed, numerical_speed, rtol=1e-2, atol=1e-3
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_path_preserves_shape_dtype_and_returns_finite_values(
    dtype: torch.dtype,
) -> None:
    clean = torch.randn(2, 2, 5, 7, dtype=dtype)
    endpoint, speed = EnergyGuidedPath(bisection_steps=4)(
        clean, torch.tensor([0.25, 0.75], dtype=torch.float64)
    )
    assert endpoint.shape == clean.shape
    assert speed.shape == clean.shape
    assert endpoint.dtype == dtype
    assert speed.dtype == dtype
    assert torch.isfinite(endpoint).all()
    assert torch.isfinite(speed).all()


def test_constant_image_has_zero_endpoint_velocity() -> None:
    clean = torch.ones(2, 3, 4, 4)
    endpoint, speed = EnergyGuidedPath()(clean, torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(endpoint, clean)
    torch.testing.assert_close(speed, torch.zeros_like(speed))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"sigma0": -1.0}, ValueError),
        ({"sigma0": float("nan")}, ValueError),
        ({"release_start": -0.1}, ValueError),
        ({"release_end": float("inf")}, ValueError),
        ({"release_start": 0.7, "release_end": 0.6}, ValueError),
        ({"bisection_steps": 0}, ValueError),
        ({"bisection_steps": 1.5}, TypeError),
        ({"curve": "missing"}, ValueError),
    ],
)
def test_constructor_validation(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        EnergyGuidedPath(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("clean", "time", "error"),
    [
        (torch.ones(2, 3, 4), torch.ones(2), ValueError),
        (torch.ones(0, 3, 4, 4), torch.ones(0), ValueError),
        (torch.ones(2, 3, 4, 4, dtype=torch.int64), torch.ones(2), TypeError),
        (torch.full((2, 3, 4, 4), float("nan")), torch.ones(2), ValueError),
        (torch.ones(2, 3, 4, 4), torch.ones(2, 1), ValueError),
        (torch.ones(2, 3, 4, 4), torch.ones(1), ValueError),
        (torch.ones(2, 3, 4, 4), torch.ones(2, dtype=torch.int64), TypeError),
        (
            torch.ones(2, 3, 4, 4),
            torch.tensor([0.2, float("inf")]),
            ValueError,
        ),
    ],
)
def test_forward_input_validation(
    clean: torch.Tensor,
    time: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        EnergyGuidedPath(bisection_steps=2)(clean, time)


def test_module_representation_contains_configuration() -> None:
    representation = repr(
        EnergyGuidedPath(
            sigma0=2.0,
            curve="linear",
            release_start=0.2,
            release_end=0.8,
            bisection_steps=3,
        )
    )
    assert "sigma0=2.0" in representation
    assert "curve=linear" in representation
    assert "release=[0.2, 0.8]" in representation
