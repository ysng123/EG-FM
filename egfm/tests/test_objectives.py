from collections.abc import Callable

import pytest
import torch

from egfm import EnergyFlowBatch, make_training_batch, prediction_to_velocity


class AffineEndpoint:
    def __init__(self) -> None:
        self.seen_time: torch.Tensor | None = None

    def __call__(
        self, clean: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_time = time
        return 2.0 * clean, torch.full_like(clean, 3.0)


def test_make_training_batch_matches_closed_form_and_normalizes_time() -> None:
    clean = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2)
    time = torch.tensor([0.25, 0.75], dtype=torch.float32)
    noise = torch.full_like(clean, -2.0)
    path = AffineEndpoint()

    batch = make_training_batch(clean, time, path, noise=noise)
    assert isinstance(batch, EnergyFlowBatch)
    assert batch.time.dtype == clean.dtype
    assert batch.time.device == clean.device
    assert path.seen_time is not None
    assert path.seen_time.dtype == clean.dtype

    t_view = batch.time[:, None, None, None]
    expected_endpoint = 2.0 * clean
    expected_endpoint_velocity = torch.full_like(clean, 3.0)
    expected_state = t_view * expected_endpoint + (1.0 - t_view) * noise
    expected_velocity = expected_endpoint + t_view * expected_endpoint_velocity - noise
    assert torch.equal(batch.endpoint, expected_endpoint)
    assert torch.equal(batch.endpoint_velocity, expected_endpoint_velocity)
    assert torch.equal(batch.state, expected_state)
    assert torch.equal(batch.velocity, expected_velocity)
    assert batch.noise is noise


def test_generated_noise_scale_zero_is_exactly_zero() -> None:
    clean = torch.randn(2, 1, 3, 3)
    batch = make_training_batch(
        clean, torch.tensor([0.2, 0.8]), AffineEndpoint(), noise_scale=0.0
    )
    assert torch.count_nonzero(batch.noise) == 0


@pytest.mark.parametrize(
    ("noise", "error"),
    [
        (torch.ones(1, 1, 2, 2), ValueError),
        (torch.ones(2, 1, 2, 2, dtype=torch.float64), TypeError),
        (torch.full((2, 1, 2, 2), float("nan")), ValueError),
        (torch.ones(2, 1, 2, 2, dtype=torch.int64), TypeError),
    ],
)
def test_explicit_noise_is_validated(
    noise: torch.Tensor,
    error: type[Exception],
) -> None:
    clean = torch.ones(2, 1, 2, 2)
    with pytest.raises(error):
        make_training_batch(clean, torch.ones(2), AffineEndpoint(), noise=noise)


@pytest.mark.parametrize("scale", [-0.1, float("nan"), float("inf")])
def test_noise_scale_is_finite_and_nonnegative(scale: float) -> None:
    clean = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError):
        make_training_batch(clean, torch.ones(1), AffineEndpoint(), noise_scale=scale)


def test_noise_scale_must_be_representable_in_image_dtype() -> None:
    clean = torch.ones(1, 1, 2, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        make_training_batch(
            clean, torch.ones(1), AffineEndpoint(), noise_scale=1e100
        )


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda clean: clean, TypeError),
        (lambda clean: (clean,), TypeError),
        (lambda clean: (clean[:, :, :1], clean[:, :, :1]), ValueError),
        (lambda clean: (clean.double(), clean.double()), TypeError),
        (
            lambda clean: (torch.full_like(clean, float("nan")), clean),
            ValueError,
        ),
    ],
)
def test_custom_path_result_is_validated(
    factory: Callable[[torch.Tensor], object],
    error: type[Exception],
) -> None:
    clean = torch.ones(2, 1, 2, 2)

    def bad_path(
        image: torch.Tensor, time: torch.Tensor
    ) -> object:
        del time
        return factory(image)

    with pytest.raises(error):
        make_training_batch(
            clean,
            torch.tensor([0.2, 0.8]),
            bad_path,  # type: ignore[arg-type]
            noise=torch.zeros_like(clean),
        )


def test_prediction_to_velocity_is_identity_without_calling_path() -> None:
    prediction = torch.randn(2, 1, 2, 2)
    state = torch.randn_like(prediction)

    def fail_if_called(*args: object) -> object:
        raise AssertionError("velocity prediction must not evaluate the path")

    actual = prediction_to_velocity(
        prediction,
        state,
        torch.tensor([0.2, 0.8]),
        fail_if_called,  # type: ignore[arg-type]
        target="velocity",
    )
    assert actual is prediction


def test_clean_prediction_conversion_matches_closed_form() -> None:
    prediction = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2)
    state = torch.full_like(prediction, 0.5)
    time = torch.tensor([0.25, 0.9], dtype=torch.float32)
    path = AffineEndpoint()
    actual = prediction_to_velocity(
        prediction, state, time, path, target="x", eps=0.05
    )
    normalized_time = time.to(torch.float64)
    t_view = normalized_time[:, None, None, None]
    expected = t_view * 3.0 + (2.0 * prediction - state) / (
        1.0 - t_view
    ).clamp_min(0.05)
    assert torch.equal(actual, expected)
    assert path.seen_time is not None
    assert path.seen_time.dtype == prediction.dtype


def test_prediction_conversion_validation() -> None:
    prediction = torch.ones(2, 1, 2, 2)
    state = torch.zeros_like(prediction)
    time = torch.tensor([0.2, 0.8])
    path = AffineEndpoint()

    with pytest.raises(ValueError, match="Unsupported"):
        prediction_to_velocity(
            prediction, state, time, path, target="noise"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive"):
        prediction_to_velocity(prediction, state, time, path, target="x", eps=0)
    with pytest.raises(ValueError, match="finite"):
        prediction_to_velocity(
            prediction, state, time, path, target="x", eps=float("nan")
        )
    with pytest.raises(ValueError, match="shape"):
        prediction_to_velocity(prediction, state[:1], time, path)
    with pytest.raises(TypeError, match="same dtype"):
        prediction_to_velocity(prediction, state.double(), time, path)
    with pytest.raises(TypeError, match="floating-point"):
        prediction_to_velocity(
            prediction, state, time.long(), path, target="x"
        )
