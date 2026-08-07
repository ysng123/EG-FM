import math

import pytest
import torch

from egfm import (
    evaluate_release_schedule,
    get_release_schedule,
    list_release_schedules,
    register_release_schedule,
)


def test_builtin_schedule_registry_is_sorted_and_complete() -> None:
    names = list_release_schedules()
    assert names == tuple(sorted(names))
    assert {"linear", "cosine", "smoothstep", "smootherstep"} <= set(names)
    assert callable(get_release_schedule("smootherstep"))


@pytest.mark.parametrize(
    ("name", "expected_value", "expected_derivative"),
    [
        ("linear", 0.25, 1.0),
        ("cosine", 0.5 - math.sqrt(2.0) / 4.0, math.pi / (2.0**1.5)),
        ("smoothstep", 0.15625, 1.125),
        ("smootherstep", 0.103515625, 1.0546875),
    ],
)
def test_builtin_schedule_values(
    name: str,
    expected_value: float,
    expected_derivative: float,
) -> None:
    value, derivative = evaluate_release_schedule(
        torch.tensor([0.25], dtype=torch.float64), name
    )
    assert value.dtype == torch.float64
    assert derivative.dtype == torch.float64
    assert value.item() == pytest.approx(expected_value)
    assert derivative.item() == pytest.approx(expected_derivative)


def test_schedule_clamps_values_and_zeros_boundary_derivatives() -> None:
    u = torch.tensor([-0.2, 0.0, 0.5, 1.0, 1.2])
    value, derivative = evaluate_release_schedule(u, "linear")
    torch.testing.assert_close(value, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    torch.testing.assert_close(
        derivative, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    )


def test_custom_schedule_can_be_registered() -> None:
    def quadratic(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return u.square(), 2.0 * u

    register_release_schedule("_test_quadratic", quadratic, overwrite=True)
    value, derivative = evaluate_release_schedule(
        torch.tensor([0.2, 0.7]), "_test_quadratic"
    )
    torch.testing.assert_close(value, torch.tensor([0.04, 0.49]))
    torch.testing.assert_close(derivative, torch.tensor([0.4, 1.4]))


def test_registry_rejects_invalid_and_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_release_schedule("", lambda u: (u, u))
    with pytest.raises(TypeError, match="callable"):
        register_release_schedule("_test_not_callable", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        register_release_schedule(
            "_test_bad_overwrite", lambda u: (u, u), overwrite=1  # type: ignore[arg-type]
        )

    register_release_schedule("_test_duplicate", lambda u: (u, u), overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_release_schedule("_test_duplicate", lambda u: (u, u))
    with pytest.raises(ValueError, match="Unsupported"):
        get_release_schedule("_test_missing")
    with pytest.raises(TypeError, match="string"):
        get_release_schedule(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_input, error",
    [
        (torch.tensor([1]), TypeError),
        (torch.tensor([float("nan")]), ValueError),
        (torch.tensor([float("inf")]), ValueError),
    ],
)
def test_evaluate_rejects_invalid_input(
    bad_input: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        evaluate_release_schedule(bad_input)


@pytest.mark.parametrize(
    ("name", "schedule", "error"),
    [
        ("_test_list_result", lambda u: [u, u], TypeError),
        ("_test_shape_result", lambda u: (u[:1], u), ValueError),
        ("_test_dtype_result", lambda u: (u.double(), u.double()), TypeError),
        (
            "_test_nonfinite_result",
            lambda u: (torch.full_like(u, float("nan")), u),
            ValueError,
        ),
    ],
)
def test_custom_schedule_result_is_validated(
    name: str,
    schedule: object,
    error: type[Exception],
) -> None:
    register_release_schedule(name, schedule, overwrite=True)  # type: ignore[arg-type]
    with pytest.raises(error):
        evaluate_release_schedule(torch.tensor([0.2, 0.4]), name)
