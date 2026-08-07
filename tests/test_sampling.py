import pytest
import torch

from sampling import Sampler


@pytest.mark.parametrize("method", ["euler", "heun", "flowdpm"])
def test_constant_velocity_is_integrated_exactly(method):
    sampler = Sampler(method, steps=4, cfg=1.0, interval_min=0.0, interval_max=1.0)
    noise = torch.zeros(2, 1, 2, 2)
    labels = torch.tensor([1, 2])

    def velocity(current, time, class_labels):
        del time, class_labels
        return torch.ones_like(current) * 2.0

    result = sampler(velocity, noise, labels, null_label=3)
    torch.testing.assert_close(result, torch.full_like(result, 2.0))


def test_invalid_sampler_configuration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        Sampler("euler", steps=0, cfg=1, interval_min=0, interval_max=1)
    with pytest.raises(ValueError, match="Unsupported"):
        Sampler("unknown", steps=1, cfg=1, interval_min=0, interval_max=1)
