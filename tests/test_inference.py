import argparse
from pathlib import Path

import pytest

from inference import parse_class_ids, resolve_settings


def _args(**overrides):
    values = {
        "checkpoint": Path("/path/to/checkpoint.pth"),
        "cfg": 2.55,
        "interval_min": 0.11,
        "interval_max": 0.975,
        "seed": 99985,
        "samples_per_class": 1,
        "batch_size": 1,
        "steps": 100,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_default_sampling_parameters():
    settings = resolve_settings(_args())
    assert settings["checkpoint"] == Path("/path/to/checkpoint.pth")
    assert settings["cfg"] == 2.55
    assert settings["interval_min"] == 0.11
    assert settings["interval_max"] == 0.975
    assert settings["seed"] == 99985


def test_checkpoint_is_required():
    with pytest.raises(ValueError, match="--checkpoint is required"):
        resolve_settings(_args(checkpoint=None))


def test_class_id_parser():
    assert parse_class_ids("0, 207,999") == [0, 207, 999]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_class_ids("1000")
