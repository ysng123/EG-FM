from collections import OrderedDict

import torch

from checkpoint import extract_model_state, load_inference_weights


def _filled_state(model, value):
    return OrderedDict(
        (name, torch.full_like(tensor, value))
        for name, tensor in model.state_dict().items()
    )


def test_extracts_legacy_ema_and_strips_net_prefix():
    state = {"net.weight": torch.ones(2, 2), "net.bias": torch.zeros(2)}
    extracted = extract_model_state({"model_ema1": state}, "ema1")
    assert set(extracted) == {"weight", "bias"}


def test_loads_full_training_checkpoint(tmp_path):
    source = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "training.pth"
    torch.save(
        {
            "backend": "pixeldit",
            "model_name": "PixDiT-XL/16",
            "epoch": 200,
            "model": _filled_state(source, 1.0),
            "ema_model": _filled_state(source, 2.0),
        },
        checkpoint_path,
    )
    target = torch.nn.Linear(2, 2)
    metadata = load_inference_weights(target, checkpoint_path, mmap=True)
    assert metadata["epoch"] == 200
    assert all(torch.equal(value, torch.full_like(value, 2.0)) for value in target.state_dict().values())


def test_rejects_bare_state_dict():
    with torch.no_grad():
        state = torch.nn.Linear(2, 2).state_dict()
    try:
        extract_model_state(state)
    except ValueError as error:
        assert "full PixelDiT checkpoint" in str(error)
    else:
        raise AssertionError("bare state dict should not be accepted")
