import torch

from models.pixeldit_core.pixeldit_c2i import PixDiT


def test_tiny_pixeldit_forward_shape():
    model = PixDiT(
        in_channels=3,
        num_groups=4,
        hidden_size=64,
        pixel_hidden_size=8,
        patch_depth=2,
        pixel_depth=1,
        patch_size=4,
        num_classes=10,
    ).eval()
    images = torch.randn(2, 3, 16, 16)
    time = torch.tensor([0.2, 0.8])
    labels = torch.tensor([1, 9])
    with torch.no_grad():
        output = model(images, time, labels)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()
