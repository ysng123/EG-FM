"""A complete, CPU-friendly EG-FM training step."""

import torch
import torch.nn as nn

from egfm import EnergyGuidedPath, make_training_batch


class TinyVelocityModel(nn.Module):
    """Small stand-in for a PixelDiT or another velocity denoiser."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
        )

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_bias = time[:, None, None, None].to(state)
        return self.network(state + time_bias)


def main() -> None:
    torch.manual_seed(0)
    images = torch.randn(2, 3, 16, 16)
    time = torch.rand(images.shape[0])

    path = EnergyGuidedPath(sigma0=3.5, curve="smootherstep")
    flow = make_training_batch(images, time, path)

    model = TinyVelocityModel(channels=images.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prediction = model(flow.state, flow.time)
    loss = (prediction - flow.velocity).square().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
