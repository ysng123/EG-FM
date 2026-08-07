# EG-FM integration examples

These integration examples preserve the layouts and launch entry points of their GitHub upstreams. The code changes are limited to using the standalone [`egfm`](../egfm) package for flow construction and velocity targets; the READMEs document that integration.

| Example | Upstream | Scope |
| --- | --- | --- |
| [`jit`](jit) | [`LTH14/JiT`](https://github.com/LTH14/JiT) at `cbc743a` | JiT ImageNet training and generation |
| [`pixeldit`](pixeldit) | [`NVlabs/PixelDiT`](https://github.com/NVlabs/PixelDiT) at `41f7300` | The complete PixelDiT repository, including C2I and T2I |

Install each upstream environment as documented in its README, then install EG-FM from the repository root with `python -m pip install -e ./egfm`. No separate C2I copy or additional multi-node launcher is maintained here; use PixelDiT's original `c2i/` and `t2i/` entry points.
