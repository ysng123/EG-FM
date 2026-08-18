"""CAME optimizer used by the reference PixelDiT-T2I recipe."""

import torch


class CAME(torch.optim.Optimizer):
    def __init__(self, params, lr, eps=(1e-30, 1e-16), clip_threshold=1.0,
                 betas=(0.9, 0.999, 0.9999), weight_decay=0.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _factored(shape):
        return len(shape) == 2 or (len(shape) == 4 and shape[2:] == torch.Size([1, 1]))

    @staticmethod
    def _matrix(tensor):
        return tensor.squeeze(-1).squeeze(-1) if tensor.ndim == 4 else tensor

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / tensor.numel() ** 0.5

    @staticmethod
    def _approx(row, column):
        row_factor = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        return row_factor * column.unsqueeze(-2).rsqrt()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float() if parameter.grad.dtype in {torch.float16, torch.bfloat16} else parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients")
                state = self.state[parameter]
                factored = self._factored(gradient.shape)
                matrix = self._matrix(gradient)
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(gradient)
                    if factored:
                        state["sq_row"] = torch.zeros(matrix.shape[0], device=gradient.device, dtype=gradient.dtype)
                        state["sq_col"] = torch.zeros(matrix.shape[1], device=gradient.device, dtype=gradient.dtype)
                        state["res_row"] = torch.zeros_like(state["sq_row"])
                        state["res_col"] = torch.zeros_like(state["sq_col"])
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                state["step"] += 1
                squared = gradient.square().add(group["eps"][0])
                if factored:
                    squared_matrix = self._matrix(squared)
                    state["sq_row"].mul_(beta2).add_(squared_matrix.mean(1), alpha=1 - beta2)
                    state["sq_col"].mul_(beta2).add_(squared_matrix.mean(0), alpha=1 - beta2)
                    update = self._approx(state["sq_row"], state["sq_col"])
                    update = update.view_as(gradient).mul_(gradient)
                else:
                    state["exp_avg_sq"].mul_(beta2).add_(squared, alpha=1 - beta2)
                    update = state["exp_avg_sq"].rsqrt().mul_(gradient)
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_min_(1.0))
                average = state["exp_avg"]
                average.mul_(beta1).add_(update, alpha=1 - beta1)
                residual = (update - average).square().add(group["eps"][1])
                if factored:
                    residual_matrix = self._matrix(residual)
                    state["res_row"].mul_(beta3).add_(residual_matrix.mean(1), alpha=1 - beta3)
                    state["res_col"].mul_(beta3).add_(residual_matrix.mean(0), alpha=1 - beta3)
                    update = self._approx(state["res_row"], state["res_col"])
                    update = update.view_as(gradient).mul_(average)
                else:
                    update = average
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss


CAMEWrapper = CAME
__all__ = ["CAME", "CAMEWrapper"]
