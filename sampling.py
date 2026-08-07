"""Shared ODE samplers operating on a velocity prediction callback."""

import torch


class Sampler:
    def __init__(self, method, steps, cfg, interval_min, interval_max, timeshift=1.0):
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        if method not in {"euler", "heun", "flowdpm"}:
            raise ValueError(f"Unsupported sampler: {method}")
        self.method = method
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.interval_min = float(interval_min)
        self.interval_max = float(interval_max)
        if timeshift <= 0:
            raise ValueError("timeshift must be positive")
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1)
        # The reference FlowDPM implementation uses linear time directly.
        if method != "flowdpm":
            timesteps = timesteps / (
                timesteps + (1.0 - timesteps) * timeshift
            )
        self.timesteps = timesteps

    def _guided_velocity(self, velocity_fn, x, t, labels, null_label):
        batch_size = x.shape[0]
        cfg_x = torch.cat([x, x], dim=0)
        cfg_t = t.repeat(2)
        cfg_labels = torch.cat([torch.full_like(labels, null_label), labels], dim=0)
        velocity = velocity_fn(cfg_x, cfg_t, cfg_labels)
        unconditional, conditional = velocity.chunk(2, dim=0)
        t_value = float(t[0])
        scale = self.cfg if self.interval_min < t_value < self.interval_max else 1.0
        return unconditional + scale * (conditional - unconditional)

    @torch.no_grad()
    def __call__(self, velocity_fn, noise, labels, null_label):
        if self.method == "flowdpm":
            return self._flowdpm(velocity_fn, noise, labels, null_label)
        return self._first_order(velocity_fn, noise, labels, null_label)

    def _first_order(self, velocity_fn, noise, labels, null_label):
        x = noise
        batch_size = x.shape[0]
        timesteps = self.timesteps.to(device=x.device, dtype=x.dtype)
        for index, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t = t_cur.repeat(batch_size)
            dt = t_next - t_cur
            velocity = self._guided_velocity(velocity_fn, x, t, labels, null_label)
            if self.method == "heun" and index < self.steps - 1:
                x_euler = x + dt * velocity
                velocity_next = self._guided_velocity(
                    velocity_fn, x_euler, t_next.repeat(batch_size), labels, null_label
                )
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
        return x

    def _flowdpm(self, velocity_fn, noise, labels, null_label):
        x = noise
        batch_size = x.shape[0]
        timesteps = self.timesteps.to(device=x.device, dtype=x.dtype)
        previous_velocity = None
        previous_dt = None
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            t = t_cur.repeat(batch_size)
            dt = t_next - t_cur
            velocity = self._guided_velocity(velocity_fn, x, t, labels, null_label)
            if previous_velocity is None:
                x = x + dt * velocity
            else:
                ratio = dt / previous_dt.clamp_min(torch.finfo(previous_dt.dtype).eps)
                x = x + dt * (
                    (1.0 + 0.5 * ratio) * velocity - 0.5 * ratio * previous_velocity
                )
            previous_velocity = velocity
            previous_dt = dt
        return x
