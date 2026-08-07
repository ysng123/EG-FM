# Modified from https://github.com/MCG-NJU/PixNerd

import json
import logging
import os
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import lightning.pytorch as pl
import lightning.pytorch.loggers.wandb as wandb
import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info

setattr(wandb, "_WANDB_AVAILABLE", True)
torch.set_float32_matmul_precision("medium")
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)


class SimpleEMA(Callback):
    def __init__(self, decay: float = 0.9999, every_n_steps: int = 1):
        super().__init__()
        self.decay = decay
        self.every_n_steps = every_n_steps
        self._stream = torch.cuda.Stream()
        self.previous_step = 0
        self.net_params = []
        self.ema_params = []

    def setup_models(self, net: torch.nn.Module, ema_net: torch.nn.Module):
        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())

    def ema_step(self):
        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            torch._foreach_mul_(ema_model_tuple, decay)
            torch._foreach_add_(ema_model_tuple, current_model_tuple, alpha=(1.0 - decay))

        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            ema_update(self.ema_params, self.net_params, self.decay)
        assert self.ema_params[0].dtype == torch.float32

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step == self.previous_step:
            return
        self.previous_step = trainer.global_step
        if trainer.global_step % self.every_n_steps == 0:
            self.ema_step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]


class CheckpointHook(ModelCheckpoint):
    """Save checkpoint with only the incremental part of the model."""

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.dirpath = trainer.default_root_dir
        self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
        pl_module.strict_loading = False

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint.pop("callbacks", None)


class SaveImagesHook(Callback):
    def __init__(self, save_dir: str = "val", save_compressed: bool = False, predict_tag: str = "predict"):
        super().__init__()
        self.save_dir = save_dir
        self.save_compressed = save_compressed
        self.predict_tag = predict_tag
        self.samples = []
        self.target_dir = None
        self.executor_pool = None
        self._saved_num = 0

    def save_start(self, target_dir: str):
        self.samples = []
        self.target_dir = target_dir
        self.executor_pool = ThreadPoolExecutor(max_workers=8)

        if os.path.exists(self.target_dir) and os.listdir(self.target_dir):
            if "debug" not in str(target_dir):
                base_dir = self.target_dir
                resume_count = 1
                while os.path.exists(self.target_dir) and os.listdir(self.target_dir):
                    self.target_dir = f"{base_dir}_resume_{resume_count}"
                    resume_count += 1
                rank_zero_info(f"Directory exists, using new path: {self.target_dir}")

        os.makedirs(self.target_dir, exist_ok=True)
        rank_zero_info(f"Save images to {self.target_dir}")
        self._saved_num = 0

    def save_image(self, trainer, pl_module, images, metadatas):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        for sample, metadata in zip(images, metadatas):
            save_fn = metadata.pop("save_fn", None)
            if save_fn:
                self.executor_pool.submit(save_fn, sample, metadata, self.target_dir)

    def process_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        samples: STEP_OUTPUT,
        batch: Any,
    ) -> None:
        xT, y, metadata = batch
        b, c, h, w = samples.shape
        if not self.save_compressed or self._saved_num < 1:
            self._saved_num += b
            self.save_image(trainer, pl_module, samples, metadata)

        all_samples = pl_module.all_gather(samples).view(-1, c, h, w)
        if trainer.is_global_zero:
            all_samples = all_samples.permute(0, 2, 3, 1).cpu().numpy()
            self.samples.append(all_samples)

    def save_end(self):
        if self.save_compressed and len(self.samples) > 0:
            samples = np.concatenate(self.samples)
            np.savez(f"{self.target_dir}/output.npz", arr_0=samples)
        if self.executor_pool:
            self.executor_pool.shutdown(wait=True)
        self.target_dir = None
        self.executor_pool = None
        self.samples = []

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, f"iter_{trainer.global_step}")
        self.save_start(target_dir)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_end()

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, self.predict_tag)
        self.save_start(target_dir)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        samples: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, samples, batch)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_end()

    def state_dict(self) -> Dict[str, Any]:
        return dict()


@torch.no_grad()
def copy_params(src_model, dst_model):
    for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
        dst_param.data.copy_(src_param.data)


@torch.no_grad()
def no_grad(net):
    assert net is not None, "net is None"
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    return net


@torch.no_grad()
def filter_nograd_tensors(params_list):
    return [param for param in params_list if param.requires_grad]


class TimeProfiler:
    def __init__(
        self,
        enabled: bool = True,
        window_size: int = 100,
        warmup_steps: int = 10,
        print_freq: int = 10,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        use_cuda_events: bool = True,
        detailed: bool = True,
        track_memory: bool = True,
        export_json: bool = False,
        export_path: str = "profiling_results.json",
    ):
        self.enabled = enabled
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.print_freq = print_freq
        self.detailed = detailed
        self.track_memory = track_memory
        self.export_json = export_json
        self.export_path = export_path

        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.world_size = world_size if world_size is not None else (dist.get_world_size() if dist.is_initialized() else 1)

        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()

        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.counts = defaultdict(int)
        self.current_timers = {}
        self.step_count = 0
        self.total_start_time = time.time()
        self.step_start_time = None

        self.memory_stats = defaultdict(lambda: deque(maxlen=window_size))
        self.timer_stack = []

        if self.use_cuda_events:
            self.cuda_events = {}

    def _get_time(self) -> float:
        if self.use_cuda_events:
            torch.cuda.synchronize()
        return time.time()

    @contextmanager
    def profile(self, name: str):
        if not self.enabled:
            yield
            return

        if self.use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            try:
                yield
            finally:
                end_event.record()
                torch.cuda.synchronize()

                if self.step_count >= self.warmup_steps:
                    cuda_time = start_event.elapsed_time(end_event) / 1000.0
                    self.timings[name].append(cuda_time)
                self.counts[name] += 1
        else:
            start_time = self._get_time()
            try:
                yield
            finally:
                end_time = self._get_time()
                if self.step_count >= self.warmup_steps:
                    self.timings[name].append(end_time - start_time)
                self.counts[name] += 1

    def start_timer(self, name: str):
        if not self.enabled:
            return

        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.current_timers[name] = event
        else:
            self.current_timers[name] = self._get_time()

    def end_timer(self, name: str):
        if not self.enabled or name not in self.current_timers:
            return

        if self.use_cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()

            if self.step_count >= self.warmup_steps:
                elapsed = self.current_timers[name].elapsed_time(end_event) / 1000.0
                self.timings[name].append(elapsed)
        else:
            end_time = self._get_time()
            if self.step_count >= self.warmup_steps:
                self.timings[name].append(end_time - self.current_timers[name])

        self.counts[name] += 1
        del self.current_timers[name]

    def start_step(self):
        if not self.enabled:
            return

        self.step_start_time = self._get_time()
        self.start_timer("total_step")

        if self.track_memory:
            self._record_memory_stats("step_start")

    def end_step(self, print_stats: bool = None):
        if not self.enabled:
            return

        self.end_timer("total_step")
        self.step_count += 1

        if self.track_memory:
            self._record_memory_stats("step_end")

        if print_stats is None:
            print_stats = (self.step_count % self.print_freq == 0) and (self.step_count > self.warmup_steps)

        if print_stats and self.rank == 0:
            self.print_step_stats()

        if self.export_json and print_stats:
            self._export_to_json()

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, times in self.timings.items():
            if len(times) > 0:
                times_array = np.array(times)
                stats[name] = {
                    "mean": np.mean(times_array),
                    "std": np.std(times_array),
                    "min": np.min(times_array),
                    "max": np.max(times_array),
                    "median": np.median(times_array),
                    "count": self.counts[name],
                    "total": np.sum(times_array),
                }
        return stats

    def _record_memory_stats(self, tag: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            self.memory_stats[f"cuda_allocated_{tag}"].append(allocated)
            self.memory_stats[f"cuda_reserved_{tag}"].append(reserved)

        cpu_percent = psutil.virtual_memory().percent
        self.memory_stats[f"cpu_percent_{tag}"].append(cpu_percent)

    def _export_to_json(self):
        if self.rank != 0:
            return

        stats = self.get_stats()
        memory = {k: float(np.mean(v)) if len(v) > 0 else 0 for k, v in self.memory_stats.items()}

        export_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "timings": {
                k: {
                    "mean_ms": v["mean"] * 1000,
                    "std_ms": v["std"] * 1000,
                    "count": v["count"],
                }
                for k, v in stats.items()
            },
            "memory": memory,
        }

        with open(self.export_path, "a") as f:
            json.dump(export_data, f)
            f.write("\n")

    def print_step_stats(self):
        if not self.enabled or self.rank != 0:
            return

        stats = self.get_stats()
        if not stats:
            return

        total_elapsed = time.time() - self.total_start_time

        print("\n" + "=" * 80)
        print(f"🚀 Training Step {self.step_count} Performance Report")
        print("=" * 80)

        if "total_step" in stats:
            step_time = stats["total_step"]["mean"]
            throughput = 1.0 / step_time if step_time > 0 else 0
            samples_per_sec = throughput * self.world_size

            print(f"\n📊 Overall Metrics:")
            print(f"  • Step Time: {step_time*1000:.2f}ms (±{stats['total_step']['std']*1000:.2f}ms)")
            print(f"  • Throughput: {throughput:.2f} steps/sec/GPU")
            print(f"  • Global Throughput: {samples_per_sec:.2f} samples/sec")
            print(f"  • Total Runtime: {str(timedelta(seconds=int(total_elapsed)))}")

        if self.detailed and len(stats) > 1:
            print(f"\n⏱️  Detailed Timing Breakdown:")
            print("-" * 60)

            sorted_stats = sorted(
                [(k, v) for k, v in stats.items() if k != "total_step"],
                key=lambda x: x[1]["mean"],
                reverse=True,
            )

            for name, timing in sorted_stats:
                mean_time = timing["mean"] * 1000
                std_time = timing["std"] * 1000
                percentage = (timing["mean"] / stats.get("total_step", {"mean": 1})["mean"]) * 100

                bar_length = int(percentage / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)

                print(f"  {name:20s}: {mean_time:7.2f}ms ±{std_time:6.2f}ms [{bar}] {percentage:5.1f}%")

        if self.track_memory and self.memory_stats:
            print(f"\n💾 Memory Usage:")
            print("-" * 60)

            if "cuda_allocated_step_end" in self.memory_stats:
                cuda_alloc = np.mean(self.memory_stats["cuda_allocated_step_end"])
                cuda_reserved = np.mean(self.memory_stats["cuda_reserved_step_end"])
                print(f"  GPU Memory: {cuda_alloc:.2f}GB allocated / {cuda_reserved:.2f}GB reserved")

            if "cpu_percent_step_end" in self.memory_stats:
                cpu_mem = np.mean(self.memory_stats["cpu_percent_step_end"])
                print(f"  CPU Memory: {cpu_mem:.1f}% used")

        print("=" * 80 + "\n")


_global_profiler = None


def get_profiler() -> TimeProfiler:
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = TimeProfiler()
    return _global_profiler


def set_profiler(profiler: TimeProfiler):
    global _global_profiler
    _global_profiler = profiler


def record_metric(name: str, value: float):
    profiler = get_profiler()
    if profiler.enabled and profiler.step_count >= profiler.warmup_steps:
        profiler.timings[name].append(value)
        profiler.counts[name] += 1

