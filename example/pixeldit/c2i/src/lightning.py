# Modified from https://github.com/MCG-NJU/PixNerd

from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback

from .diffusion import BaseSampler, BaseTrainer
from .utils import SimpleEMA, copy_params, filter_nograd_tensors, no_grad, record_metric, TimeProfiler


def fp2uint8(x: torch.Tensor) -> torch.Tensor:
    x = torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
    return x

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

class BaseConditioner(nn.Module):
    def __init__(self):
        super().__init__()

    def _impl_condition(self, y, metadata) -> torch.Tensor:
        raise NotImplementedError()

    def _impl_uncondition(self, y, metadata) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def __call__(self, y, metadata: dict = None):
        if metadata is None:
            metadata = {}
        condition = self._impl_condition(y, metadata)
        uncondition = self._impl_uncondition(y, metadata)
        if condition.dtype in (torch.float64, torch.float32, torch.float16):
            condition = condition.to(torch.bfloat16)
        if uncondition.dtype in (torch.float64, torch.float32, torch.float16):
            uncondition = uncondition.to(torch.bfloat16)
        return condition, uncondition


class LabelConditioner(BaseConditioner):
    def __init__(self, num_classes: int):
        super().__init__()
        self.null_condition = num_classes

    def _impl_condition(self, y, metadata):
        return torch.tensor(y).long().cuda()

    def _impl_uncondition(self, y, metadata):
        return torch.full((len(y),), self.null_condition, dtype=torch.long).cuda()


class LightningModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 denoiser: nn.Module,
                 diffusion_trainer: Optional[BaseTrainer] = None,
                 diffusion_sampler: Optional[BaseSampler] = None,
                 ema_tracker: SimpleEMA=None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 override_lr_on_resume: Optional[float] = None,
                 override_ema_decay_on_resume: Optional[float] = None,
                 eval_original_model: bool = False,
                 enable_profiling: bool = True,
                 profiling_print_freq: int = 10,
                 profiling_warmup_steps: int = 5,
                 conditioner: Optional[nn.Module] = None,
                 ):
        super().__init__()
        self.conditioner = conditioner or LabelConditioner(num_classes)
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.override_lr_on_resume = override_lr_on_resume
        self.override_ema_decay_on_resume = override_ema_decay_on_resume

        self.eval_original_model = eval_original_model

        self._strict_loading = False
        
        self.enable_profiling = enable_profiling
        self.profiling_print_freq = profiling_print_freq
        self.profiling_warmup_steps = profiling_warmup_steps
        self.profiler = None

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        no_grad(self.conditioner)
        no_grad(self.ema_denoiser)

        self.denoiser.compile()
        self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker] if self.ema_tracker is not None else []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        param_groups = [{"params": params_denoiser}]
        for component, extra in [(self.diffusion_trainer, {}), (self.diffusion_sampler, {"lr": 1e-3})]:
            if component is not None:
                params = filter_nograd_tensors(component.parameters())
                if params:
                    param_groups.append({"params": params, **extra})
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)
        
        resumed = (getattr(self, "global_step", 0) or 0) > 0 or getattr(self.trainer, "ckpt_path", None) is not None
        if resumed and self.override_lr_on_resume is not None:
            self._apply_override_lr(float(self.override_lr_on_resume))

        if resumed and self.override_ema_decay_on_resume is not None and self.ema_tracker is not None:
            self._apply_override_ema_decay(float(self.override_ema_decay_on_resume))

        if self.enable_profiling and self.profiler is None:
            export_path = os.path.join(self.trainer.default_root_dir, 'profiling_results.jsonl')
            self.profiler = TimeProfiler(
                enabled=True,
                print_freq=self.profiling_print_freq,
                warmup_steps=self.profiling_warmup_steps,
                window_size=100,
                detailed=True,
                track_memory=True,
                export_json=True,
                export_path=export_path,
                rank=self.global_rank if hasattr(self, 'global_rank') else 0,
                world_size=self.trainer.world_size if hasattr(self.trainer, 'world_size') else 1
            )
    
    def on_train_end(self) -> None:
        if self.profiler:
            self.profiler.summary()


    def training_step(self, batch, batch_idx):
        if self.profiler:
            self.profiler.start_step()
        
        x, y, metadata = batch
        
        with torch.no_grad():
            if self.profiler:
                with self.profiler.profile("data/conditioning"):
                    condition, uncondition = self.conditioner(y, metadata)
            else:
                condition, uncondition = self.conditioner(y, metadata)
        
        if self.profiler:
            with self.profiler.profile("training/diffusion_forward"):
                loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)
        else:
            loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)
        
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        
        if self.profiler:
            for key, value in loss.items():
                if isinstance(value, (float, int, torch.Tensor)):
                    val = value.item() if isinstance(value, torch.Tensor) else value
                    record_metric(f"loss/{key}", val)
            
            self.profiler.end_step()
        
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        if self.eval_original_model:
            samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        else:
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)

        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        if self.diffusion_trainer is not None:
            self.diffusion_trainer.state_dict(
                destination=destination,
                prefix=prefix+"diffusion_trainer.",
                keep_vars=keep_vars)
        return destination

    def _apply_override_lr(self, new_lr: float) -> None:
        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
        try:
            lr_sched_configs = getattr(self.trainer, "lr_scheduler_configs", None)
            if lr_sched_configs is not None:
                for cfg in lr_sched_configs:
                    scheduler = getattr(cfg, "scheduler", None) or cfg
                    base_lrs = getattr(scheduler, "base_lrs", None)
                    if isinstance(base_lrs, list) and len(base_lrs) > 0:
                        scheduler.base_lrs = [new_lr for _ in base_lrs]
        except Exception:
            pass

    def _apply_override_ema_decay(self, new_decay: float) -> None:
        if self.ema_tracker is None:
            return
        self.ema_tracker.decay = new_decay

    def _compute_grad_norm(self) -> float:
        total_norm_sq = None
        for parameter in self.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if total_norm_sq is None:
                total_norm_sq = torch.zeros(1, device=grad.device, dtype=torch.float32)
            total_norm_sq += torch.norm(grad.float(), 2) ** 2
        if total_norm_sq is None:
            return 0.0
        return torch.sqrt(total_norm_sq).item()

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: Optional[int] = None) -> None:
        if optimizer_idx not in (None, 0):
            return

        grad_norm = self._compute_grad_norm()
        self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False, sync_dist=False)

        lrs = [group.get("lr", None) for group in optimizer.param_groups]
        lrs = [lr for lr in lrs if lr is not None]
        if lrs:
            mean_lr = sum(lrs) / len(lrs)
            self.log("train/lr", mean_lr, on_step=True, prog_bar=False, sync_dist=False)

        if self.ema_tracker is not None:
            self.log("train/ema_decay", float(self.ema_tracker.decay), on_step=True, prog_bar=False, sync_dist=False)