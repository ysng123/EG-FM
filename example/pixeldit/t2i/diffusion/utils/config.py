import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class BaseConfig:
    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def pop(self, attribute_name, default=None):
        if hasattr(self, attribute_name):
            value = getattr(self, attribute_name)
            delattr(self, attribute_name)
            return value
        else:
            return default

    def __str__(self):
        return json.dumps(asdict(self), indent=4)


@dataclass
class DataConfig(BaseConfig):
    data_dir: List[Optional[str]] = field(default_factory=list)
    caption_proportion: Dict[str, int] = field(default_factory=lambda: {"prompt": 1})
    external_caption_suffixes: List[str] = field(default_factory=list)
    external_clipscore_suffixes: List[str] = field(default_factory=list)
    clip_thr_temperature: float = 1.0
    clip_thr: float = 0.0
    del_img_clip_thr: float = 0.0
    sort_dataset: bool = False
    load_text_feat: bool = False
    transform: str = "default_train"
    # default dataset type kept as string for registry lookup
    type: str = "PixDiTWebDatasetMS"
    image_size: int = 512
    hq_only: bool = False
    valid_num: int = 0
    data: Any = None
    extra: Any = None


@dataclass
class ModelConfig(BaseConfig):
    model: str = "PixDiTTrainer"
    teacher: Optional[str] = None
    image_size: int = 512
    mixed_precision: str = "fp16"  # ['fp16', 'fp32', 'bf16']
    fp32_attention: bool = True
    load_from: Optional[str] = None
    discriminator_model: Optional[str] = None
    teacher_model: Optional[str] = None
    teacher_model_weight_dtype: Optional[str] = None
    resume_from: Optional[Union[Dict[str, Any], str]] = field(
        default_factory=lambda: {
            "checkpoint": None,
            "load_ema": False,
            "resume_lr_scheduler": True,
            "resume_optimizer": True,
        }
    )
    aspect_ratio_type: str = "ASPECT_RATIO_1024"
    multi_scale: bool = True
    pe_interpolation: float = 1.0
    micro_condition: bool = False
    attn_type: str = "linear"
    autocast_linear_attn: bool = False
    ffn_type: str = "glumbconv"
    mlp_acts: List[Optional[str]] = field(default_factory=lambda: ["silu", "silu", None])
    mlp_ratio: float = 2.5
    use_pe: bool = False
    pos_embed_type: str = "sincos"
    qk_norm: bool = False
    class_dropout_prob: float = 0.0
    linear_head_dim: int = 32
    cross_norm: bool = False
    cross_attn_type: str = "flash"
    logvar: bool = False
    cfg_scale: int = 4
    cfg_embed: bool = False
    cfg_embed_scale: float = 1.0
    guidance_type: str = "classifier-free"
    # for ladd
    ladd_multi_scale: bool = True
    head_block_ids: Optional[List[int]] = None
    extra: Any = None


@dataclass
class AEConfig(BaseConfig):
    vae_type: str = "none"
    vae_pretrained: str = "none"
    weight_dtype: str = "float32"
    scale_factor: float = 1.0
    vae_latent_dim: int = 3
    vae_downsample_rate: int = 1
    sample_posterior: bool = False
    extra: Any = None


@dataclass
class TextEncoderConfig(BaseConfig):
    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 1.0
    model_max_length: int = 300
    chi_prompt: List[Optional[str]] = field(default_factory=lambda: [])
    extra: Any = None


@dataclass
class SchedulerConfig(BaseConfig):
    train_sampling_steps: int = 1000
    predict_flow_v: bool = True
    noise_schedule: str = "linear_flow"
    pred_sigma: bool = False
    learn_sigma: bool = True
    vis_sampler: str = "flow_dpm-solver"
    flow_shift: float = 1.0
    egfm_sigma0: float = 3.5
    egfm_curve: str = "smootherstep"
    egfm_release_start: float = 0.0
    egfm_release_end: float = 1.0
    # logit-normal timestep
    weighting_scheme: Optional[str] = "logit_normal"
    weighting_scheme_discriminator: Optional[str] = "logit_normal_trigflow"
    add_noise_timesteps: List[float] = field(default_factory=lambda: [1.57080])
    logit_mean: float = 0.0
    logit_std: float = 1.0
    logit_mean_discriminator: float = 0.0
    logit_std_discriminator: float = 1.0
    sigma_data: float = 0.5
    timestep_norm_scale_factor: float = 1.0
    extra: Any = None


@dataclass
class TrainingConfig(BaseConfig):
    num_workers: int = 4
    seed: int = 42
    train_batch_size: int = 32
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    grad_checkpointing: bool = False
    gradient_clip: float = 1.0
    gc_step: int = 1
    optimizer: Dict[str, Any] = field(
        default_factory=lambda: {"eps": 1.0e-10, "lr": 0.0001, "type": "AdamW", "weight_decay": 0.03}
    )
    optimizer_D: Dict[str, Any] = field(
        default_factory=lambda: {"eps": 1.0e-10, "lr": 0.0001, "type": "AdamW", "weight_decay": 0.03}
    )
    load_from_optimizer: bool = False
    load_from_lr_scheduler: bool = False
    resume_lr_scheduler: bool = True
    lr_schedule: str = "constant"
    lr_schedule_args: Dict[str, int] = field(default_factory=lambda: {"num_warmup_steps": 500})
    # When resuming from a checkpoint, if True, force optimizer and scheduler base LRs
    # to follow the YAML value in optimizer["lr"] rather than the checkpoint's LR.
    override_lr_on_resume: bool = False
    auto_lr: Dict[str, str] = field(default_factory=lambda: {"rule": "sqrt"})
    eval_batch_size: int = 16
    use_fsdp: bool = False
    use_flash_attn: bool = False
    eval_sampling_steps: int = 250
    lora_rank: int = 4
    log_interval: int = 50
    mask_type: str = "null"
    mask_loss_coef: float = 0.0
    load_mask_index: bool = False
    snr_loss: bool = False
    real_prompt_ratio: float = 1.0
    early_stop_hours: float = 10000.0
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    save_model_steps: int = 1000000
    visualize: bool = False
    null_embed_root: str = "output/pretrained_models/"
    valid_prompt_embed_root: str = "output/tmp_embed/"
    validation_prompts: List[str] = field(
        default_factory=lambda: [
            "dog",
            "portrait photo of a girl, photograph, highly detailed face, depth of field",
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        ]
    )
    local_save_vis: bool = False
    deterministic_validation: bool = True
    online_metric: bool = False
    eval_metric_step: int = 5000
    online_metric_dir: str = "metric_helper"
    work_dir: str = "./experiments"
    skip_step: int = 0
    loss_type: str = "huber"
    huber_c: float = 0.001
    num_ddim_timesteps: int = 50
    ema_decay: float = 0.95
    ema_update: bool = False
    ema_rate: float = 0.9999
    repa_loss_weight: float = 0.1
    repa_encoder_index: int = 14



@dataclass
class PixDiTConfig(BaseConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vae: AEConfig = field(default_factory=AEConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    work_dir: str = "output/"
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    debug: bool = False
    caching: bool = False
    report_to: str = "wandb"
    tracker_project_name: str = "pixdit-baseline"
    name: str = "baseline"
    loss_report_name: str = "loss"


def model_init_config(config: PixDiTConfig, latent_size: int = 32):

    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    return {
        "input_size": latent_size,
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": config.text_encoder.caption_channels,
        "class_dropout_prob": config.model.class_dropout_prob,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "pos_embed_type": config.model.pos_embed_type,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "cross_norm": config.model.cross_norm,
        "cross_attn_type": config.model.cross_attn_type,
        "timestep_norm_scale_factor": config.scheduler.timestep_norm_scale_factor,
        "extra": getattr(config.model, "extra", None),
    }
