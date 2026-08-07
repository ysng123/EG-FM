import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pyrallis
import torch
from termcolor import colored
from torchvision.utils import save_image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")  # ignore warning

from diffusion import DPMS
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
)
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import PixDiTConfig, model_init_config
from diffusion.utils.logger import get_root_logger


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_dict_chunks(data, bs):
    keys = []
    for k in data:
        keys.append(k)
        if len(keys) == bs:
            yield keys
            keys = []
    if keys:
        yield keys


def create_tar(data_path):
    tar_path = f"{data_path}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(data_path, arcname=os.path.basename(data_path))
    print(f"Created tar file: {tar_path}")
    return tar_path


def delete_directory(exp_name):
    if os.path.exists(exp_name):
        subprocess.run(["rm", "-r", exp_name], check=True)
        print(f"Deleted directory: {exp_name}")


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale):
    if isinstance(items, dict):
        get_chunks = get_dict_chunks
    else:
        from diffusion.data.datasets.utils import get_chunks

    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    for chunk in tqdm(list(get_chunks(items, bs)), desc=tqdm_desc, unit="batch", position=args.gpu_id, leave=True):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        # data prepare
        prompts, hw, ar = (
            [],
            torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1),
            torch.tensor([[1.0]], device=device).repeat(bs, 1),
        )
        if bs == 1:
            prompt = data_dict[chunk[0]]["prompt"] if dict_prompt else chunk[0]
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
            # Override with CLI-provided custom size if specified
            if args.custom_height is not None and args.custom_width is not None:
                hw = torch.tensor(
                    [[float(args.custom_height), float(args.custom_width)]], dtype=torch.float, device=device
                )
                ar = (hw[:, 0] / hw[:, 1]).unsqueeze(1)
            else:
                # Prefer custom_hw from prompt if provided; otherwise default bucket
                hw = custom_hw
                ar = (hw[:, 0] / hw[:, 1]).unsqueeze(1)
            latent_size_h, latent_size_w = (int(hw[0, 0]), int(hw[0, 1])) if bs == 1 else (latent_size, latent_size)
            prompts.append(prompt_clean.strip())
        else:
            for data in chunk:
                prompt = data_dict[data]["prompt"] if dict_prompt else data
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        # check exists
        save_file_name = f"{chunk[0]}.jpg" if dict_prompt else f"{prompts[0][:100]}.jpg"
        save_path = os.path.join(save_root, save_file_name)
        if os.path.exists(save_path):
            # make sure the noise is totally same
            if bs == 1:
                torch.randn(bs, 3, latent_size_h, latent_size_w, device=device, generator=generator)
            else:
                torch.randn(bs, 3, latent_size, latent_size, device=device, generator=generator)
            continue

        # prepare text feature
        if not config.text_encoder.chi_prompt:
            max_length_all = config.text_encoder.model_max_length
            prompts_all = prompts
        else:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            prompts_all = [chi_prompt + prompt for prompt in prompts]
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = (
                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
            )  # magic number 2: [bos], [_]

        caption_token = tokenizer(
            prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
        caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
            :, :, select_index
        ]
        emb_masks = caption_token.attention_mask[:, select_index]
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

        # start sampling
        with torch.no_grad():
            n = len(prompts)
            z = torch.randn(
                n,
                3,
                latent_size_h,
                latent_size_w,
                device=device,
                generator=generator,
            )
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

            if args.sampling_algo != "flow_dpm-solver":
                raise ValueError(f"{args.sampling_algo} is not supported; use flow_dpm-solver.")

            # Use tensor-returning wrapper to avoid dict outputs in sampler
            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=caption_embs,
                uncondition=null_y,
                guidance_type=guidance_type,
                cfg_scale=cfg_scale,
                model_type="flow",
                model_kwargs=model_kwargs,
                schedule="FLOW",
                interval_guidance=args.interval_guidance,
            )
            samples = dpm_solver.sample(
                z,
                steps=sample_steps,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=flow_shift,
            )

        torch.cuda.empty_cache()

        os.umask(0o000)
        for i, sample in enumerate(samples):
            save_file_name = f"{chunk[i]}.jpg" if dict_prompt else f"{prompts[i][:100]}.jpg"
            save_path = os.path.join(save_root, save_file_name)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class PixelDiTInference(PixDiTConfig):
    config: Optional[str] = "configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
    model_path: Optional[str] = None
    work_dir: Optional[str] = None
    version: str = "sigma"
    txt_file: str = "asset/samples/samples_mini.txt"
    json_file: Optional[str] = None
    sample_nums: int = 100_000
    bs: int = 1
    cfg_scale: float = 3.5
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    custom_height: Optional[int] = None
    custom_width: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    if_save_dirname: bool = False
    negative_prompt: str = ""  # optional negative prompt applied at inference


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=PixelDiTInference, config_path=args.config)

    from tools.download import resolve_checkpoint
    args.model_path = resolve_checkpoint(args.model_path or "pixeldit_t2i_v1.pth")

    args.image_size = config.model.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size
    max_sequence_length = config.text_encoder.model_max_length
    flow_shift = config.scheduler.flow_shift
    guidance_type = "classifier-free"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    default_sample_steps = 50
    sample_steps = args.step if args.step != -1 else default_sample_steps

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)

    null_caption_token = tokenizer(
        args.negative_prompt if hasattr(args, "negative_prompt") and len(args.negative_prompt) > 0 else "",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]

    # model setting
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs
    ).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info("Generating sample from ckpt: %s" % args.model_path)
    assert os.path.isfile(args.model_path), f"Could not find checkpoint at {args.model_path}"
    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    if args.model_path.endswith(".bin"):
        logger.info("Loading fsdp bin checkpoint....")
        old_state_dict = state_dict
        state_dict = dict()
        state_dict["state_dict"] = old_state_dict

    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")
    if args.sampling_algo != "flow_dpm-solver":
        raise ValueError("Only flow_dpm-solver sampling is supported.")

    if args.work_dir is None:
        # Robustly compute a parent directory for saving images based on model_path
        mp = args.model_path
        mp_parent = os.path.dirname(mp)
        mp_grandparent = os.path.dirname(mp_parent) if mp_parent else ""
        if mp_grandparent and mp_grandparent != mp_parent:
            work_dir = mp_grandparent
        elif mp_parent:
            work_dir = mp_parent
        else:
            work_dir = os.getcwd()
    else:
        work_dir = args.work_dir
    config.work_dir = work_dir
    img_save_dir = os.path.join(str(work_dir), "vis")

    logger.info(colored(f"Saving images at {img_save_dir}", "green"))
    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        items = list(data_dict.keys())
    else:
        with open(args.txt_file) as f:
            items = [item.strip() for item in f.readlines()]
    logger.info(f"Eval first {min(args.sample_nums, len(items))}/{len(items)} samples")
    items = items[: max(0, args.sample_nums)]
    items = items[max(0, args.start_index) : min(len(items), args.end_index)]

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}")

    def create_save_root(args, dataset, epoch_name, step_name, sample_steps):
        # Reflect custom size if provided for single-image generation
        if args.bs == 1 and args.custom_height is not None and args.custom_width is not None:
            size_tag = f"_size{args.custom_height}x{args.custom_width}"
        else:
            size_tag = f"_size{args.image_size}"

        save_root = os.path.join(
            img_save_dir,
            f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}{size_tag}_bs{args.bs}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )

        if flow_shift != 1.0:
            save_root += f"_flowshift{flow_shift}"
        if args.interval_guidance[0] != 0 and args.interval_guidance[1] != 1:
            save_root += f"_intervalguidance{args.interval_guidance[0]}{args.interval_guidance[1]}"

        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    dataset = args.dataset
    if args.ablation_selections and args.ablation_key:
        for ablation_factor in args.ablation_selections:
            setattr(args, args.ablation_key, eval(ablation_factor))
            print(f"Setting {args.ablation_key}={eval(ablation_factor)}")
            sample_steps = args.step if args.step != -1 else default_sample_steps

            save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps)
            os.makedirs(save_root, exist_ok=True)
            if args.if_save_dirname and args.gpu_id == 0:
                os.makedirs(f"{work_dir}/metrics", exist_ok=True)
                # save at work_dir/metrics/tmp_xxx.txt for metrics testing
                with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                    print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
                    f.write(os.path.basename(save_root))
            logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

            visualize(
                config=config,
                args=args,
                model=model,
                items=items,
                bs=args.bs,
                sample_steps=sample_steps,
                cfg_scale=args.cfg_scale,
            )
    else:
        logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

        save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps)
        os.makedirs(save_root, exist_ok=True)
        if args.if_save_dirname and args.gpu_id == 0:
            os.makedirs(f"{work_dir}/metrics", exist_ok=True)
            # save at work_dir/metrics/tmp_xxx.txt for metrics testing
            with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
                f.write(os.path.basename(save_root))

        visualize(
            config=config,
            args=args,
            model=model,
            items=items,
            bs=args.bs,
            sample_steps=sample_steps,
            cfg_scale=args.cfg_scale,
        )

        if args.tar_and_del:
            create_tar(save_root)
            delete_directory(save_root)

    print(
        colored("PixelDiT inference has finished. Results stored at ", "green"),
        colored(f"{img_save_dir}", attrs=["bold"]),
        ".",
    )
