# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from https://github.com/NVlabs/Sana

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint


class ModelRegistry:

    def __init__(self):
        self._module_dict = {}

    def register_module(self, name=None):
        def _register(cls):
            key = name or cls.__name__
            if key in self._module_dict:
                raise KeyError(f"Model '{key}' already registered.")
            self._module_dict[key] = cls
            return cls

        return _register

    def build(self, cfg, default_args=None):
        if isinstance(cfg, str):
            cfg = dict(type=cfg)
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise ValueError("Model config must be a string or dict with key 'type'.")

        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        if obj_type not in self._module_dict:
            raise KeyError(f"Model '{obj_type}' is not registered.")

        kwargs = {}
        if default_args:
            kwargs.update(default_args)
        kwargs.update(cfg)
        return self._module_dict[obj_type](**kwargs)


MODELS = ModelRegistry()

transformers_logging.set_verbosity_error()

# Ensure built-in models are registered on import
try:
    from diffusion.model.trainer import PixDiTTrainer  # noqa: F401
except Exception:
    PixDiTTrainer = None


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)

    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(name="gemma-2-2b-it", device="cuda"):
    text_encoder_dict = {
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }
    assert name in list(text_encoder_dict.keys()), f"not support this text encoder: {name}"
    if "gemma" in name or "Qwen" in name:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dict[name])
        tokenizer.padding_side = "right"
        print(f"loading text encoder from {text_encoder_dict[name]}")
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_dict[name], torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
    else:
        print("error load text encoder")
        exit()

    return tokenizer, text_encoder
