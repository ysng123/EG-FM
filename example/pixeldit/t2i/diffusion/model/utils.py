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

import re

import torch
import torch.nn as nn


def set_grad_checkpoint(model, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def set_fp32_attention(model):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.fp32_attention = True

    model.apply(set_attr)


def prepare_prompt_ar(prompt, ratios, device="cpu", show=True):
    # get aspect_ratio or ar
    aspect_ratios = re.findall(r"--aspect_ratio\s+(\d+:\d+)", prompt)
    ars = re.findall(r"--ar\s+(\d+:\d+)", prompt)
    custom_hw = re.findall(r"--hw\s+(\d+:\d+)", prompt)
    if show:
        print("aspect_ratios:", aspect_ratios, "ars:", ars, "hws:", custom_hw)
    prompt_clean = prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0]
    if len(aspect_ratios) + len(ars) + len(custom_hw) == 0 and show:
        print(
            "Wrong prompt format. Set to default ar: 1. change your prompt into format '--ar h:w or --hw h:w' for correct generating"
        )
    if len(aspect_ratios) != 0:
        ar = float(aspect_ratios[0].split(":")[0]) / float(aspect_ratios[0].split(":")[1])
    elif len(ars) != 0:
        ar = float(ars[0].split(":")[0]) / float(ars[0].split(":")[1])
    else:
        ar = 1.0
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    if len(custom_hw) != 0:
        custom_hw = [float(custom_hw[0].split(":")[0]), float(custom_hw[0].split(":")[1])]
    else:
        custom_hw = ratios[closest_ratio]
    default_hw = ratios[closest_ratio]
    prompt_show = f"prompt: {prompt_clean.strip()}\nSize: --ar {closest_ratio}, --bin hw {ratios[closest_ratio]}, --custom hw {custom_hw}"
    return (
        prompt_clean,
        prompt_show,
        torch.tensor(default_hw, device=device)[None],
        torch.tensor([float(closest_ratio)], device=device)[None],
        torch.tensor(custom_hw, device=device)[None],
    )


def get_weight_dtype(mixed_precision):
    if mixed_precision in ["fp16", "float16"]:
        return torch.float16
    elif mixed_precision in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif mixed_precision in ["fp32", "float32"]:
        return torch.float32
    else:
        raise ValueError(f"weigh precision {mixed_precision} is not defined")
