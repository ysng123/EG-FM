"""Frozen Gemma decoder used to condition PixelDiT-T2I."""

from contextlib import nullcontext

import torch
import torch.nn as nn


class GemmaTextEncoder(nn.Module):
    def __init__(
        self,
        model_path,
        max_length=300,
        dtype=torch.bfloat16,
        y_norm=True,
        y_norm_scale_factor=0.01,
        use_chi_prompt=True,
        local_files_only=True,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "T2I text encoding requires transformers; install t2i/requirements.txt"
            ) from exc
        self.max_length = int(max_length)
        self.y_norm = bool(y_norm)
        self.y_norm_scale_factor = float(y_norm_scale_factor)
        self.use_chi_prompt = bool(use_chi_prompt)
        self.chi_prompt = "\n".join(
            [
                'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation.',
                "If it is simple, add concrete colors, shapes, textures, and spatial relationships; if detailed, refine it lightly.",
                "Return only the enhanced description for this prompt:",
                "User Prompt: ",
            ]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="right", local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        language_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=local_files_only
        )
        self.encoder = self._resolve_decoder(language_model).eval()
        self.encoder.requires_grad_(False)

    @staticmethod
    def _resolve_decoder(model):
        if hasattr(model, "get_decoder"):
            try:
                decoder = model.get_decoder()
                if decoder is not None:
                    return decoder
            except Exception:
                pass
        for name in ("model", "language_model", "transformer", "decoder"):
            decoder = getattr(model, name, None)
            if decoder is not None:
                return decoder
        raise RuntimeError(f"Could not resolve decoder from {model.__class__.__name__}")

    @torch.no_grad()
    def encode(self, prompts, device):
        prompts = list(prompts)
        if self.use_chi_prompt:
            prompts = [self.chi_prompt + prompt for prompt in prompts]
            prefix_length = len(self.tokenizer.encode(self.chi_prompt))
            total_length = prefix_length + self.max_length - 2
        else:
            total_length = self.max_length
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=total_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        self.encoder.to(device)
        autocast = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
        if self.use_chi_prompt:
            selection = [0] + list(range(-self.max_length + 1, 0))
            hidden, attention_mask = hidden[:, selection], attention_mask[:, selection]
        if self.y_norm:
            hidden = hidden * self.y_norm_scale_factor
        return hidden, attention_mask

    @torch.no_grad()
    def encode_null(self, batch_size, device):
        use_chi_prompt = self.use_chi_prompt
        self.use_chi_prompt = False
        try:
            hidden, mask = self.encode([""], device)
        finally:
            self.use_chi_prompt = use_chi_prompt
        return hidden.repeat(batch_size, 1, 1), mask.repeat(batch_size, 1)


__all__ = ["GemmaTextEncoder"]
