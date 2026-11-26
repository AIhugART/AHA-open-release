"""Build Qwen-Image-2.1 from the community ConvRot checkpoints (JoaoZaokk transformers, Comfy-Org text encoder).

Written from the checkpoints' own headers, diffusers (Apache-2.0: the transformer, VAE, scheduler, pipeline) and
comfy-kitchen (Apache-2.0: the ConvRot kernels). No code from ComfyUI (GPL-3.0) was read or copied.

Facts this file relies on, with where each was checked (plan 2026-11-25_qwen_convrot_model_manager_plan.md section 1):
  * the transformer file names every tensor like diffusers' ``QwenImage21Transformer2DModel`` except one: the fused
    ``img_mlp.gate_up`` stands for two layers there, ``img_mlp.gate_layer`` then ``img_mlp.proj`` (order proved against the
    bitsandbytes build, plan D5);
  * the per-layer quantization format sits in the safetensors header, ``__metadata__._quantization_metadata`` (192 layers,
    ``convrot_w4a4`` or ``int8_tensorwise``, group 256); every other tensor is bfloat16 and loaded as is;
  * the text encoder file stores each tensor's format beside it (``<name>.comfy_quant``, JSON as bytes) and also quantizes
    ``embed_tokens`` and ``lm_head`` (INT8 ConvRot, per-row scale).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from convrot import CONVROT_GROUP, ConvRotLinear, StreamedConvRotLinear, dequantize
from lenient_safetensors import SafetensorsFile


class UnsupportedCheckpoint(ValueError):
    """The file is not one of the layouts this loader was written and checked against."""


def _is_packed4(meta: dict, where: str) -> bool:
    """True when the layer holds two INT4 weights per byte (``convrot_w4a4``), False for INT8; anything else is refused."""
    if meta.get("convrot_groupsize") != CONVROT_GROUP:
        raise UnsupportedCheckpoint(f"{where}: quantization {meta} is not a group-{CONVROT_GROUP} ConvRot layer")
    fmt = meta.get("format")
    if fmt == "int8_tensorwise":
        return False
    if fmt == "convrot_w4a4":
        return True
    raise UnsupportedCheckpoint(f"{where}: quantization format {fmt!r} is not one this loader handles")


# ------------------------------------------------------------------------------------------------ the transformer


def transformer_layers(file: SafetensorsFile) -> dict[str, dict]:
    raw = file.metadata.get("_quantization_metadata")
    if not raw:
        raise UnsupportedCheckpoint("the transformer file has no _quantization_metadata in its header")
    return json.loads(raw)["layers"]


def build_transformer(weights: str | Path, config: str | Path, device: str | torch.device = "cuda", gate_first: bool = True):
    """Load a ConvRot Qwen-Image-2.1 transformer onto ``device``. Raises if a single tensor is left over or missing."""
    from diffusers import QwenImage21Transformer2DModel

    device = torch.device(device)
    cfg = {k: v for k, v in json.loads(Path(config).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    file = SafetensorsFile(weights)
    layers = transformer_layers(file)
    with torch.device("meta"):
        model = QwenImage21Transformer2DModel(**cfg)

    built = 0
    for name, meta in layers.items():
        linear = ConvRotLinear(
            file.get(name + ".weight").to(device), file.get(name + ".weight_scale").to(device), _is_packed4(meta, name)
        )
        if name.endswith(".img_mlp.gate_up"):  # one fused layer in the file, two in diffusers
            half = linear.out_features // 2
            prefix = name[: -len("gate_up")]
            first, second = ("gate_layer", "proj") if gate_first else ("proj", "gate_layer")
            model.set_submodule(prefix + first, linear.rows(0, half))
            model.set_submodule(prefix + second, linear.rows(half, 2 * half))
            built += 2
        else:
            model.set_submodule(name, linear)
            built += 1

    plain: dict[str, torch.Tensor] = {}
    for key in file.keys():
        if key.endswith(".weight_scale") or (key.endswith(".weight") and key[: -len(".weight")] in layers):
            continue
        plain[key] = file.get(key).to(device)
    # the quantized layers are modules with buffers now, so the model's own state dict no longer asks for their weights
    result = model.load_state_dict(plain, strict=False, assign=True)
    if result.unexpected_keys or result.missing_keys:
        raise UnsupportedCheckpoint(
            f"{Path(weights).name}: {len(result.unexpected_keys)} tensors the model has no place for "
            f"(first: {result.unexpected_keys[:3]}), {len(result.missing_keys)} model parameters the file did not give "
            f"(first: {result.missing_keys[:3]})"
        )
    quantized = sum(isinstance(m, ConvRotLinear) for m in model.modules())
    if quantized != built:
        raise UnsupportedCheckpoint(f"{quantized} ConvRot layers in the model, {built} built from the file")
    # Two things diffusers computes in __init__ were built on meta above and are not in the file: the timestep frequencies
    # (a non-persistent buffer) and the rotary table (a plain list of tensors, which named_buffers() does not list).
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope, QwenImage21TemporalTimesteps

    with torch.device(device):
        model.time_text_embed.time_proj = QwenImage21TemporalTimesteps(timestep_dim=model.time_text_embed.time_proj.timestep_dim)
        model.pos_embed = QwenImage21Rope(theta=model.pos_embed.theta, axes_dim=model.pos_embed.axes_dim)
    if any(t.is_meta for t in model.pos_embed.freqs):
        raise UnsupportedCheckpoint("the rotary table is still on the meta device")
    still_meta = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if still_meta:
        raise UnsupportedCheckpoint(f"tensors still on the meta device: {still_meta[:5]}")
    file.close()
    return model.eval().requires_grad_(False)


# ------------------------------------------------------------------------------------------------ the text encoder


class ConvRotEmbedding(nn.Module):
    """``embed_tokens`` stored as INT8 ConvRot rows: only the rows a prompt uses are decoded."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.padding_idx = None
        self._dtype = dtype
        self.register_buffer("weight_i8", weight, persistent=False)
        self.register_buffer("scale", scale.reshape(-1).float(), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        rows = dequantize(self.weight_i8[flat], self.scale[flat], self._dtype)
        return rows.reshape(*ids.shape, self.embedding_dim)


def _text_linear(file: SafetensorsFile, prefix: str, device: torch.device, streamed: bool) -> nn.Module:
    meta = file.get_json(prefix + ".comfy_quant")
    if meta.get("format") != "int8_tensorwise" or not meta.get("convrot") or meta.get("convrot_groupsize") != CONVROT_GROUP:
        raise UnsupportedCheckpoint(f"{prefix}: quantization {meta} is not INT8 ConvRot of group {CONVROT_GROUP}")
    if streamed:
        return StreamedConvRotLinear(file, prefix, device)
    return ConvRotLinear(file.get(prefix + ".weight").to(device), file.get(prefix + ".weight_scale").to(device), False)


def build_text_encoder(
    weights: str | Path, config_dir: str | Path, device: str | torch.device = "cuda", with_vision: bool = True, streamed: bool = False,
):
    """Qwen3-VL-8B (36 layers) from the Comfy-Org INT8 ConvRot file.

    ``streamed`` keeps the linears in the file and reads them when called (a few GB less memory, slower); otherwise they sit
    on ``device``. ``with_vision`` keeps the vision tower, which ``image.edit`` needs.
    """
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    device = torch.device(device)
    cfg = Qwen3VLConfig.from_pretrained(str(config_dir))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # meta tensors carry the dtype the pipeline will read back from text_encoder.dtype
    try:
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(cfg)
    finally:
        torch.set_default_dtype(previous)
    model.lm_head = nn.Identity()  # the pipeline reads hidden states, never logits

    file = SafetensorsFile(weights)

    def target(key: str) -> str:
        if key.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
            return "model.language_model." + key[len("model.") :]
        if key.startswith("model.visual."):
            return key
        raise UnsupportedCheckpoint(f"unexpected text encoder tensor {key}")

    plain: dict[str, torch.Tensor] = {}
    for key in list(file.keys()):
        if key.startswith("lm_head.") or key.endswith((".comfy_quant", ".weight_scale")):
            continue
        if key.startswith("model.visual.") and not with_vision:
            continue
        prefix = key[: -len(".weight")] if key.endswith(".weight") else None
        if prefix is not None and prefix + ".comfy_quant" in file:
            if prefix == "model.embed_tokens":
                model.model.language_model.embed_tokens = ConvRotEmbedding(
                    file.get(key).to(device), file.get(prefix + ".weight_scale").to(device), torch.bfloat16
                )
            else:
                model.set_submodule(target(prefix), _text_linear(file, prefix, device, streamed))
            continue
        plain[target(key)] = file.get(key).to(device)

    result = model.load_state_dict(plain, strict=False, assign=True)
    missing = [k for k in result.missing_keys if not (k.startswith("model.visual.") and not with_vision)]
    if result.unexpected_keys or missing:
        raise UnsupportedCheckpoint(f"text encoder: unexpected {result.unexpected_keys[:3]}, missing {missing[:3]} ({len(missing)} in all)")
    lm = model.model.language_model
    lm.rotary_emb = type(lm.rotary_emb)(cfg.text_config, device=device)  # a computed buffer, built on meta above
    if with_vision:
        visual = model.model.visual
        visual.rotary_pos_emb = type(visual.rotary_pos_emb)(cfg.vision_config, device=device)
    else:
        model.model.visual = nn.Identity()
    still_meta = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if still_meta:
        raise UnsupportedCheckpoint(f"text encoder: tensors still on meta: {still_meta[:5]}")
    model._qwen_file = file  # keeps the memory map alive for streamed layers
    return model.eval().requires_grad_(False)
