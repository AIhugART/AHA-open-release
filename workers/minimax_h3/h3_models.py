"""Build MiniMax H3 from the community ConvRot checkpoints without ComfyUI.

Everything here is written from the checkpoints' own headers, diffusers (Apache-2.0: the H3 transformer, VAEs, scheduler and
pipeline blocks) and comfy-kitchen (Apache-2.0: the ConvRot kernels). No code from ComfyUI (GPL-3.0) was read or copied.

Facts this file relies on, with where each was checked (plan 2026-11-24_minimax_h3_plan.md section 3.2):
  * the "pruned" checkpoints replace the timestep MLP by a table ``adaln_t_table [1025, 8]`` holding
    ``SiLU(time_embedder(sin-cos(t)))`` at ``t = i / 1024`` (matches the full path of the same model to 0.12 %), followed by
    adaLN projections that take those 8 numbers directly (K1);
  * the fused ``qkv_proj`` is stacked ``[q; k; v]`` (not interleaved per head) and ``mlp.fc1`` is ``[gate; value]``, which
    diffusers' SwiGLU reads as ``[value; gate]`` (K5).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch
import torch.nn as nn

from convrot import QUANTIZED_LINEARS, ConvRotLinear, Nvfp4Linear, StreamedConvRotLinear
from lenient_safetensors import SafetensorsFile

STREAM_DTYPE = torch.bfloat16  # dtype of the packed sequence; the checkpoint's own context_embedder is bf16
TIME_TABLE = "adaln_t_table"


class UnsupportedCheckpoint(ValueError):
    """The file is not one of the layouts this loader was written and checked against."""


# ------------------------------------------------------------------------------------------------ the transformer


class BakedTimeEmbedder(nn.Module):
    """Replaces diffusers' ``time_proj`` + ``time_embedder``: linear interpolation in the 1025-row table of SiLU'd time embeddings."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = nn.Parameter(table.float(), requires_grad=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:  # [T] in [0, 1] -> [T, 8]
        rows = self.table.shape[0]
        position = timestep.float().clamp(0.0, 1.0) * (rows - 1)
        low = position.floor().long().clamp(max=rows - 2)
        frac = (position - low.to(position.dtype)).unsqueeze(-1)
        return (1.0 - frac) * self.table[low] + frac * self.table[low + 1]


def _modulation_forward(self, temb: torch.Tensor):
    """diffusers' block modulation without its SiLU (the table already holds the activated vector); float32 math, stream dtype out."""
    out = self.linear(temb.to(self.linear.weight.dtype)).to(STREAM_DTYPE)
    return out.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def _norm_out_forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
    shift, scale = self.linear(temb.to(self.linear.weight.dtype)).to(hidden_states.dtype).chunk(2, dim=-1)
    hidden_states = self.norm(hidden_states)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


# The feed-forward's 28672-wide intermediate for 37k tokens alone is 2.1 GB of bf16. 2048 tokens (not 8192): with the INT8 build's
# 19.5 GiB of weights resident, 8192 ran out of memory on the second step; 2048 passes and measured no slower per step.
FF_CHUNK_TOKENS = 2048


def _chunked_feed_forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """diffusers' SwiGLU feed-forward (``proj -> value * silu(gate) -> out``) over slices of the token axis.

    Why: at 1344x768 the whole-sequence intermediate pushed the GPU past its physical memory, and on Windows the driver then
    spills to system RAM silently: the same block measured 3.95 s instead of 1.19 s. Both weights are decoded once per call.
    """
    proj, out = self.net[0].proj, self.net[2]
    w_in, w_out = proj.dense(hidden_states.dtype), out.dense(hidden_states.dtype)
    result = torch.empty(*hidden_states.shape[:-1], w_out.shape[0], dtype=hidden_states.dtype, device=hidden_states.device)
    for i in range(0, hidden_states.shape[-2], FF_CHUNK_TOKENS):
        value, gate = torch.nn.functional.linear(hidden_states[..., i : i + FF_CHUNK_TOKENS, :], w_in).chunk(2, dim=-1)
        result[..., i : i + FF_CHUNK_TOKENS, :] = torch.nn.functional.linear(value * torch.nn.functional.silu(gate), w_out)
    return result


def _linear_module(file: SafetensorsFile, prefix: str, device: torch.device) -> ConvRotLinear | Nvfp4Linear:
    meta = file.get_json(prefix + ".comfy_quant")
    fmt = meta.get("format")
    if meta == {"format": "nvfp4"}:
        return Nvfp4Linear.from_file(file.get(prefix + ".weight").to(device), file.get(prefix + ".weight_scale").to(device),
                                     file.get(prefix + ".weight_scale_2").to(device))
    if fmt == "int8_tensorwise" and meta.get("convrot") and meta.get("convrot_groupsize") == 256:
        packed4 = False
    elif fmt == "convrot_w4a4" and meta.get("linear_dtype") == "int8" and meta.get("convrot_groupsize") == 256:
        packed4 = True  # INT4 weights, INT8 product
    else:
        raise UnsupportedCheckpoint(f"{prefix}: quantization {meta} is not one this loader handles")
    return ConvRotLinear(
        file.get(prefix + ".weight").to(device), file.get(prefix + ".weight_scale").to(device), packed4
    )


def _rename(key: str) -> str:
    """ComfyUI-layout name -> diffusers name (names only; tensor changes are done by the caller)."""
    if key.startswith("token_refiner.blocks."):
        key = "token_refiner.refiner_blocks." + key[len("token_refiner.blocks.") :]
    elif key.startswith("blocks."):
        key = "transformer_blocks." + key[len("blocks.") :]
    for old, new in (
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
        (".mlp.fc2.", ".ff.net.2."),
    ):
        key = key.replace(old, new)
    return key


def build_transformer(weights: str | Path, config: str | Path, device: str | torch.device = "cuda"):
    """Load a pruned ConvRot FL2VA/Ref2VA transformer onto ``device``. Raises if a single tensor is left over or missing."""
    from diffusers import MiniMaxH3Transformer3DModel

    device = torch.device(device)
    cfg = {k: v for k, v in json.loads(Path(config).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    file = SafetensorsFile(weights)
    if TIME_TABLE not in file:
        raise UnsupportedCheckpoint(f"{Path(weights).name}: no {TIME_TABLE}; only the pruned checkpoints are supported")
    table = file.get(TIME_TABLE)
    cfg["time_embed_dim"] = table.shape[1]  # 8: the adaLN projections of the pruned files take the baked vector directly
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**cfg)

    inner = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    ffn = cfg["ffn_dim"]
    plain: dict[str, torch.Tensor] = {}
    quantized: set[str] = set()

    for key in list(file.keys()):
        if key in ("rope.inv_freq", TIME_TABLE) or key.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        prefix = key[: -len(".weight")] if key.endswith(".weight") else None
        if prefix is not None and prefix + ".comfy_quant" in file:
            base = _rename(prefix + ".")[:-1]  # the rename rules match whole path segments, so they need the trailing dot
            linear = _linear_module(file, prefix, device)
            if prefix.endswith(".attn.qkv_proj"):
                target = base[: -len("qkv_proj")]
                for i, name in enumerate(("to_q", "to_k", "to_v")):
                    model.set_submodule(target + name, linear.rows(i * inner, (i + 1) * inner))
            elif prefix.endswith(".mlp.fc1"):  # [gate; value] -> [value; gate]
                model.set_submodule(base.replace(".mlp.fc1", ".ff.net.0.proj"), linear.swapped(ffn))
            else:
                model.set_submodule(base, linear)
            quantized.add(prefix)
            continue

        tensor = file.get(key)
        name = _rename(key)
        if name.endswith(".attn.qkv_proj.weight"):  # unquantized (the token refiner)
            for i, part in enumerate(("to_q", "to_k", "to_v")):
                plain[name.replace("qkv_proj", part)] = tensor[i * inner : (i + 1) * inner].contiguous().to(device)
        elif name.endswith(".mlp.fc1.weight"):
            plain[name.replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat([tensor[ffn:], tensor[:ffn]]).to(device)
        else:
            if "adaln_proj" in key or key.startswith("final_layer.adaln_proj"):
                tensor = tensor.float()  # tiny, and the modulation is accumulated in float32 then cast to the stream dtype
            plain[name] = tensor.to(device)

    model.time_proj = nn.Identity()
    model.time_embedder = BakedTimeEmbedder(table.to(device))
    for block in model.transformer_blocks:
        block.adaln_proj.forward = types.MethodType(_modulation_forward, block.adaln_proj)
        if isinstance(block.ff.net[0].proj, QUANTIZED_LINEARS) and isinstance(block.ff.net[2], QUANTIZED_LINEARS):
            block.ff.forward = types.MethodType(_chunked_feed_forward, block.ff)
    model.norm_out.forward = types.MethodType(_norm_out_forward, model.norm_out)

    result = model.load_state_dict(plain, strict=False, assign=True)
    leftovers = [k for k in result.missing_keys if not k.startswith("time_embedder.")]
    if result.unexpected_keys or leftovers:
        raise UnsupportedCheckpoint(
            f"{Path(weights).name}: {len(result.unexpected_keys)} tensors the model has no place for "
            f"(first: {result.unexpected_keys[:3]}), {len(leftovers)} model parameters the file did not give "
            f"(first: {leftovers[:3]})"
        )

    # rope.inv_freq is computed, not stored (non-persistent buffer): built on meta, so rebuild it and prove it equals the file's.
    freq = cfg["rope_freq_dim"]
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 2 * freq, 2, dtype=torch.float32) / (2 * freq)))
    if "rope.inv_freq" in file and not torch.equal(inv, file.get("rope.inv_freq")):
        raise UnsupportedCheckpoint("rope.inv_freq in the file differs from the one diffusers computes")
    model.rope.inv_freq = inv.to(device)

    still_meta = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if still_meta:
        raise UnsupportedCheckpoint(f"tensors still on the meta device: {still_meta[:5]}")
    file.close()
    return model.eval().requires_grad_(False)


# ------------------------------------------------------------------------------------------------ the text encoder


def _sdpa_with_repeated_kv(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """transformers' "sdpa" attention without ``enable_gqa``.

    Why: for grouped-query attention (the encoder has 64 query heads and 8 key/value heads) transformers asks SDPA for ``enable_gqa=True``.
    Only the flash and the math kernels accept that flag, and the Windows builds of PyTorch have no flash kernel, so SDPA silently
    computes the full 64 x L x L score matrix in float32: 10.2 GiB at the 6 544 tokens of one reference image (measured, plan section 3.10).
    Repeating the key/value heads first keeps the memory-efficient kernel in play, which never builds that matrix.
    """
    import torch.nn.functional as F
    from transformers.integrations.sdpa_attention import repeat_kv

    groups = getattr(module, "num_key_value_groups", 1)  # the vision tower has none
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    is_causal = bool(query.shape[2] > 1 and attention_mask is None and is_causal)
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal)
    return out.transpose(1, 2).contiguous(), None


def build_text_encoder(weights: str | Path, config_dir: str | Path, device: str | torch.device = "cuda", with_vision: bool = False):
    """Qwen3-VL-32B cut at layer 50, its linears streamed from the file when called.

    The pipeline reads ``text_encoder.model(...).hidden_states[50]`` and refuses a stack of 50 or fewer layers (it cannot tell
    that the last state of a short stack is not normalised), so the config keeps its 64 layers, only the first 50 exist, and the
    final norm is an identity: ``hidden_states[50]`` is then the output of layer 50 before any norm, which is what the file's
    metadata says it holds ("unnormalized_hidden_after_layer_50").
    """
    from transformers import AttentionInterface, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    AttentionInterface.register("sdpa", _sdpa_with_repeated_kv)  # this worker process only runs this encoder (and the diffusers transformer)
    device = torch.device(device)
    cfg = Qwen3VLConfig.from_pretrained(str(config_dir))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # meta tensors carry the dtype the pipeline will read back from text_encoder.dtype
    try:
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(cfg)
    finally:
        torch.set_default_dtype(previous)

    file = SafetensorsFile(weights)
    layers = 1 + max(int(k.split(".")[2]) for k in file.keys() if k.startswith("model.layers."))
    lm = model.model.language_model
    lm.layers = nn.ModuleList(list(lm.layers)[:layers])
    lm.norm = nn.Identity()
    model.lm_head = nn.Identity()

    def target(key: str) -> str:
        if key.startswith("model.layers.") or key.startswith("model.embed_tokens."):
            return "model.language_model." + key[len("model.") :]
        if key.startswith("visual."):
            return "model." + key
        raise UnsupportedCheckpoint(f"unexpected text encoder tensor {key}")

    plain: dict[str, torch.Tensor] = {}
    for key in list(file.keys()):
        if key.endswith((".comfy_quant", ".weight_scale")):
            continue
        if key.startswith("visual.") and not with_vision:
            continue
        prefix = key[: -len(".weight")] if key.endswith(".weight") else None
        if prefix is not None and prefix + ".comfy_quant" in file:
            model.set_submodule(target(prefix), StreamedConvRotLinear(file, prefix, device))
            continue
        plain[target(key)] = file.get(key).to(device)

    result = model.load_state_dict(plain, strict=False, assign=True)
    missing = [k for k in result.missing_keys if not (k.startswith("model.visual.") and not with_vision)]
    if result.unexpected_keys or missing:
        raise UnsupportedCheckpoint(
            f"text encoder: unexpected {result.unexpected_keys[:3]}, missing {missing[:3]} ({len(missing)} in all)"
        )
    lm.rotary_emb = type(lm.rotary_emb)(cfg.text_config, device=device)  # a computed buffer, built on meta above
    if with_vision:  # the vision tower's rotary table is computed too (not in the file), so it is built here as well
        visual = model.model.visual
        visual.rotary_pos_emb = type(visual.rotary_pos_emb)(cfg.vision_config, device=device)
    if not with_vision:
        # Text-only requests never touch the vision tower. Left as meta placeholders it would make the pipeline read the
        # encoder's device as "meta" (it takes the device of the first parameter), so it is replaced by an empty module.
        model.model.visual = nn.Identity()
    still_meta = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if still_meta:
        raise UnsupportedCheckpoint(f"text encoder: tensors still on meta: {still_meta[:5]}")
    model._h3_file = file  # keeps the memory map alive for the streamed layers
    return model.eval().requires_grad_(False)


# ------------------------------------------------------------------------------------------------ the two VAEs


def _vae_config(path: str | Path) -> dict:
    return {k: v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items() if not k.startswith("_")}


def _rename_video_vae(key: str) -> str:
    if key.startswith("encoder.down."):
        level, rest = key[len("encoder.down.") :].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        key = f"encoder.down_blocks.{level}.{rest}"
    return (
        key.replace("decoder.x_embedder.", "decoder.proj_in.")
        .replace(".attn.to_out.", ".attn.to_out.0.")
        .replace(".ff.w1.", ".ff.net.0.proj.")
        .replace(".ff.w2.", ".ff.net.2.")
    )


def _deinterleave_qkv(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """[h0: q k v, h1: q k v, ...] rows -> [q_all; k_all; v_all]."""
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0)


def build_video_vae(
    weights: str | Path, config: str | Path, device: str | torch.device = "cuda", qkv_interleaved: bool = True,
    dtype: torch.dtype = torch.float16,
):
    """The video VAE from the original-layout file. ``qkv_interleaved`` says how the decoder's fused qkv rows are stacked."""
    from diffusers import AutoencoderKLMiniMaxH3

    device = torch.device(device)
    cfg = _vae_config(config)
    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3(**cfg)
    heads, head_dim = cfg["decoder_num_attention_heads"], cfg["decoder_attention_head_dim"]
    inner = heads * head_dim
    file = SafetensorsFile(weights)
    state: dict[str, torch.Tensor] = {}
    for key in file.keys():
        if key in ("decoder.mask_token", "latents_mean", "latents_std"):
            continue  # mask_token is a training-time buffer; the statistics are in the config
        tensor = file.get(key)
        if ".attn.to_qkv." in key:
            if qkv_interleaved:
                tensor = _deinterleave_qkv(tensor, heads, head_dim)
            prefix, suffix = key.split(".attn.to_qkv.")
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                state[f"{prefix}.attn.{name}.{suffix}"] = tensor[i * inner : (i + 1) * inner].contiguous().to(device, dtype)
            continue
        if ".ff.w1." in key:  # [gate; up] -> [up; gate]
            gate, up = tensor.chunk(2, dim=0)
            tensor = torch.cat([up, gate], dim=0)
        state[_rename_video_vae(key)] = tensor.to(device, dtype)
    result = vae.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys or result.missing_keys:
        raise UnsupportedCheckpoint(
            f"video VAE: unexpected {result.unexpected_keys[:3]}, missing {result.missing_keys[:3]} "
            f"({len(result.unexpected_keys)} / {len(result.missing_keys)})"
        )
    # the decoder's rotary frequencies are computed in __init__ (non-persistent buffer), so they were built on meta: rebuild them
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed

    with torch.device(device):
        vae.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"]), theta=cfg["decoder_rope_theta"]
        )
    still_meta = [n for n, t in [*vae.named_parameters(), *vae.named_buffers()] if t.is_meta]
    if still_meta:
        raise UnsupportedCheckpoint(f"video VAE: tensors still on meta: {still_meta[:5]}")
    file.close()
    return vae.eval().requires_grad_(False)


def build_audio_vae(weights: str | Path, config: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
    """The audio VAE. The file stores plain conv weights; diffusers' class keeps them as weight_norm (g, v) pairs, so g = norm of w, v = w."""
    from diffusers import AutoencoderKLMiniMaxH3Audio

    device = torch.device(device)
    cfg = _vae_config(config)
    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3Audio(**cfg)
    file = SafetensorsFile(weights)
    state: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    for name, param in vae.state_dict().items():
        if name.endswith((".weight_g", ".weight_v")):
            source = name.rsplit(".", 1)[0] + ".weight"
            if source not in file:
                raise UnsupportedCheckpoint(f"audio VAE: the file has no {source}")
            w = file.get(source).float()
            if name.endswith(".weight_v"):
                state[name] = w.to(device, dtype)
            else:  # weight_g keeps size 1 on every dimension except the one weight_norm was applied over
                dims = [d for d, n in enumerate(param.shape) if n == 1]
                state[name] = torch.linalg.vector_norm(w, dim=dims, keepdim=True).to(device, dtype)
            used.add(source)
        elif name in file:
            state[name] = file.get(name).to(device, dtype)
            used.add(name)
    unused = [k for k in file.keys() if k not in used and k not in ("latents_mean", "latents_std")]
    result = vae.load_state_dict(state, strict=False, assign=True)
    if unused or result.unexpected_keys or result.missing_keys:
        raise UnsupportedCheckpoint(
            f"audio VAE: file tensors with no place {unused[:3]} ({len(unused)}), unexpected {result.unexpected_keys[:3]}, "
            f"missing {result.missing_keys[:3]} ({len(result.missing_keys)})"
        )
    still_meta = [n for n, t in [*vae.named_parameters(), *vae.named_buffers()] if t.is_meta]
    if still_meta:
        raise UnsupportedCheckpoint(f"audio VAE: tensors still on meta: {still_meta[:5]}")
    file.close()
    return vae.eval().requires_grad_(False)
