"""Run the diffusers MiniMax H3 pipeline stage by stage so that only one big component sits on the GPU at a time.

The encoder (27 GB on disk), the transformer (15-20 GB) and the video VAE (5 GB) cannot share a 24 GiB card, so the blocks of the
``t2va`` workflow are run in three groups (``text_encoder``, ``denoise.*``, ``decode.*``) and each group's component is built
before it and dropped after it. Cancellation is checked at every transformer call (one per sampling step), which is the
finest grain the sampling loop offers; a cancelled request raises ``Cancelled`` and never produces a result.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import h3_models


class Cancelled(Exception):
    """The caller asked for the job to stop; no result exists."""


@dataclass
class Weights:
    transformer: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path
    config_dir: Path  # the official repository's small files: configs, tokenizer, processor, schedulers
    transformer_config: str = "transformer/config.json"  # the Ref2VA checkpoint is described by transformer_ref/config.json
    accepts_references: bool = False  # True for the Ref2VA checkpoint: its jobs carry ordered image/video/audio references
    memory_limit_gib: float | None = None  # the operator's limit on GPU memory (None = whatever is free)


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


GPU_MARGIN_GIB = 0.5


def cap_gpu_memory(device: str = "cuda", limit_gib: float | None = None) -> float:
    """Never let PyTorch take more than the GPU memory that is physically free, so going over is an out-of-memory error.

    Why: on Windows the NVIDIA driver can move GPU allocations to system RAM when the card is full instead of failing. Nothing
    reports it; kernels touching that memory just run several times slower (one transformer block: 3.95 s instead of 1.19 s at
    1344x768, measured). A clear error the caller can act on is better than a 40-minute clip that should take 20.
    ``limit_gib`` is the operator's ``[gpu] memory_limit_gib``: the job behaves as on a card of that size. Returns the cap in GiB.
    """
    dev = torch.device(device)
    if dev.index is None:  # set_per_process_memory_fraction needs an explicit index
        dev = torch.device(dev.type, torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(dev)
    usable = torch.cuda.memory_reserved(dev) + free
    if limit_gib:
        usable = min(usable, limit_gib * 2**30)
    allowed = usable - GPU_MARGIN_GIB * 2**30
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, allowed / total)), dev)
    return allowed / 2**30


def _make_state(pipe, **kwargs):
    from diffusers.modular_pipelines import PipelineState

    state = PipelineState()
    passed = dict(kwargs)
    for param in pipe._blocks.inputs:
        if param.name in passed:
            state.set(param.name, passed.pop(param.name), param.kwargs_type)
        elif param.name is not None and param.name not in state.values:
            state.set(param.name, param.default, param.kwargs_type)
    if passed:
        raise TypeError(f"inputs the t2va workflow does not take: {sorted(passed)}")
    return state


def _run(pipe, state, group: str):
    for name, block in pipe._blocks.sub_blocks.items():
        if name == group or name.startswith(group + "."):
            _, state = block(pipe, state)
    return state


def generate(
    weights: Weights,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    seed: int,
    device: str = "cuda",
    cancelled: Callable[[], bool] = lambda: False,
    on_stage: Callable[[str], None] = lambda s: None,
    on_step: Callable[[int, int], None] = lambda i, n: None,
    log: Callable[[str], None] = print,
    references: list | None = None,
) -> dict:
    """Text (and, for Ref2VA, ordered image/video/audio references) -> video + audio. Returns ``{"frames": uint8 [T,H,W,3], "audio": float32 [2,N], "sampling_rate": int, "timings": {...}}``."""
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    cfg = Path(weights.config_dir)
    timings: dict[str, float] = {}
    log(f"GPU memory cap {cap_gpu_memory(device, weights.memory_limit_gib):.1f} GiB")
    torch.cuda.reset_peak_memory_stats()  # so the peak reported at the end is this job's, whichever stage it was in
    ref2va = bool(references)
    pipe = MiniMaxH3Blocks().get_workflow("ref2va" if ref2va else "t2va").init_pipeline()
    pipe.update_components(
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(cfg / "FL2VA" / "tokenizer")),
        processor=Qwen3VLProcessor.from_pretrained(str(cfg / "FL2VA" / "processor")),
        scheduler=MiniMaxH3Scheduler(shift=12.0),  # video; audio uses 3.0 (scheduler/ and audio_scheduler/ of the official repository)
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    state = _make_state(
        pipe,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
        **({"references": references} if ref2va else {}),
    )

    def check() -> None:
        if cancelled():
            raise Cancelled()

    # 1. text -> embeddings (the encoder streams its layers from the file; nothing else is on the GPU)
    on_stage("encoding the prompt")
    check()
    t = time.monotonic()
    if ref2va:
        state = _run(pipe, state, "before_encode")  # plans the canvas and normalises the references (CPU)
    # The reference workflow sends images and videos through the encoder's vision tower, so it is built only for such requests.
    pipe.update_components(text_encoder=h3_models.build_text_encoder(
        weights.text_encoder, cfg / "FL2VA" / "text_encoder", device, with_vision=ref2va))
    state = _run(pipe, state, "text_encoder")
    pipe.update_components(text_encoder=None)
    _free()
    if ref2va:
        # References become latents through the two VAEs' encoders; they leave the GPU again before the transformer arrives.
        on_stage("encoding the references")
        check()
        pipe.update_components(
            vae=h3_models.build_video_vae(weights.video_vae, cfg / "vae" / "config.json", device),
            audio_vae=h3_models.build_audio_vae(weights.audio_vae, cfg / "audio_vae" / "config.json", device),
        )
        state = _run(pipe, state, "vae_encoder")
        pipe.update_components(vae=None, audio_vae=None)
        _free()
    timings["encode_s"] = round(time.monotonic() - t, 1)
    log(f"encoded in {timings['encode_s']} s")

    # 2. denoising (the transformer is on the GPU only for this group; every forward is a cancellation point and a progress tick)
    on_stage("loading the video model")
    check()
    t = time.monotonic()
    transformer = h3_models.build_transformer(weights.transformer, cfg / weights.transformer_config, device)
    timings["load_transformer_s"] = round(time.monotonic() - t, 1)
    calls = {"n": 0}

    # Hooks, not a wrapper around forward(): the pipeline reads the *signature* of transformer.forward to decide which layout
    # fields (token_tags, position_ids, ...) to pass, and a (*args, **kwargs) wrapper hides it.
    def before_step(module, args, kwargs):
        check()

    def after_step(module, args, kwargs, output):
        calls["n"] += 1
        # The scheduler's grid is linspace(1, 0, steps): its last point is the end, not a step, so `steps` requested means
        # `len(timesteps)` forward passes (19 for 20). The total shown must be the number that will really run.
        on_step(calls["n"], len(pipe.scheduler.timesteps))

    hooks = [
        transformer.register_forward_pre_hook(before_step, with_kwargs=True),
        transformer.register_forward_hook(after_step, with_kwargs=True),
    ]
    slot = "transformer_ref" if ref2va else "transformer"  # the reference workflow reads its own checkpoint partition
    pipe.update_components(**{slot: transformer})
    on_stage("generating")
    t = time.monotonic()
    state = _run(pipe, state, "denoise")
    timings["denoise_s"] = round(time.monotonic() - t, 1)
    log(f"denoised in {timings['denoise_s']} s ({calls['n']} transformer calls)")
    pipe.update_components(**{slot: None})
    for hook in hooks:
        hook.remove()
    del transformer, hooks
    _free()

    # 3. decoding
    on_stage("decoding")
    check()
    t = time.monotonic()
    pipe.update_components(
        vae=h3_models.build_video_vae(weights.video_vae, cfg / "vae" / "config.json", device),
        audio_vae=h3_models.build_audio_vae(weights.audio_vae, cfg / "audio_vae" / "config.json", device),
    )
    state = _run(pipe, state, "decode")
    timings["decode_s"] = round(time.monotonic() - t, 1)
    timings["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)  # allocated by PyTorch, the number docs/product/05_models.md quotes
    check()

    videos = state.get("videos")
    audio = state.get("audio")
    rate = state.get("sampling_rate")
    frames = np.asarray(videos[0] if isinstance(videos, (list, tuple)) else videos)
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if frames.ndim == 5:  # [B, T, H, W, C]
        frames = frames[0]
    return {
        "frames": frames,
        "audio": audio.float().cpu().numpy()[0] if audio.ndim == 3 else audio.float().cpu().numpy(),
        "sampling_rate": int(rate),
        "timings": timings,
    }
