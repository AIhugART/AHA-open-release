"""What this machine can run, and which model variant to use (Qwen-Image-2.1 is offered as INT8 only).

Every threshold below is taken from a measurement recorded in docs/05_models.md section 3 (RTX 3090 Ti, 24 GiB); a
number that has not been measured is not used to refuse anything. Nothing here imports torch.
"""

from __future__ import annotations

import ctypes
import shutil
import sys
from pathlib import Path

# Qwen-Image-2.1 on the 24 GiB reference GPU, 1024x1024, 40 steps (docs/05_models.md section 3).
INT8_MIN_VRAM_GIB = 10.0  # measured peak 9.47 GiB allocated / 9.53 GiB reserved


def detect_ram_gib() -> float | None:
    try:
        if sys.platform == "win32":

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total_phys / 2**30
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError, AttributeError):
        pass
    return None


def free_disk_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 2**30


def recommend_variant(gpu_mib: int | None, ram_gib: float | None) -> tuple[str | None, str]:
    """(variant, one sentence for the operator). None when the machine cannot run the model at all."""
    if not gpu_mib:
        return None, "no NVIDIA GPU was found, so Qwen-Image-2.1 cannot run on this machine"
    gpu_gib = gpu_mib / 1024
    if gpu_gib >= INT8_MIN_VRAM_GIB:
        ram = f" and {ram_gib:.0f} GiB of RAM" if ram_gib else ""
        return "int8", f"your GPU has {gpu_gib:.0f} GiB{ram}: int8 needs about 18 GB of disk and ~9.5 GiB of GPU memory at 1024x1024"
    return None, f"your GPU has {gpu_gib:.0f} GiB; Qwen-Image-2.1 needs at least {INT8_MIN_VRAM_GIB:.0f} GiB"


TIGHT_HEADROOM_GIB = 1.5  # less spare GPU memory than this over the measured peak: it runs, but a bigger size or a LoRA may not
QWEN_PREFERENCE = ("int8-convrot", "mixed-p010", "mixed-p015", "int8")  # best measured quality first; the first one that fits is suggested


def _fit(peak: float | None, gpu_mib: int | None) -> str:
    if peak is None:
        return "unknown"
    if not gpu_mib:
        return "too_small"
    spare = gpu_mib / 1024 - peak
    return "too_small" if spare < 0 else ("tight" if spare < TIGHT_HEADROOM_GIB else "fits")


def fit_for(info: dict, gpu_mib: int | None) -> str:
    """``fits`` | ``tight`` | ``too_small`` | ``unknown``: can this variant run here at all, from measured peaks only.

    The 1024x1024 figure for an image model; for a model whose sizes are named otherwise (video), its best size among the measured ones.
    A number that was not measured never refuses anything (docs/05_models.md section 3): that is ``unknown``.
    """
    vram = info.get("vram_gib") or {}
    if "1k" in vram:
        return _fit(vram["1k"], gpu_mib)
    by_tier = fit_by_tier(info, gpu_mib)
    judged = [by_tier[t] for t, v in vram.items() if v is not None]
    return next((f for f in ("fits", "tight", "too_small") if f in judged), "unknown")


def fit_by_tier(info: dict, gpu_mib: int | None) -> dict[str, str]:
    """The same judgement for every size tier the variant lists, so a client can hide the sizes this machine cannot make.

    ``too_small_up_to_gib`` = {tier: G}: that size was run on a G GiB card and ran out of memory (a measured peak alone does not say
    what the card must hold besides it); a card of that size or smaller is ``too_small``, a larger one is judged by the peak, or is
    ``unknown`` without one.
    """
    vram = info.get("vram_gib") or {}
    failed = info.get("too_small_up_to_gib") or {}
    if "1k" in vram:  # an image model's peaks are taken with offload on: they are not a minimum, so no size is judged
        return {}
    return {tier: "too_small" if tier in failed and (gpu_mib or 0) <= failed[tier] * 1024 else _fit(peak, gpu_mib)
            for tier, peak in vram.items()}


def gpu_budget_mib(cfg, gpu: dict | None) -> int | None:
    """GPU memory the server may use: the card, or the operator's ``[gpu] memory_limit_gib`` when that is smaller."""
    if not gpu:
        return None
    limit = cfg.gpu_memory_limit_gib
    return min(gpu["memory_total_mib"], int(limit * 1024)) if limit else gpu["memory_total_mib"]


def _recommend_qwen(gpu_mib: int | None, ram_gib: float | None) -> tuple[str | None, str]:
    # catalog imports this module, so its table is read when called, not at import
    from .catalog import QWEN_VARIANTS

    if not gpu_mib:
        return None, "no NVIDIA GPU was found, so Qwen-Image-2.1 cannot run on this machine"
    gpu_gib = gpu_mib / 1024
    for name in QWEN_PREFERENCE:
        info = QWEN_VARIANTS[name]
        if fit_for(info, gpu_mib) in ("fits", "tight"):
            ram = f" and {ram_gib:.0f} GiB of RAM" if ram_gib else ""
            return name, (f"your GPU has {gpu_gib:.0f} GiB{ram}: {info['label']} needs about {info['disk_gib']} GB of disk and "
                          f"~{info['vram_gib']['1k']:.1f} GiB of GPU memory at 1024x1024")
    least = min(i["vram_gib"]["1k"] for i in QWEN_VARIANTS.values() if i["vram_gib"]["1k"] is not None)
    return None, f"your GPU has {gpu_gib:.0f} GiB; Qwen-Image-2.1 needs at least {least:.0f} GiB"


H3_PREFERENCE = ("int8", "mixed", "nvfp4")  # the publisher's own quality order (Abiray README), not judged here


def _recommend_h3(model_id: str, gpu_mib: int | None) -> tuple[str | None, str]:
    from .catalog import CATALOG  # catalog imports this module, so its table is read when called

    if not gpu_mib:
        return None, "no NVIDIA GPU was found, so MiniMax H3 cannot run on this machine"
    spec, gpu_gib = CATALOG[model_id], gpu_mib / 1024
    # a build without a measured minimum is never suggested: a suggestion is a promise that it runs
    runnable = [n for n in H3_PREFERENCE if n in spec.variants and n in spec.min_vram_mib and gpu_mib >= spec.min_vram_mib[n]]
    if not runnable:
        least = min(spec.min_vram_mib.values()) / 1024
        return None, f"your GPU has {gpu_gib:.0f} GiB; MiniMax H3 needs at least {least:.0f} GiB"
    name = next((n for n in runnable if fit_by_tier(spec.variants[n], gpu_mib).get("native") in ("fits", "tight")), runnable[0])
    info = spec.variants[name]
    return name, f"your GPU has {gpu_gib:.0f} GiB: the {info['label']} build needs about {info['disk_gib']} GB of disk"


def recommend_variant_for(model_id: str, gpu_mib: int | None, ram_gib: float | None) -> tuple[str | None, str]:
    """``recommend_variant`` for any installable model. Qwen picks among its four variants by measured fit (D8)."""
    if model_id == "qwen-image-2.1":
        return _recommend_qwen(gpu_mib, ram_gib)
    if model_id in ("minimax-h3-fl2va", "minimax-h3-ref2va"):
        return _recommend_h3(model_id, gpu_mib)
    return None, f"no variant rule for {model_id}"
