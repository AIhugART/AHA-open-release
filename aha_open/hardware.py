"""What this machine can run, and which model variant to suggest (plan: variant choice, 2026-11-23).

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
# Keep the BF16 pipeline entirely on the GPU only when the weights (~33 GiB) plus activations fit. Must equal
# BF16_RESIDENT_GIB in workers/qwen_image/worker.py (tests/test_install.py checks that).
BF16_RESIDENT_GIB = 40


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
    if gpu_gib >= BF16_RESIDENT_GIB:
        return "bf16", f"your GPU has {gpu_gib:.0f} GiB: the full model fits on it (best quality, no CPU offload, about 33 GB of disk)"
    if gpu_gib >= INT8_MIN_VRAM_GIB:
        ram = f" and {ram_gib:.0f} GiB of RAM" if ram_gib else ""
        return "int8", (
            f"your GPU has {gpu_gib:.0f} GiB{ram}: int8 (about 18 GB of disk, ~9.5 GiB of GPU memory at 1024x1024) is the "
            "comfortable choice; bf16 (about 33 GB) would have to swap parts of the model in and out of the GPU"
        )
    return None, f"your GPU has {gpu_gib:.0f} GiB; Qwen-Image-2.1 needs at least {INT8_MIN_VRAM_GIB:.0f} GiB even in int8"
