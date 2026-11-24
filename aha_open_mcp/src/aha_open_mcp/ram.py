"""Free system RAM and the gate before loading a real model.

Why a gate: loading the Qwen INT8 worker held 17.6 GiB of RAM (measured on a 31.7 GiB machine, 2026-11-23). A native crash
with no Python error happened once at load when other programs left only a few GiB free. A tool that an agent can call in a
loop must not walk into that without being told. The threshold is a POLICY (the measurement plus about 15 %), not a measurement."""

from __future__ import annotations

import ctypes
import sys

from . import client
from .targets import Target, TargetError

REAL_MODEL_NEEDS_GIB = 20.0  # measured peak 17.6 GiB RSS + ~15 % headroom (policy)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_ram_gib() -> float | None:
    """Memory available to start a new process, in GiB; None when this platform gives no cheap answer."""
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        return status.ullAvailPhys / 2**30
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20  # kB -> GiB
    except OSError:
        pass
    return None


def model_resident(target: Target) -> bool:
    """A model already on the GPU holds its RAM: free RAM is then low BECAUSE it is loaded, and the next job loads nothing."""
    try:
        health = client.probe(target, "/v1/health").get("health", {})
    except TargetError:
        return False
    return any(g.get("active_model") for g in health.get("gpus", []))


def gate(target: Target, allow_low_ram: bool, free: float | None = None) -> None:
    if not [k for k in target.worker_kinds() if k != "fake"] or allow_low_ram or model_resident(target):
        return
    free = free_ram_gib() if free is None else free
    if free is None:
        raise TargetError("cannot measure free RAM on this platform; loading a real model needs about 17.6 GiB - pass allow_low_ram=true if you accept that")
    if free < REAL_MODEL_NEEDS_GIB:
        raise TargetError(
            f"free RAM is {free:.1f} GiB but loading the real model needs about {REAL_MODEL_NEEDS_GIB:.0f} GiB (the Qwen INT8 worker "
            "measured 17.6 GiB; with only a few GiB free its load once died with a native crash). Close programs and try again, "
            "or pass allow_low_ram=true to accept the risk"
        )
