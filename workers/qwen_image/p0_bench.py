"""P0 measurement for Qwen-Image-2.1. Measurement tool, not product code.

One process = one (variant, resolution) cell, so peak numbers are not polluted by an earlier cell.
Prints ONE json line prefixed with `P0_RESULT ` on success. Prompts are never logged.

  python p0_bench.py --variant int8 --size 1024

(The BF16 cells of docs/05_models.md section 3 were measured with an earlier version of this tool; BF16 is no longer offered.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

REF = Path(__file__).resolve().parents[2] / "reference" / "Qwen-Image-2.1"
PATHS = {"int8": REF / "Image21-INT8"}
PROMPT = 'A neon sign reading "CREATE WITH LIGHT", rainy night'  # same prompt as the upstream INT8 README


class RamPeak:
    """Sample this process's RSS: Windows exposes no cheap ru_maxrss, so poll."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def gib(n: int) -> float:
    return round(n / 2**30, 2)


def load(variant: str):
    path = PATHS[variant]
    sys.path.insert(0, str(path))
    from scripts.runtime import load_int8_pipeline  # community loader, measurement only

    return load_int8_pipeline(str(path), local_files_only=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=PATHS, required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=2, help="run 1 = cold (includes warm-up), run 2+ = warm")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "p0_out")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    assert torch.cuda.is_available(), "no CUDA"

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": gib(torch.cuda.get_device_properties(0).total_memory),
        "torch": torch.__version__,
        "variant": a.variant,
        "size": a.size,
        "steps": a.steps,
        "offload": "model_cpu_offload",
    }
    torch.cuda.reset_peak_memory_stats()
    with RamPeak() as ram:
        t0 = time.perf_counter()
        pipe = load(a.variant)
        result["load_s"] = round(time.perf_counter() - t0, 1)
        result["vram_peak_alloc_after_load_gib"] = gib(torch.cuda.max_memory_allocated())
        runs = []
        for i in range(a.runs):
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            kwargs = {
                "prompt": PROMPT, "width": a.size, "height": a.size, "num_inference_steps": a.steps,
                "generator": torch.Generator("cpu").manual_seed(a.seed),
                "true_cfg_scale": 1.0, "use_kv_cache": True,  # upstream INT8 recipe
            }
            img = pipe(**kwargs).images[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            f = a.out / f"{a.variant}_{a.size}_run{i + 1}.png"
            img.save(f)
            runs.append({
                "run": i + 1, "seconds": round(dt, 1),
                "vram_peak_alloc_gib": gib(torch.cuda.max_memory_allocated()),
                "vram_peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                "size_px": list(img.size), "mode": img.mode,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
            })
        result["runs"] = runs
    result["ram_peak_rss_gib"] = gib(ram.peak)
    print("P0_RESULT " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
