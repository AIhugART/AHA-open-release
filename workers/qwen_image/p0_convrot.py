"""Q0 measurement for the ConvRot Qwen-Image-2.1 build. Measurement tool, not product code.

One process = one (variant, compute path, placement, size) cell. Prints ONE json line prefixed with `P0_RESULT `.
Prompts are never logged.

  python p0_convrot.py --variant p015 --path dequant --placement gpu --size 1024
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AHA-open" / "models" / "qwen-image-2.1" / "convrot"
FILES = {
    "int8": "qwen_image_2.1_bf16_int8_convrot.safetensors",
    "p015": "qwen_image_2.1_bf16_mixed_p015_int8.safetensors",
    "p010": "qwen_image_2.1_bf16_mixed_p010_int8.safetensors",
}
TEXT_ENCODER = ROOT / "comfyorg" / "text_encoders" / "qwen3vl_8b_int8_convrot.safetensors"
PROMPT = 'A neon sign reading "CREATE WITH LIGHT", rainy night'  # same prompt as the ixim baseline in p0_out


class RamPeak:
    def __init__(self) -> None:
        self.proc, self.peak, self._stop = psutil.Process(), 0, threading.Event()
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


def use_kitchen_path() -> None:
    """Path (ii): ck.int8_linear (rotate + quantize the activation, integer product) instead of decoding the weight."""
    import convrot

    def forward(self, x):
        weight = convrot.unpack_int4(self.weight) if self.packed4 else self.weight
        return convrot.convrot_matmul(x, weight, self.scale, self.bias)

    convrot.ConvRotLinear.forward = forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=FILES, required=True)
    ap.add_argument("--path", choices=("dequant", "kitchen"), default="dequant")
    ap.add_argument("--placement", choices=("gpu", "offload"), default="gpu")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=HERE / "p0_out")
    ap.add_argument("--edit", type=Path, help="source image: run an image.edit job (exercises the vision tower) instead of text-to-image")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    if a.path == "kitchen":
        use_kitchen_path()
    from diffusers import QwenImage21Pipeline
    from qwen_models import build_text_encoder, build_transformer

    result = {"gpu": torch.cuda.get_device_name(0), "variant": a.variant, "path": a.path, "placement": a.placement, "size": a.size, "steps": a.steps}
    torch.cuda.reset_peak_memory_stats()
    with RamPeak() as ram:
        t0 = time.perf_counter()
        device = "cuda" if a.placement == "gpu" else "cpu"
        transformer = build_transformer(ROOT / "transformers" / FILES[a.variant], ROOT / "transformer" / "config.json", device)
        result["transformer_s"] = round(time.perf_counter() - t0, 1)
        text_encoder = build_text_encoder(TEXT_ENCODER, ROOT / "text_encoder", device, with_vision=True)
        result["text_encoder_s"] = round(time.perf_counter() - t0, 1)
        pipe = QwenImage21Pipeline.from_pretrained(
            str(ROOT), transformer=transformer, text_encoder=text_encoder, dtype=torch.bfloat16, local_files_only=True
        )
        if a.placement == "gpu":
            pipe.vae.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        result["load_s"] = round(time.perf_counter() - t0, 1)
        result["vram_after_load_gib"] = gib(torch.cuda.memory_allocated())
        runs = []
        for i in range(a.runs):
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            extra = {}
            prompt = PROMPT
            if a.edit:
                from PIL import Image

                extra["image"] = [Image.open(a.edit).convert("RGB")]
                prompt = "Change the colour of the neon sign from red to blue; keep everything else"
            img = pipe(
                prompt=prompt, width=a.size, height=a.size, num_inference_steps=a.steps,
                generator=torch.Generator("cpu").manual_seed(a.seed), true_cfg_scale=1.0, use_kv_cache=True, **extra,
            ).images[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            f = a.out / f"convrot_{a.variant}_{a.path}_{a.placement}_{a.size}{'_edit' if a.edit else ''}_run{i + 1}.png"
            img.save(f)
            runs.append({
                "run": i + 1, "seconds": round(dt, 1), "vram_peak_alloc_gib": gib(torch.cuda.max_memory_allocated()),
                "size_px": list(img.size), "mode": img.mode, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
            })
        result["runs"] = runs
    result["ram_peak_rss_gib"] = gib(ram.peak)
    print("P0_RESULT " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
