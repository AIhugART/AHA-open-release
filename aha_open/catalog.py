"""What each model is, what it can do, what license it carries, and whether it can run on this machine.

Capabilities are the contract: a job whose parameters fall outside them is refused before it is queued
(docs/03_api_v1.md section 5). Nothing here imports torch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, ModelConfig
from .errors import ApiError, invalid
from .hardware import detect_ram_gib, recommend_variant
from .loras import scan as scan_loras

ROOT = Path(__file__).resolve().parents[1]
# Worker sources. In an installed wheel they sit inside the package (pyproject maps `workers/` to `aha_open/workers`);
# in a source checkout they are `workers/` at the repository root, next to the developer's `.venv` for each worker.
WORKERS_DIR = Path(__file__).resolve().parent / "workers" if (Path(__file__).resolve().parent / "workers").is_dir() else ROOT / "workers"

# Two tiers per aspect ratio. "native" are the model's own sizes (~4 MP); "1k" are the same ratios at ~1 MP.
# On a 24 GiB GPU the 1k tier takes ~40 s and fits in
# VRAM, the native tier spills past VRAM and takes minutes (docs/05_models.md section 3). With only an aspect_ratio given,
# the first match wins, so the 1k tier is the default; ask for width/height to get a native size.
_QWEN_NATIVE = [
    ("1:1", 2048, 2048), ("4:3", 2400, 1792), ("3:4", 1792, 2400), ("3:2", 2528, 1696),
    ("2:3", 1696, 2528), ("16:9", 2752, 1536), ("9:16", 1536, 2752),
]
# ~1 MP at the same aspect ratio. Every side must be a multiple of 32: the pipeline rounds DOWN to 32 (a real run asked for
# 1200x896 and got 1184x896, which the gateway refused), so half of a native side is not always allowed.
_QWEN_1K = [
    ("1:1", 1024, 1024), ("4:3", 1152, 864), ("3:4", 864, 1152), ("3:2", 1248, 832),
    ("2:3", 832, 1248), ("16:9", 1376, 768), ("9:16", 768, 1376),
]
QWEN_SIZES = [
    {"aspect_ratio": r, "width": w, "height": h, "tier": "1k"} for r, w, h in _QWEN_1K
] + [{"aspect_ratio": r, "width": w, "height": h, "tier": "native"} for r, w, h in _QWEN_NATIVE]

# LoRA adapters the operator put in <data_dir>/loras (aha_open/loras.py). Clients pick them by id from the manifest.
LORA_CAPS = {"max": 3, "scale": {"min": 0.0, "max": 2.0, "default": 1.0}, "format": "safetensors"}

# What Qwen-Image-2.1 costs (INT8 is the only variant offered; the full BF16 model was dropped), measured on one RTX 3090 Ti 24 GiB (docs/05_models.md section 3). None = not measured yet.
QWEN_VARIANTS = {
    # "loras": True = a real LoRA run succeeded on this variant; None = not tried (nothing is refused); False = refused.
    "int8": {"label": "int8", "disk_gib": 18, "vram_gib": {"1k": 9.5, "native": 26.9}, "loras": True},
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    family: str  # worker family: fake | qwen_image | minimax_h3
    engine: str
    tasks: tuple[str, ...]
    capabilities: dict
    license: dict
    min_vram_mib: dict = field(default_factory=dict)  # variant -> MiB; empty = no GPU needed
    variants: dict = field(default_factory=dict)  # variant -> {label, disk_gib, vram_gib{tier}}: what the operator may choose
    requires_hosted_obligations: bool = False
    internal: bool = False  # test-only models, never listed unless configured


CATALOG: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="example-image",
            display_name="Example image model (synthetic)",
            family="fake",
            engine="synthetic",
            tasks=("image.generate", "image.edit"),
            capabilities={
                "sizes": [
                    {"aspect_ratio": "1:1", "width": 256, "height": 256},
                    {"aspect_ratio": "16:9", "width": 320, "height": 180},
                ],
                "max_reference_images": 3,
                "mask": True,
                "rgba": False,
                "max_outputs": 2,
                "steps": {"min": 1, "max": 50, "default": 4},
                "loras": LORA_CAPS,
            },
            license={"name": "Apache-2.0 (synthetic test model)", "commercial_use": "allowed"},
            internal=True,
        ),
        ModelSpec(
            id="example-video",
            display_name="Example video model (synthetic)",
            family="fake",
            engine="synthetic",
            tasks=("video.generate",),
            capabilities={
                "durations_s": {"min": 1, "max": 4},
                "fps": 8,
                "short_edges": [64],
                "aspect_ratios": ["1:1", "16:9"],
                "audio": None,
                "first_frame": True,
                "last_frame": True,
                "max_reference": {"image": 0, "video": 0, "audio": 0, "total": 0},
                "max_outputs": 1,
            },
            license={"name": "Apache-2.0 (synthetic test model)", "commercial_use": "allowed"},
            internal=True,
        ),
        ModelSpec(
            id="qwen-image-2.1",
            display_name="Qwen-Image-2.1",
            family="qwen_image",
            engine="diffusers",
            tasks=("image.generate", "image.edit"),
            capabilities={
                "sizes": QWEN_SIZES,
                "max_reference_images": 10,
                # No upstream API takes a mask (the README only says "separate masks"); not claimed until a real run
                # shows how it behaves. Painted annotations on the source image are an ordinary image.
                "mask": False,
                "rgba": True,
                "max_outputs": 1,
                "steps": {"min": 10, "max": 60, "default": 40},
                "loras": LORA_CAPS,
            },
            license={
                "name": "Qwen RESEARCH LICENSE AGREEMENT",
                "commercial_use": "not_allowed",
                "url": "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
                "notes": "Research or evaluation use only; a separate license is needed for commercial use.",
            },
            min_vram_mib={"int8": 10240},  # measured peak at 1024x1024 plus headroom (hardware.py)
            variants=QWEN_VARIANTS,
        ),
        ModelSpec(
            id="minimax-h3-fl2va",
            display_name="MiniMax H3 (text/frames to video)",
            family="minimax_h3",
            engine="diffusers",
            tasks=("video.generate",),
            capabilities={
                "durations_s": {"min": 4, "max": 15},
                "fps": 24,
                "short_edges": [768],
                "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                "audio": {"channels": 2, "sample_rate": 32000},
                "first_frame": True,
                "last_frame": True,
                "max_reference": {"image": 0, "video": 0, "audio": 0, "total": 0},
                "max_outputs": 1,
            },
            license={
                "name": "MiniMax H3 Community License Agreement",
                "commercial_use": "allowed_with_conditions",
                "excluded_territories": ["EU", "GB", "KR", "US"],
                "required_ui_label": "MiniMax H3",
                "url": "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
                "notes": "Not licensed in the EU, UK, South Korea or the US; extra duties when offered as a hosted service.",
            },
            # Only the community INT4/INT8 build will be used, never the ~133 GB full model (docs/05_models.md section 2.0).
            # Its VRAM need has not been measured, and an unmeasured number must not refuse anything: empty on purpose.
            min_vram_mib={},
            requires_hosted_obligations=True,
        ),
    ]
}

_VALID_STATES = {"ready", "loading", "not_installed", "unsupported_hardware", "error"}


# ---------------------------------------------------------------------------------------------- hardware

_gpu_cache: tuple[float, list[dict]] = (0.0, [])


def detect_gpus(ttl_s: float = 30.0) -> list[dict]:
    """GPUs via nvidia-smi (never torch). Empty list when there is none; cached to keep /v1/health cheap."""
    global _gpu_cache
    now = time.monotonic()
    if now - _gpu_cache[0] < ttl_s and _gpu_cache[0] != 0.0:
        return _gpu_cache[1]
    gpus: list[dict] = []
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout
            for line in out.strip().splitlines():
                index, name, mem = [p.strip() for p in line.split(",")]
                gpus.append({"index": int(index), "name": name, "memory_total_mib": int(float(mem))})
        except (OSError, ValueError, subprocess.SubprocessError):
            gpus = []
    _gpu_cache = (now, gpus)
    return gpus


# ---------------------------------------------------------------------------------------------- workers


def worker_python(cfg: Config, mc: ModelConfig) -> Path | None:
    if mc.python:
        return Path(mc.python)
    if mc.worker == "fake":
        return Path(sys.executable)
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    for base in (cfg.data_dir / "workers" / mc.worker / "venv", WORKERS_DIR / mc.worker / ".venv"):
        candidate = base / scripts
        if candidate.exists():
            return candidate
    return None


def worker_command(cfg: Config, mc: ModelConfig) -> list[str]:
    python = worker_python(cfg, mc)
    script = WORKERS_DIR / mc.worker / "worker.py"
    if python is None or not script.exists():
        raise ApiError(409, "model_not_installed", f"worker for {mc.id} is not installed")
    return [str(python), str(script)]


def weights_dir(cfg: Config, mc: ModelConfig) -> Path | None:
    if mc.weights_dir:
        return Path(mc.weights_dir)
    if mc.worker == "fake":
        return None
    base = cfg.data_dir / "models" / mc.id
    return base / mc.variant if mc.variant else base  # one directory per variant: switching never mixes weights


# ---------------------------------------------------------------------------------------------- state


def model_state(cfg: Config, mc: ModelConfig, spec: ModelSpec, loading: bool = False) -> tuple[str, str | None]:
    """(state, detail) for one configured model on this machine."""
    if spec.requires_hosted_obligations and cfg.profile == "internet" and cfg.hosted_service_obligations != "accepted":
        return "error", "hosted_service_obligations_not_accepted"
    if spec.variants and mc.variant and mc.variant not in spec.variants:
        return "error", (
            f"variant {mc.variant!r} is not offered any more (only: {', '.join(spec.variants)}); "
            f"run `aha-open models install {spec.id} --accept-license <key>`"
        )
    if spec.family != "fake":
        if worker_python(cfg, mc) is None or not (WORKERS_DIR / mc.worker / "worker.py").exists():
            return "not_installed", "worker environment not found"
        wdir = weights_dir(cfg, mc)
        if wdir is not None and not (wdir / "model_index.json").exists():
            return "not_installed", f"weights not found at {wdir}"
    need = spec.min_vram_mib.get(mc.variant or next(iter(spec.min_vram_mib), ""), 0)
    if need:
        gpus = {g["index"]: g for g in detect_gpus()}
        gpu = gpus.get(mc.gpu)
        if gpu is None:
            return "unsupported_hardware", "no CUDA GPU found"
        if gpu["memory_total_mib"] < need:
            return "unsupported_hardware", f"needs about {need} MiB of GPU memory, found {gpu['memory_total_mib']} MiB"
    return ("loading", None) if loading else ("ready", None)


def manifest_model(cfg: Config, mc: ModelConfig, spec: ModelSpec, loading: bool = False) -> dict:
    state, detail = model_state(cfg, mc, spec, loading)
    entry = {
        "id": spec.id,
        "display_name": spec.display_name,
        "engine": spec.engine,
        "tasks": list(spec.tasks),
        "state": state,
        "state_detail": detail,
        "capabilities": spec.capabilities,
        "license": spec.license,
    }
    if "loras" in spec.capabilities:  # what is offered depends on the files on this server, so it is read per request
        available, _ = scan_loras(cfg)
        entry["capabilities"] = {
            **spec.capabilities,
            "loras": {**spec.capabilities["loras"], "available": [lora.public() for lora in available.values()]},
        }
    if spec.variants:  # additive fields (docs/03_api_v1.md section 4): which variant runs here, and what the others cost
        gpu = {g["index"]: g for g in detect_gpus()}.get(mc.gpu)
        suggestion, _ = recommend_variant(gpu["memory_total_mib"] if gpu else None, detect_ram_gib())
        entry["variant"] = mc.variant or None
        entry["variants"] = {
            name: {**info, "selected": name == mc.variant, "recommended": name == suggestion}
            for name, info in spec.variants.items()
        }
    return entry


def enabled_models(cfg: Config) -> list[tuple[ModelConfig, ModelSpec]]:
    out = []
    for mc in cfg.models:
        spec = CATALOG.get(mc.id)
        if mc.enabled and spec is not None:
            out.append((mc, spec))
    return out


# ---------------------------------------------------------------------------------------------- validation

IMAGE_PARAMS = {"width", "height", "aspect_ratio", "seed", "num_outputs", "steps", "transparent", "loras"}
VIDEO_PARAMS = {"aspect_ratio", "duration_s", "short_edge", "seed", "num_outputs"}
ROLES_BY_TASK = {
    "image.generate": set(),
    "image.edit": {"source", "reference", "mask"},
    "video.generate": {"first_frame", "last_frame", "reference", "reference_video", "reference_audio"},
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_image_size(caps: dict, params: dict) -> dict:
    sizes = caps.get("sizes", [])
    width, height, ratio = params.get("width"), params.get("height"), params.get("aspect_ratio")
    if (width is None) != (height is None):
        raise invalid("width and height must be given together")
    if width is not None:
        for entry in sizes:
            if entry["width"] == width and entry["height"] == height:
                return entry
        allowed = ", ".join(f"{s['width']}x{s['height']}" for s in sizes)
        raise invalid(f"size {width}x{height} is not supported (allowed: {allowed})")
    if ratio is not None:
        for entry in sizes:
            if entry["aspect_ratio"] == ratio:
                return entry
        raise invalid(f"aspect_ratio {ratio!r} is not supported")
    return sizes[0]


def validate_loras(spec: ModelSpec, variant: str, value: object, available: frozenset[str]) -> list[dict]:
    caps = spec.capabilities.get("loras")
    if not caps:
        raise invalid("this model does not accept LoRAs")
    if variant and spec.variants.get(variant, {}).get("loras") is False:
        raise invalid(f"LoRAs are not supported with the {variant} variant on this server")
    if not isinstance(value, list) or not all(isinstance(i, dict) for i in value):
        raise invalid('loras must be a list like [{"id": "...", "scale": 0.8}]')
    if len(value) > caps["max"]:
        raise invalid(f"at most {caps['max']} LoRAs can be used in one job")
    lo, hi = caps["scale"]["min"], caps["scale"]["max"]
    chosen, seen = [], set()
    for item in value:
        unknown = sorted(set(item) - {"id", "scale"})
        if unknown:
            raise invalid(f"unknown LoRA field(s): {', '.join(unknown)}")
        lora_id = item.get("id")
        if not isinstance(lora_id, str) or lora_id not in available:
            raise invalid(f"LoRA {lora_id!r} is not available on this server (see capabilities.loras.available in the manifest)")
        if lora_id in seen:
            raise invalid(f"LoRA {lora_id!r} is listed twice")
        seen.add(lora_id)
        scale = item.get("scale", caps["scale"]["default"])
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not lo < scale <= hi:
            raise invalid(f"the scale of LoRA {lora_id!r} must be above {lo} and at most {hi}")
        chosen.append({"id": lora_id, "scale": float(scale)})
    return chosen


def validate_job(
    spec: ModelSpec, task: str, prompt: str | None, params: dict, inputs: list[dict],
    available_loras: frozenset[str] = frozenset(), variant: str = "",
) -> dict:
    """Return the accepted (normalised) params, or raise 400. No silent resizing, ever (docs/04_architecture.md section 8)."""
    if task not in spec.tasks:
        raise ApiError(400, "unsupported_task", f"model {spec.id} does not support {task}")
    caps = spec.capabilities
    if not isinstance(params, dict):
        raise invalid("params must be an object")
    kind = "image" if task.startswith("image.") else "video"
    allowed = IMAGE_PARAMS if kind == "image" else VIDEO_PARAMS
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise invalid(f"unknown params: {', '.join(unknown)}")
    if task in ("image.generate", "video.generate") and (not isinstance(prompt, str) or not prompt.strip()):
        raise invalid("prompt is required")
    if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 8000):
        raise invalid("prompt must be a string of at most 8000 characters")
    seed = params.get("seed")
    if seed is not None and not (_is_int(seed) and 0 <= seed < 2**63):
        raise invalid("seed must be a non-negative integer")
    n = params.get("num_outputs", 1)
    if not _is_int(n) or not 1 <= n <= caps.get("max_outputs", 1):
        raise invalid(f"num_outputs must be between 1 and {caps.get('max_outputs', 1)}")

    accepted: dict = {"num_outputs": n}
    if seed is not None:
        accepted["seed"] = seed

    roles = [i["role"] for i in inputs]
    bad_roles = sorted(set(roles) - ROLES_BY_TASK.get(task, set()))
    if bad_roles:
        raise invalid(f"input role(s) not accepted for {task}: {', '.join(bad_roles)}")

    if kind == "image":
        entry = resolve_image_size(caps, params)
        accepted.update(width=entry["width"], height=entry["height"], aspect_ratio=entry["aspect_ratio"])
        if "loras" in params:
            accepted["loras"] = validate_loras(spec, variant, params["loras"], available_loras)
        transparent = params.get("transparent")
        if transparent is not None:
            if not isinstance(transparent, bool):
                raise invalid("transparent must be true or false")
            if transparent and not caps.get("rgba"):
                raise invalid("this model cannot produce transparency")
            accepted["transparent"] = transparent
        steps = params.get("steps")
        if steps is not None:
            rng = caps.get("steps")
            if not rng or not _is_int(steps) or not rng["min"] <= steps <= rng["max"]:
                raise invalid("steps is outside the supported range")
            accepted["steps"] = steps
        elif caps.get("steps"):
            accepted["steps"] = caps["steps"]["default"]  # the accepted params are complete: a worker never guesses
        if task == "image.edit":
            if roles.count("source") != 1:
                raise invalid("image.edit needs exactly one source input")
            if roles.count("mask") > 1:
                raise invalid("at most one mask is accepted")
            if "mask" in roles and not caps.get("mask"):
                raise invalid("this model does not accept a mask")
            if roles.count("source") + roles.count("reference") > caps.get("max_reference_images", 0) + 1:
                raise invalid(f"at most {caps.get('max_reference_images', 0)} reference images are accepted")
    else:
        ratio = params.get("aspect_ratio", caps["aspect_ratios"][0])
        if ratio not in caps["aspect_ratios"]:
            raise invalid(f"aspect_ratio {ratio!r} is not supported")
        dur = params.get("duration_s", caps["durations_s"]["min"])
        if not isinstance(dur, (int, float)) or isinstance(dur, bool) or not (
            caps["durations_s"]["min"] <= dur <= caps["durations_s"]["max"]
        ):
            raise invalid(f"duration_s must be between {caps['durations_s']['min']} and {caps['durations_s']['max']}")
        edge = params.get("short_edge", caps["short_edges"][0])
        if edge not in caps["short_edges"]:
            raise invalid(f"short_edge must be one of {caps['short_edges']}")
        accepted.update(aspect_ratio=ratio, duration_s=dur, short_edge=edge)
        if "first_frame" in roles and not caps.get("first_frame"):
            raise invalid("this model does not accept a first frame")
        if "last_frame" in roles and not caps.get("last_frame"):
            raise invalid("this model does not accept a last frame")
        for role, key in (("reference", "image"), ("reference_video", "video"), ("reference_audio", "audio")):
            if roles.count(role) > caps["max_reference"].get(key, 0):
                raise invalid(f"too many {role} inputs for this model")
        if roles.count("first_frame") > 1 or roles.count("last_frame") > 1:
            raise invalid("at most one first_frame and one last_frame are accepted")
    return accepted
