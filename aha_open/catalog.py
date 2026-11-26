"""What each model is, what it can do, what license it carries, and whether it can run on this machine.

Capabilities are the contract: a job whose parameters fall outside them is refused before it is queued
(docs/03_api_v1.md section 5). Nothing here imports torch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import Config, ModelConfig
from .errors import ApiError, invalid
from .hardware import detect_ram_gib, fit_by_tier, fit_for, gpu_budget_mib, recommend_variant_for
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

# What Qwen-Image-2.1 costs, measured on one RTX 3090 Ti 24 GiB at 1024x1024, 40 steps (docs/05_models.md section 3). None = not measured yet.
# The full BF16 model is not offered. Four variants: the community bitsandbytes INT8 build ("int8", its own folder) and three ConvRot
# builds that share one folder ("convrot": text encoder, VAE, configs) and differ only in the transformer file.
#   loras: True = a real LoRA run succeeded on this variant; None = not tried (nothing is refused); False = refused.
#   folder/files/runtime are internal (never in the manifest); the rest is what a client or the web page may show.
QWEN_CONVROT_FOLDER = "convrot"
QWEN_VARIANTS = {
    "int8": {
        "label": "INT8 (bitsandbytes)", "runtime": "bnb", "folder": "int8", "files": (),
        "disk_gib": 18, "vram_gib": {"1k": 9.5, "native": 26.9}, "ram_gib": 20.0, "seconds": {"1k": 39.8}, "loras": True,
        "quality_note": "Community bitsandbytes INT8 build. The fastest of the four here, and the one tried with real LoRAs. "
                        "Needs about 20 GiB of system memory while it loads.",
    },
    "int8-convrot": {
        "label": "INT8 ConvRot", "runtime": "convrot", "folder": QWEN_CONVROT_FOLDER,
        "files": ("transformers/qwen_image_2.1_bf16_int8_convrot.safetensors",),
        "disk_gib": 17, "vram_gib": {"1k": 8.2}, "ram_gib": 23.0, "seconds": {"1k": 51.3}, "loras": None,
        "quality_note": "Best quality of the ConvRot builds: in the publisher's own blind review it was never picked as the worst "
                        "version and was repeatedly called nearly identical to BF16.",
    },
    "mixed-p010": {
        "label": "Mixed W4A4/INT8 (p010)", "runtime": "convrot", "folder": QWEN_CONVROT_FOLDER,
        "files": ("transformers/qwen_image_2.1_bf16_mixed_p010_int8.safetensors",),
        "disk_gib": 17, "vram_gib": {"1k": 8.2}, "ram_gib": 23.0, "seconds": {"1k": 54.7}, "loras": None,
        "quality_note": "The cheapest mix that never looked bad in the publisher's blind review (one rater, 24 scenes); on the hardest "
                        "scenes it can look slightly forced.",
    },
    "mixed-p015": {
        "label": "Mixed W4A4/INT8 (p015)", "runtime": "convrot", "folder": QWEN_CONVROT_FOLDER,
        "files": ("transformers/qwen_image_2.1_bf16_mixed_p015_int8.safetensors",),
        "disk_gib": 15, "vram_gib": {"1k": 8.2}, "ram_gib": 22.0, "seconds": {"1k": 52.3}, "loras": None,
        "quality_note": "Smallest download. In the publisher's blind review it failed the hardest scenes (motion blur, crowded "
                        "interiors) in 6 of 24.",
    },
}
PUBLIC_VARIANT_KEYS = ("label", "disk_gib", "vram_gib", "too_small_up_to_gib", "ram_gib", "seconds", "loras", "quality_note")  # what the manifest shows


def variant_public(info: dict) -> dict:
    return {k: info[k] for k in PUBLIC_VARIANT_KEYS if k in info}


# MiniMax H3, measured with AHA open's own loader on one RTX 3090 Ti 24 GiB (docs/05_models.md section 2.3). Only 16:9 was run, so only
# 16:9 is declared; every side is a multiple of 32 (the video VAE's grid).
H3_SIZES = [
    {"aspect_ratio": "16:9", "short_edge": 768, "width": 1344, "height": 768, "tier": "native"},
    {"aspect_ratio": "16:9", "short_edge": 352, "width": 608, "height": 352, "tier": "preview"},
]
# The three community builds of the transformer that exist on the Hub (the publisher's README also lists a pure INT4 build and INT4/NVFP4
# text encoders: no such files exist). All share one folder (text encoder, both VAEs, configs) and differ only in the transformer file;
# the folder keeps its old name "mixed" so that an existing install is found where it already is.
H3_FOLDER = "mixed"
_H3_FILE = "abiray/MiniMax_H3_{model}_pruned_{build}.safetensors"
_H3_BUILDS = {
    "int8": ("INT8", "int8_convrot", "The publisher calls this the highest quality pruned build. It needs nearly all of a 24 GB card: "
                                     "close other programs that use the GPU."),
    "mixed": ("mixed INT4/INT8", "mixed_int4_int8_convrot", "The publisher's mixed-precision build: smaller than INT8."),
    "nvfp4": ("NVFP4 (FP4)", "nvfp4", "The smallest build. The publisher made it for Blackwell cards (RTX 50); AHA open decodes it to "
                                       "16 bit, which runs on any NVIDIA card."),
}


def _h3_variants(model: str, numbers: dict[str, dict]) -> dict:
    """``numbers``: per build that was run for real, what was measured. A build without an entry is not offered."""
    return {
        name: {"label": label, "folder": H3_FOLDER, "files": (_H3_FILE.format(model=model, build=suffix),), "quality_note": note,
               **numbers[name]}
        for name, (label, suffix, note) in _H3_BUILDS.items()
        if name in numbers
    }


H3_VARIANTS = _h3_variants("FL2VA", {
    # INT8 at 1344x768 ran out of memory on the 24 GiB card, before the first step
    "int8": {"disk_gib": 50, "vram_gib": {"preview": 21.4, "native": None}, "too_small_up_to_gib": {"native": 24}},
    "mixed": {"disk_gib": 46, "vram_gib": {"preview": 16.7, "native": 19.6}},
    "nvfp4": {"disk_gib": 43, "vram_gib": {"preview": 13.1, "native": 16.4}},
})
H3_REF_VARIANTS = _h3_variants("Ref2VA", {
    "mixed": {"disk_gib": 45, "vram_gib": {"preview": 17.1, "native": None}},
    # with a reference: 14.55 GiB peak, but it ran out of memory under a 15.5 GiB cap (the 16 GB card imitated), first denoising step
    "nvfp4": {"disk_gib": 43, "vram_gib": {"preview": 14.6, "native": None}, "too_small_up_to_gib": {"preview": 16}},
})
H3_FRAME_GRID = {"step": 17, "base": 5}  # the video VAE decodes 17*n + 5 frames


def video_num_frames(duration_s: float, fps: int, grid: dict | None) -> int:
    """Frames for a clip: ``round(duration * fps)``, raised to the next ``base + n * step`` when the model has a frame grid.

    MiniMax H3: 5 s at 24 fps -> 120 -> 124 frames (5.17 s).
    """
    frames = max(1, round(duration_s * fps))
    if grid:
        frames = max(grid["base"], frames)
        while (frames - grid["base"]) % grid["step"]:
            frames += 1
    return frames


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
    shares_weights_with: str = ""  # model id whose install folder this model also lives in (one encoder and VAE set on disk, not two)
    required_files: tuple[str, ...] = ()  # files, relative to the install folder, without which the model is "not installed"


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
        ModelSpec(  # test-only: the shape of a model that declares exact sizes, a frame grid and steps (what MiniMax H3 declares)
            id="example-video-sized",
            display_name="Example video model with exact sizes (synthetic)",
            family="fake",
            engine="synthetic",
            tasks=("video.generate",),
            capabilities={
                "durations_s": {"min": 2, "max": 2},
                "fps": 8,
                "short_edges": [64, 32],
                "aspect_ratios": ["16:9"],
                "sizes": [
                    {"aspect_ratio": "16:9", "short_edge": 64, "width": 128, "height": 64, "tier": "native"},
                    {"aspect_ratio": "16:9", "short_edge": 32, "width": 64, "height": 32, "tier": "preview"},
                ],
                "frame_grid": {"step": 4, "base": 1},
                "audio": None,
                "first_frame": False,
                "last_frame": False,
                "max_reference": {"image": 0, "video": 0, "audio": 0, "total": 0},
                "max_outputs": 1,
                "steps": {"min": 2, "max": 2, "default": 2},
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
            # measured peak at 1024x1024 plus headroom (hardware.py); a variant that was not measured yet is not refused
            min_vram_mib={"int8": 10240, "int8-convrot": 9216, "mixed-p010": 9216, "mixed-p015": 9216},
            variants=QWEN_VARIANTS,
        ),
        ModelSpec(
            id="minimax-h3-fl2va",
            display_name="MiniMax H3 (text/frames to video)",
            family="minimax_h3",
            engine="diffusers",
            tasks=("video.generate",),
            # Only what was run for real on the 24 GiB reference GPU is declared (docs/05_models.md section 2.3): the publisher's README
            # lists 4-15 s and six aspect ratios, but nothing else than a 5 s clip at these two 16:9 sizes has been generated here.
            capabilities={
                "durations_s": {"min": 5, "max": 5},  # 124 frames = 5.17 s; longer clips are not measured (and grow fast in time and memory)
                "fps": 24,
                "short_edges": [768, 352],
                "aspect_ratios": ["16:9"],
                "sizes": H3_SIZES,
                "frame_grid": H3_FRAME_GRID,
                "audio": {"channels": 2, "sample_rate": 32000},
                "first_frame": False,  # the worker takes text only for now; a request with a frame is refused with 400, not accepted and failed
                "last_frame": False,
                "max_reference": {"image": 0, "video": 0, "audio": 0, "total": 0},
                "max_outputs": 1,
                "steps": {"min": 20, "max": 20, "default": 20},  # the only value tried; the scheduler makes 19 forward passes of it
            },
            license={
                "name": "MiniMax H3 Community License Agreement",
                "commercial_use": "allowed_with_conditions",
                "excluded_territories": ["EU", "GB", "KR", "US"],
                "required_ui_label": "MiniMax H3",
                "url": "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
                "notes": "Not licensed in the EU, UK, South Korea or the US; extra duties when offered as a hosted service.",
            },
            # Only the community builds are offered, never the ~133 GB full model (docs/05_models.md section 2.0). MiniMax H3 is
            # supported from 16 GB cards (owner, 2026-11-26): NVFP4 ran under a 15.5 GiB cap; INT8 reserves 22 GiB at the small size.
            min_vram_mib={"int8": 23552, "mixed": 18432, "nvfp4": 15872},
            variants=H3_VARIANTS,
            requires_hosted_obligations=True,
        ),
    ]
}

# MiniMax H3 reference-to-video-and-audio ("Ref2VA"): the same sizes, clip length and licence as the text/frame model, its own
# transformer checkpoint (14 GiB) and ordered image/video/audio references. It lives in the folder of minimax-h3-fl2va because the
# encoder, both VAEs and the small configs are identical (docs/05_models.md section 2.4). Only what was run for real is declared.
_FL2VA = CATALOG["minimax-h3-fl2va"]
CATALOG["minimax-h3-ref2va"] = replace(
    _FL2VA,
    id="minimax-h3-ref2va",
    display_name="MiniMax H3 (references to video)",
    capabilities={
        **_FL2VA.capabilities,
        "reference_required": True,  # without a reference this is just the text model: the request is refused, not run
        # With a reference only the preview size was run for real (17.1 GiB, 657 s); native 1344x768 was not, so it is not offered yet.
        "short_edges": [352],
        "sizes": [s for s in _FL2VA.capabilities["sizes"] if s["short_edge"] == 352],
        "max_reference": {"image": 1, "video": 0, "audio": 0, "total": 1},
    },
    # Measured with one reference image at preview: peak 17.07 GiB allocated. Native is not measured with references, so no figure is claimed.
    variants=H3_REF_VARIANTS,
    min_vram_mib={"mixed": 18432, "nvfp4": 18432},  # no build ran with a reference on a 16 GB card: it needs more than the text model
    shares_weights_with="minimax-h3-fl2va",
    required_files=("transformer_ref/config.json",),  # the transformer itself is the variant's own file
)

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
            for index, name, mem in _smi(exe, "index,name,memory.total"):
                gpus.append({"index": int(index), "name": name, "memory_total_mib": int(float(mem))})
        except (OSError, ValueError, subprocess.SubprocessError):
            gpus = []
    if gpus:  # asked apart: a driver too old to know the field must cost this one field, not the whole GPU list
        try:
            caps = {int(i): cap for i, cap in _smi(exe, "index,compute_cap")}
        except (OSError, ValueError, subprocess.SubprocessError):
            caps = {}
        for gpu in gpus:
            gpu["compute_cap"] = caps.get(gpu["index"])
    _gpu_cache = (now, gpus)
    return gpus


def describe_gpu(gpu: dict, separator: str = " ") -> str:
    cap = f", compute capability {gpu['compute_cap']}" if gpu.get("compute_cap") else ""
    return f"{gpu['name']}{separator}{gpu['memory_total_mib']} MiB{cap}"


def _smi(exe: str, fields: str) -> list[list[str]]:
    result = subprocess.run(
        [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"nvidia-smi cannot answer {fields}")
    return [[p.strip() for p in line.split(",")] for line in result.stdout.strip().splitlines()]


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
    spec = CATALOG.get(mc.id)
    base = cfg.data_dir / "models" / ((spec.shares_weights_with if spec else "") or mc.id)
    if not mc.variant:
        return base
    info = (spec.variants.get(mc.variant) if spec else None) or {}
    return base / (info.get("folder") or mc.variant)  # variants with different weights never share a folder; ConvRot ones do on purpose


# ---------------------------------------------------------------------------------------------- state

# Where huggingface_hub keeps the bookkeeping of a download into a folder: a file still arriving is `<name>.incomplete` here.
HF_DOWNLOAD_STATE = Path(".cache") / "huggingface" / "download"


def download_incomplete(wdir: Path) -> bool:
    """True when a download into `wdir` was started and not finished. `model_index.json` arrives among the first files, so its
    presence says nothing about the big ones: a download cut off after it left the model 'ready' on a real run (2026-11-24).

    `install.json` is written only AFTER the download returned successfully, so it is the proof of "finished": `.incomplete` files can
    outlive a successful resume (an earlier cut-off attempt leaves its own, named after the etag, and nothing removes it)."""
    if (wdir / "install.json").exists():
        return False
    state = wdir / HF_DOWNLOAD_STATE
    try:
        return state.is_dir() and any(state.rglob("*.incomplete"))
    except OSError:
        return False


def variant_installed(cfg: Config, spec: ModelSpec, variant: str) -> bool:
    """Are the weights of ``variant`` complete on disk? Independent of which variant is selected in the configuration."""
    wdir = weights_dir(cfg, ModelConfig(id=spec.id, worker=spec.family, variant=variant))
    if wdir is None or not (wdir / "model_index.json").exists():
        return False
    own = (spec.variants.get(variant) or {}).get("files", ())
    return all((wdir / f).is_file() for f in (*spec.required_files, *own)) and not download_incomplete(wdir)


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
        own = (spec.variants.get(mc.variant) or {}).get("files", ()) if mc.variant else ()  # the transformer file of THIS variant
        absent = [f for f in (*own, *spec.required_files) if wdir is not None and not (wdir / f).is_file()]
        if absent:
            return "not_installed", f"{absent[0]} is missing; run `aha-open models install {spec.id} --accept-license <key>`"
        if wdir is not None and download_incomplete(wdir):
            return "not_installed", "the download did not finish; start it again and it continues where it stopped"
    need = spec.min_vram_mib.get(mc.variant or next(iter(spec.min_vram_mib), ""), 0)
    if need:
        gpus = {g["index"]: g for g in detect_gpus()}
        gpu = gpus.get(mc.gpu)
        if gpu is None:
            return "unsupported_hardware", "no CUDA GPU found"
        have = gpu_budget_mib(cfg, gpu)
        if have < need:
            limited = " ([gpu] memory_limit_gib)" if have < gpu["memory_total_mib"] else ""
            least = min(spec.min_vram_mib.values())
            if have < least:
                return "unsupported_hardware", f"needs a GPU of at least {round(least / 1024)} GB; this server has {have / 1024:.0f} GB{limited}"
            fits = ", ".join(n for n, mib in spec.min_vram_mib.items() if mib <= have)
            return "unsupported_hardware", (f"the {mc.variant} variant needs about {need} MiB of GPU memory, found {have} MiB{limited}; "
                                            f"variants that fit here: {fits}")
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
        gpu_mib = gpu_budget_mib(cfg, gpu)
        suggestion, _ = recommend_variant_for(spec.id, gpu_mib, detect_ram_gib())
        entry["variant"] = mc.variant or None
        entry["variants"] = {
            name: {
                **variant_public(info), "selected": name == mc.variant, "recommended": name == suggestion,
                "installed": variant_installed(cfg, spec, name), "fit": fit_for(info, gpu_mib),
                **({"fit_by_tier": by_tier} if (by_tier := fit_by_tier(info, gpu_mib)) else {}),
            }
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
VIDEO_PARAMS = {"aspect_ratio", "duration_s", "short_edge", "seed", "num_outputs", "steps"}
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
        if caps.get("sizes"):  # a model that declares exact sizes (MiniMax H3): the accepted params are complete, a worker never guesses
            entry = next((s for s in caps["sizes"] if s["aspect_ratio"] == ratio and s["short_edge"] == edge), None)
            if entry is None:
                offered = ", ".join(f"{s['aspect_ratio']} at short_edge {s['short_edge']}" for s in caps["sizes"])
                raise invalid(f"no size is offered for aspect_ratio {ratio} with short_edge {edge}; offered: {offered}")
            accepted.update(width=entry["width"], height=entry["height"], num_frames=video_num_frames(dur, caps["fps"], caps.get("frame_grid")))
        steps = params.get("steps")
        if steps is not None:
            rng = caps.get("steps")
            if not rng or not _is_int(steps) or not rng["min"] <= steps <= rng["max"]:
                raise invalid("steps is outside the supported range")
            accepted["steps"] = steps
        elif caps.get("steps"):
            accepted["steps"] = caps["steps"]["default"]
        if "first_frame" in roles and not caps.get("first_frame"):
            raise invalid("this model does not accept a first frame")
        if "last_frame" in roles and not caps.get("last_frame"):
            raise invalid("this model does not accept a last frame")
        for role, key in (("reference", "image"), ("reference_video", "video"), ("reference_audio", "audio")):
            if roles.count(role) > caps["max_reference"].get(key, 0):
                raise invalid(f"too many {role} inputs for this model")
        if roles.count("first_frame") > 1 or roles.count("last_frame") > 1:
            raise invalid("at most one first_frame and one last_frame are accepted")
        given = [r for r in roles if r in ("reference", "reference_video", "reference_audio")]
        if len(given) > caps["max_reference"].get("total", 0):
            raise invalid(f"at most {caps['max_reference'].get('total', 0)} references are accepted in all")
        if caps.get("reference_required") and not (roles.count("reference") + roles.count("reference_video")):
            raise invalid("this model needs at least one reference image or video (an audio reference cannot stand alone); "
                          "for text only use minimax-h3-fl2va")
    return accepted
