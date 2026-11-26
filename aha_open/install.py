"""`aha-open models install`: licence acknowledgement, variant choice, weights, worker environment, config.

The operator downloads the weights and accepts the licence; AHA open never ships or mirrors them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .catalog import (
    CATALOG,
    HF_DOWNLOAD_STATE,
    WORKERS_DIR,
    detect_gpus,
    download_incomplete,
    variant_installed,
    weights_dir,
    worker_python,
)
from .config import Config, ModelConfig, render
from .hardware import detect_ram_gib, fit_for, free_disk_gib, gpu_budget_mib, recommend_variant_for

# MiniMax H3 comes from three repositories (docs/05_models.md section 2.0): the community transformer and VAEs (Abiray), the text
# encoder Abiray does not have (Comfy-Org), and the small configuration/tokenizer files of the official repository. Each part names
# its files exactly; the large ones carry the sha256 the publisher's Hub records, checked after the download.
_H3_OFFICIAL = {
    "repo": "MiniMaxAI/MiniMax-H3", "revision": "12253298adb81e36757eabebb81d2ba1c0cc70cc", "subdir": "",
    "patterns": [  # small files only (configs, tokenizer, processor, schedulers, licence); model_index.json marks "weights present"
        "model_index.json", "LICENSE", "transformer/config.json", "transformer_ref/config.json", "vae/config.json", "audio_vae/config.json",
        "scheduler/scheduler_config.json", "audio_scheduler/scheduler_config.json",
        "FL2VA/text_encoder/config.json", "FL2VA/tokenizer/*", "FL2VA/processor/*",
    ],
    "files": {},
}
_H3_VAES = {
    "vae/minimax_h3_video_vae_fp16.safetensors": (5207808557, "88539d729fd7e9f9de765498428cd8e9152b8420317ec0e7542a362fffb7e0e1"),
    "vae/minimax_h3_audio_vae_fp32.safetensors": (605254869, "83043bf3afccf4cab4d7e114f6e04607e9687bef16930a2f6088d721a893d3c4"),
}
_H3_ENCODER = {
    "repo": "Comfy-Org/MiniMax-H3", "revision": "4b53a35191437de05370da548e59aeabe2a5a09a", "subdir": "comfyorg", "patterns": [],
    "files": {
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors":
            (27141342152, "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"),
    },
}

# the transformer file of each (model, build): size and sha256 as the publisher's Hub records them at the pinned revision
_H3_TRANSFORMERS = {
    ("FL2VA", "int8"): (20970379688, "f07a54277ca85b59e5243a924fb5c54d79768f820fc4155c1b614b6f613e9dbf"),
    ("FL2VA", "mixed"): (15903012791, "541c7797d815a165a3b71fa6cfe03cafa1ba658cb03af1dc9f17cfe3985e7f99"),
    ("FL2VA", "nvfp4"): (12528636865, "6ab7f0c48141e7919b32f925ca3def22e06a6aebeb9e0b6f5a0be0fe8409976f"),
    ("Ref2VA", "int8"): (20970379689, "ed7e9aa541b945f44298650f0dc2e2e867bfd3b67cd07bcc90f6081c168f0d3d"),
    ("Ref2VA", "mixed"): (15093774276, "434c372cbb68ff055ffb5aa2c35de42211d19ebacc2ae8e9ae9202b00ee48afb"),
    ("Ref2VA", "nvfp4"): (12528636866, "3e1be702c95bc057c05a7d1867e8aeea33073dcf5743835f2f27f06a2f34c596"),
}


def _h3_plan(model_id: str, model: str) -> dict:
    """One install plan per build the catalog offers for this model: the shared parts plus that build's transformer."""
    plans = {}
    for name, info in CATALOG[model_id].variants.items():
        file = info["files"][0].removeprefix("abiray/")
        plans[name] = {
            "disk_gib": info["disk_gib"],
            "parts": [
                _H3_OFFICIAL,
                {
                    "repo": "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot", "revision": "908eccad7e68751190d04c171956f163bfeed741",
                    "subdir": "abiray", "patterns": [], "files": {file: _H3_TRANSFORMERS[model, name], **_H3_VAES},
                },
                _H3_ENCODER,
            ],
        }
    return plans


# Qwen-Image-2.1, ConvRot builds (docs/05_models.md section 3): the official repository gives the configs, tokenizer and VAE, Comfy-Org
# the INT8 ConvRot text encoder, JoaoZaokk the transformer. The first two parts are the same for all three variants and live in one
# folder, so a second variant downloads only its own transformer file.
_QWEN_OFFICIAL = {
    "repo": "Qwen/Qwen-Image-2.1", "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3", "subdir": "",
    "patterns": [
        "model_index.json", "LICENSE", "processor/*", "scheduler/scheduler_config.json", "text_encoder/config.json",
        "text_encoder/generation_config.json", "transformer/config.json", "vae/config.json",
    ],
    "files": {"vae/diffusion_pytorch_model.safetensors": (1350989512, "a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417")},
}
_QWEN_ENCODER = {
    "repo": "Comfy-Org/Qwen-Image-2.1", "revision": "aacf31a6d48e707f352ecba714cee0680da3479b", "subdir": "comfyorg", "patterns": [],
    "files": {
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors":
            (9350798360, "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"),
    },
}
_QWEN_TRANSFORMER_REPO = ("JoaoZaokk/ConvRot-Mixed-INT8-for-Qwen-Image-2.1", "ea9b661848a5bb89ebd23f3e86c9648622bd8f15")


def _qwen_convrot(name: str, size: int, sha256: str, disk_gib: int) -> dict:
    repo, revision = _QWEN_TRANSFORMER_REPO
    return {
        "disk_gib": disk_gib,
        "parts": [
            _QWEN_OFFICIAL, _QWEN_ENCODER,
            {
                "repo": repo, "revision": revision, "subdir": "transformers",
                "patterns": ["NOTICE.txt", "LICENSE_QWEN_RESEARCH.txt", f"{name}.quant.json"],
                "files": {f"{name}.safetensors": (size, sha256)},
            },
        ],
    }


# What may be installed. Revisions are pinned: a different upstream revision is a different (untested) model.
INSTALLABLE = {
    "qwen-image-2.1": {
        "license_key": "qwen-research",
        "variants": {
            "int8": {"repo": "ixim/Image21-INT8", "revision": "099a9acacd0f1ae640ad4c740aed3af9fd12be19", "disk_gib": 18},
            "int8-convrot": _qwen_convrot(
                "qwen_image_2.1_bf16_int8_convrot", 7256770456,
                "c4d1b6506c9a5ba897786c21e59fa67a41e836e92638b84872ea98111d72e72a", 17),
            "mixed-p010": _qwen_convrot(
                "qwen_image_2.1_bf16_mixed_p010_int8", 6552125872,
                "705b72bec8e0c643ab3e0fac081ed1c9ab5b79264aea57f9ecad25e3a42f3b49", 17),
            "mixed-p015": _qwen_convrot(
                "qwen_image_2.1_bf16_mixed_p015_int8", 4605967624,
                "809d1bb659de614c200c21fff0a1625f2a0a7b318b864bcdc2c727228213735a", 15),
        },
    },
    "minimax-h3-fl2va": {
        "license_key": "minimax-h3",
        "territory": True,  # the licence excludes territories: the operator confirms with --confirm-territory (a checkbox on the web page)
        "variants": _h3_plan("minimax-h3-fl2va", "FL2VA"),
    },
    # Reference-to-video-and-audio. Same folder as minimax-h3-fl2va (aha_open/catalog.py `shares_weights_with`): the encoder, VAEs and
    # configs are the same files, so a machine that has the text model only downloads the transformer.
    "minimax-h3-ref2va": {
        "license_key": "minimax-h3",
        "territory": True,
        "variants": _h3_plan("minimax-h3-ref2va", "Ref2VA"),
    },
}


def _recorded_variants(target: Path) -> list[str]:
    """Variants a folder's install.json already vouches for (ConvRot variants share one folder). Unreadable = none."""
    try:
        return [str(v) for v in json.loads((target / "install.json").read_text(encoding="utf-8")).get("variants", [])]
    except (OSError, ValueError, AttributeError):
        return []


def _missing_gib(parts: list[dict], target: Path) -> float:
    """GiB of pinned files that are not (completely) on disk yet; a file whose size is right counts as present."""
    return sum(_missing_bytes(part, target) for part in parts) / 2**30


def _missing_bytes(part: dict, target: Path) -> int:
    dest = target / part["subdir"] if part["subdir"] else target
    return sum(size for name, (size, _) in part["files"].items()
               if not ((dest / name).is_file() and (dest / name).stat().st_size == size))


def _say(text: str) -> None:
    print(text)


def uv_executable() -> str | None:
    """The `uv` that builds worker environments: `AHA_OPEN_UV` (the Windows installer points it at the copy it ships), else PATH."""
    named = os.environ.get("AHA_OPEN_UV", "")
    if named:
        return named if Path(named).is_file() else None
    found = shutil.which("uv")
    if found:
        return found
    # The Windows installer puts its own uv next to the interpreter (<app>\venv\Scripts\uv.exe), where a pip-installed uv would be.
    sibling = Path(sys.executable).with_name("uv.exe" if sys.platform == "win32" else "uv")
    return str(sibling) if sibling.is_file() else None


def install_model(cfg: Config, args: argparse.Namespace) -> int:
    spec = CATALOG.get(args.model)
    plan = INSTALLABLE.get(args.model)
    if spec is None or spec.internal:
        _say(f"error: unknown model {args.model!r}")
        return 2
    if plan is None:
        _say(f"{spec.display_name} cannot be installed by this version yet (deferred; see docs/05_models.md).")
        return 2

    lic = spec.license
    key = plan["license_key"]
    if args.accept_license != key:
        _say(
            f"{spec.display_name} is licensed as: {lic['name']}\n"
            f"  commercial use: {lic['commercial_use']}\n  {lic.get('notes', '')}\n  full text: {lic.get('url', '')}\n\n"
            f"You, the operator, are responsible for complying with it. To confirm that you read it, run again with:\n"
            f"  aha-open models install {args.model} --accept-license {key}"
        )
        return 2
    if plan.get("territory") and not getattr(args, "confirm_territory", False):
        _say(
            f"The {lic['name']} does not allow use in: {', '.join(lic.get('excluded_territories', []))}.\n"
            f"If you are not in any of them, run again adding:\n"
            f"  aha-open models install {args.model} --accept-license {key} --confirm-territory"
        )
        return 2

    gpus = detect_gpus(ttl_s=0)
    ram = detect_ram_gib()
    gpu_mib = gpu_budget_mib(cfg, gpus[0] if gpus else None)
    variant, reason = recommend_variant_for(args.model, gpu_mib, ram)
    wanted = getattr(args, "variant", None)
    if wanted:  # the operator chose: it is theirs to decide, so a poor fit is said, not refused (an unmeasured number never refuses)
        if wanted not in plan["variants"]:
            _say(f"error: {args.model} has no variant {wanted!r}; choose one of: {', '.join(plan['variants'])}")
            return 2
        fit = fit_for(spec.variants.get(wanted, {}), gpu_mib)
        variant, reason = wanted, ("it needs more GPU memory than this machine has; it may not run" if fit == "too_small"
                                   else f"you chose it (fit on this machine: {fit})")
    elif variant is None and not args.from_dir:
        _say(f"error: {reason}")  # --from-dir says the weights are already on disk; `aha-open doctor` checks the GPU afterwards
        return 2
    variant = variant or next(iter(plan["variants"]))
    _say(f"Installing {variant}: {reason}")

    info = plan["variants"][variant]
    target = None
    earlier: list[str] = []
    if args.from_dir:
        target = Path(args.from_dir).resolve()
        earlier = _recorded_variants(target)
        if not (target / "model_index.json").exists():
            _say(f"error: {target} has no model_index.json; it is not a {spec.display_name} download")
            return 2
    else:
        target = weights_dir(cfg, ModelConfig(id=args.model, worker=spec.family, variant=variant))
        earlier = _recorded_variants(target)  # before the download starts: it removes install.json
        free = free_disk_gib(target)
        need_gib = info["disk_gib"] if "parts" not in info else min(info["disk_gib"], _missing_gib(info["parts"], target) + 1)
        if free < need_gib * 1.1:
            _say(f"error: {need_gib:.0f} GB are needed for {variant} but only {free:.0f} GB are free where {target} lives")
            return 2
        # `model_index.json` is among the first files to arrive, so it proves nothing: after an interrupted download the big files are
        # missing. Unfinished = a file still marked incomplete, or hf began downloading here and no install.json says it ended.
        started_by_hf = (target / HF_DOWNLOAD_STATE).is_dir()
        unfinished = download_incomplete(target) or (started_by_hf and not (target / "install.json").exists())
        files_missing = "parts" in info and _missing_gib(info["parts"], target) > 0  # e.g. the text model is here, the Ref2VA file is not
        if unfinished or files_missing or not (target / "model_index.json").exists():
            (target / "install.json").unlink(missing_ok=True)  # a download starts: an older record must not vouch for it
            if "parts" in info:  # resumes too: snapshot_download skips files already complete
                code = _download_parts(info["parts"], target)
            else:
                code = _download(info["repo"], info["revision"], target)
            if code:
                return code
    if "parts" in info:  # also for --from-dir: a file that is not the pinned one is never vouched for by install.json
        code = _verify_parts(info["parts"], target)
        if code:
            return code
    record = {"model": args.model, "variant": variant, "variants": sorted({*earlier, variant}), "license_key": key,
              "installed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if "parts" in info:
        record["parts"] = [{"repo": p["repo"], "revision": p["revision"], "files": sorted(p["files"])} for p in info["parts"]]
    else:
        record.update(repo=info["repo"], revision=info["revision"])
    (target / "install.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    mc = cfg.model(args.model)
    if mc is None:
        mc = ModelConfig(id=args.model, worker=spec.family)
        cfg.models.append(mc)
    # Never switch the running variant behind the operator's back: a second variant is only added; `models use` selects it.
    keeps = bool(mc.variant) and mc.variant != variant and mc.variant in spec.variants and variant_installed(cfg, spec, mc.variant)
    if not keeps:
        mc.variant = variant
    mc.enabled = True
    mc.weights_dir = str(target) if args.from_dir else ""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "config.toml").write_text(render(cfg), encoding="utf-8")

    if not args.skip_env and worker_python(cfg, mc) is None:
        code = _make_worker_env(cfg, spec.family)
        if code:
            return code
    notice = target / "transformers" / "NOTICE.txt"  # the publisher's modification notice travels with the weights (Qwen licence)
    if notice.is_file():
        _say("Notice from the publisher of this build:\n  " + notice.read_text(encoding="utf-8", errors="replace").strip().replace("\n", "\n  "))
    if keeps:
        _say(f"Installed {spec.display_name} ({variant}). The server still uses {mc.variant}; switch with "
             f"`aha-open models use {args.model} {variant}` (restart the server to apply).")
    else:
        _say(f"Installed {spec.display_name} ({variant}). Check with `aha-open doctor`, then start with `aha-open serve`.")
    return 0


def _dir_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30 if path.exists() else 0.0


def use_variant(cfg: Config, model_id: str, variant: str) -> tuple[bool, str]:
    """Select an installed variant in the configuration. The running server keeps its worker until it is restarted."""
    spec = CATALOG.get(model_id)
    if spec is None or spec.internal or not spec.variants:
        return False, f"{model_id!r} has no variants to choose from"
    if variant not in spec.variants:
        return False, f"{model_id} has no variant {variant!r}; choose one of: {', '.join(spec.variants)}"
    if not variant_installed(cfg, spec, variant):
        return False, f"{variant} is not installed; run `aha-open models install {model_id} --variant {variant} --accept-license <key>`"
    mc = cfg.model(model_id)
    if mc is None:
        mc = ModelConfig(id=model_id, worker=spec.family)
        cfg.models.append(mc)
    mc.variant, mc.enabled = variant, True
    mc.weights_dir = ""  # a variant lives in its own place under the data folder; an old --from-dir pointer would name another one
    (cfg.data_dir / "config.toml").write_text(render(cfg), encoding="utf-8")
    return True, f"{model_id} now uses {variant}; restart the server to load it"


def _others_in_folder(cfg: Config, spec, variant: str) -> list[str]:
    """Installed builds, of this model or of another one sharing its weights folder, that live in the same folder as ``variant``.

    The two MiniMax H3 models keep one text encoder and one set of VAEs: deleting one model's build must not take the other's.
    """
    base = spec.shares_weights_with or spec.id
    folder = spec.variants[variant].get("folder") or variant
    return [
        f"{other.id}:{name}"
        for other in CATALOG.values()
        if (other.shares_weights_with or other.id) == base
        for name, info in other.variants.items()
        if (other.id, name) != (spec.id, variant) and (info.get("folder") or name) == folder and variant_installed(cfg, other, name)
    ]


def remove_variant(cfg: Config, model_id: str, variant: str, include_shared: bool = False, dry_run: bool = False) -> tuple[bool, str]:
    """Delete the files of one variant. Only ever inside ``<data>/models/<model>/``. Files other variants still need stay unless
    ``include_shared`` and no other variant of the folder is installed."""
    spec = CATALOG.get(model_id)
    if spec is None or variant not in spec.variants:
        return False, f"unknown variant {variant!r} of {model_id!r}"
    root = (cfg.data_dir / "models" / (spec.shares_weights_with or model_id)).resolve()
    folder = weights_dir(cfg, ModelConfig(id=model_id, worker=spec.family, variant=variant))
    if folder is None or not folder.resolve().is_relative_to(root):
        return False, f"{folder} is outside {root}; AHA open only deletes inside its own models folder"
    own = [folder / f for f in spec.variants[variant].get("files", ())]
    own += [p.with_name(p.name.removesuffix(".safetensors") + ".quant.json") for p in own]
    siblings = _others_in_folder(cfg, spec, variant)
    whole = not own or (include_shared and not siblings)  # a variant with a folder of its own (int8) is deleted as that folder
    if not own and not folder.exists():
        return False, f"{variant} is not installed"
    if own and not any(p.exists() for p in own) and not (include_shared and not siblings and folder.exists()):
        return False, f"{variant} is not installed"
    size = _dir_gib(folder) if whole else sum(p.stat().st_size for p in own if p.exists()) / 2**30
    if dry_run:
        return True, f"would free {size:.1f} GB ({'the whole ' + str(folder) if whole else str(len(own)) + ' file(s) of ' + variant})"
    if whole:
        shutil.rmtree(folder)
    else:
        for p in own:
            p.unlink(missing_ok=True)
        try:
            record = json.loads((folder / "install.json").read_text(encoding="utf-8"))
            record["variants"] = [v for v in record.get("variants", []) if v != variant]
            (folder / "install.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass
    kept = "" if whole or not folder.exists() else (
        f"; the files shared with other variants ({_dir_gib(folder):.1f} GB) were kept" if siblings
        else f"; the shared files ({_dir_gib(folder):.1f} GB) were kept: add --include-shared to remove them too")
    return True, f"removed {variant}, freed {size:.1f} GB{kept}"


def _download(repo: str, revision: str, target: Path) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _say("error: downloading needs huggingface-hub. Install it with: pip install \"aha-open[download]\"")
        return 2
    _say(f"Downloading {repo} @ {revision[:8]} to {target} (this is large; it resumes if interrupted)...")
    target.mkdir(parents=True, exist_ok=True)
    extra = {"tqdm_class": _progress_lines_class()} if os.environ.get(PROGRESS_ENV) == "lines" else {}
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), **extra)
    return 0


def _download_parts(parts: list[dict], target: Path) -> int:
    """Several repositories into one folder, each part with exactly its own files (``allow_patterns``), progress summed over all parts."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _say("error: downloading needs huggingface-hub. Install it with: pip install \"aha-open[download]\"")
        return 2
    missing = [_missing_bytes(part, target) for part in parts]  # what the bar counts: files already here are not fetched again
    total = sum(missing)
    done = 0
    for part, part_missing in zip(parts, missing):
        dest = target / part["subdir"] if part["subdir"] else target
        dest.mkdir(parents=True, exist_ok=True)
        _say(f"Downloading {part['repo']} @ {part['revision'][:8]} to {dest} (it resumes if interrupted)...")
        extra = {"tqdm_class": _progress_lines_class(offset=done, grand_total=total)} if os.environ.get(PROGRESS_ENV) == "lines" else {}
        snapshot_download(
            repo_id=part["repo"], revision=part["revision"], local_dir=str(dest),
            allow_patterns=[*part["patterns"], *part["files"]], **extra,
        )
        done += part_missing
    return 0


def _verify_parts(parts: list[dict], target: Path) -> int:
    """Size, then sha256, of every pinned file. A file that does not match is removed so the next run downloads it again."""
    import hashlib

    for part in parts:
        dest = target / part["subdir"] if part["subdir"] else target
        for name, (size, digest) in part["files"].items():
            path = dest / name
            if not path.is_file():
                _say(f"error: {name} is missing from {dest}; run the install again to download it")
                return 2
            actual = path.stat().st_size
            if actual != size:
                path.unlink()
                _say(f"error: {name} has {actual} bytes, expected {size}; it was removed, run the install again to download it")
                return 2
            _say(f"Checking {name} ({size / 2**30:.1f} GiB)...")
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(16 * 2**20), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                path.unlink()
                _say(f"error: {name} is not the pinned file (sha256 {h.hexdigest()[:16]}..., expected {digest[:16]}...); "
                     "it was removed, run the install again to download it")
                return 2
    return 0


PROGRESS_ENV = "AHA_OPEN_PROGRESS"
PROGRESS_PREFIX = "AHAOPEN_PROGRESS"


def _progress_lines_class(offset: int = 0, grand_total: int | None = None):
    """A tqdm class that prints `AHAOPEN_PROGRESS <bytes done> <bytes total>` about once a second instead of drawing bars.

    The web page asks for it (model_install.py sets PROGRESS_ENV). It cannot measure the weights folder instead: `hf-xet` downloads into
    its own cache and assembles each file only at the end, so a large file shows 0 bytes in the folder for minutes (observed on a real
    18 GB download). The numbers come from `snapshot_download`'s own aggregate "Downloading bytes" bar. Imported lazily: the gateway
    does not need tqdm.
    """
    import time

    from tqdm.auto import tqdm

    class ProgressLines(tqdm):
        _last = 0.0

        def display(self, msg=None, pos=None):  # never draw; the counters (n, total) still advance
            return None

        def update(self, n=1):
            shown = super().update(n)
            if self.unit == "B" and str(self.desc or "").startswith("Downloading bytes") and time.monotonic() - ProgressLines._last >= 1.0:
                ProgressLines._last = time.monotonic()
                # a multi-part install reports the whole: bytes of the parts already done + this part, over the sum of all parts
                total = grand_total if grand_total is not None else int(self.total or 0)
                print(f"{PROGRESS_PREFIX} {offset + int(self.n)} {total}", flush=True)
            return shown

    return ProgressLines


def _make_worker_env(cfg: Config, family: str) -> int:
    """Create the worker's own virtual environment from its pinned pyproject."""
    src = WORKERS_DIR / family
    if not (src / "pyproject.toml").exists():
        _say(f"error: {src} has no pyproject.toml; this install does not include the worker; reinstall aha-open, or pass --skip-env")
        return 2
    venv = cfg.data_dir / "workers" / family / "venv"
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    uv = uv_executable()
    if uv is None:
        _say("error: `uv` was not found (it builds the worker environment). Install it from https://docs.astral.sh/uv/")
        return 2
    _say(f"Creating the {family} worker environment in {venv} (pinned, downloads PyTorch; several minutes)...")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}  # install into our venv, never into the package folder
    for cmd in (
        [uv, "venv", "--python", "3.13", str(venv)],
        # --frozen: exactly the versions in the worker's uv.lock; a lock that no longer matches is an error, not a re-resolve
        [uv, "sync", "--project", str(src), "--frozen", "--no-install-project", "--python", str(venv / scripts)],
    ):
        done = subprocess.run(cmd, check=False, env=env)
        if done.returncode:
            _say(f"error: `uv {' '.join(cmd[1:3])}` failed (exit {done.returncode}); the lines above say why. https://docs.astral.sh/uv/")
            return 2
    return 0
