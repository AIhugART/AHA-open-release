"""`aha-open models install`: licence acknowledgement, variant choice, weights, worker environment, config.

The operator downloads the weights and accepts the licence; AHA open never ships or mirrors them (plan R4).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .catalog import CATALOG, WORKERS_DIR, detect_gpus, weights_dir, worker_python
from .config import Config, ModelConfig, render
from .hardware import detect_ram_gib, free_disk_gib, recommend_variant

# What may be installed. Revisions are pinned: a different upstream revision is a different (untested) model.
INSTALLABLE = {
    "qwen-image-2.1": {
        "license_key": "qwen-research",
        "variants": {
            "int8": {"repo": "ixim/Image21-INT8", "revision": "099a9acacd0f1ae640ad4c740aed3af9fd12be19", "disk_gib": 18},
        },
    },
}


def _say(text: str) -> None:
    print(text)


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

    gpus = detect_gpus(ttl_s=0)
    ram = detect_ram_gib()
    variant, reason = recommend_variant(gpus[0]["memory_total_mib"] if gpus else None, ram)
    if variant is None and not args.from_dir:
        _say(f"error: {reason}")  # --from-dir says the weights are already on disk; `aha-open doctor` checks the GPU afterwards
        return 2
    variant = variant or next(iter(plan["variants"]))
    _say(f"Installing {variant}: {reason}")

    info = plan["variants"][variant]
    target = None
    if args.from_dir:
        target = Path(args.from_dir).resolve()
        if not (target / "model_index.json").exists():
            _say(f"error: {target} has no model_index.json; it is not a {spec.display_name} download")
            return 2
    else:
        target = weights_dir(cfg, ModelConfig(id=args.model, worker=spec.family, variant=variant))
        free = free_disk_gib(target)
        if free < info["disk_gib"] * 1.1:
            _say(f"error: {info['disk_gib']} GB are needed for {variant} but only {free:.0f} GB are free where {target} lives")
            return 2
        if not (target / "model_index.json").exists():
            code = _download(info["repo"], info["revision"], target)
            if code:
                return code
    (target / "install.json").write_text(
        json.dumps({
            "model": args.model, "variant": variant, "repo": info["repo"], "revision": info["revision"],
            "license_key": key, "installed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, indent=2),
        encoding="utf-8",
    )

    mc = cfg.model(args.model)
    if mc is None:
        mc = ModelConfig(id=args.model, worker=spec.family)
        cfg.models.append(mc)
    mc.variant = variant
    mc.enabled = True
    mc.weights_dir = str(target) if args.from_dir else ""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "config.toml").write_text(render(cfg), encoding="utf-8")

    if not args.skip_env and worker_python(cfg, mc) is None:
        code = _make_worker_env(cfg, spec.family)
        if code:
            return code
    _say(f"Installed {spec.display_name} ({variant}). Check with `aha-open doctor`, then start with `aha-open serve`.")
    return 0


def _download(repo: str, revision: str, target: Path) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _say("error: downloading needs huggingface-hub. Install it with: pip install \"aha-open[download]\"")
        return 2
    _say(f"Downloading {repo} @ {revision[:8]} to {target} (this is large; it resumes if interrupted)...")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))
    return 0


def _make_worker_env(cfg: Config, family: str) -> int:
    """Create the worker's own virtual environment from its pinned pyproject (plan R2)."""
    src = WORKERS_DIR / family
    if not (src / "pyproject.toml").exists():
        _say(f"error: {src} has no pyproject.toml; this install does not include the worker; reinstall aha-open, or pass --skip-env")
        return 2
    venv = cfg.data_dir / "workers" / family / "venv"
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    _say(f"Creating the {family} worker environment in {venv} (pinned, downloads PyTorch; several minutes)...")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}  # install into our venv, never into the package folder
    for cmd in (
        ["uv", "venv", "--python", "3.13", str(venv)],
        # --frozen: exactly the versions in the worker's uv.lock (plan R2); a lock that no longer matches is an error, not a re-resolve
        ["uv", "sync", "--project", str(src), "--frozen", "--no-install-project", "--python", str(venv / scripts)],
    ):
        done = subprocess.run(cmd, check=False, env=env)
        if done.returncode:
            _say(f"error: `{' '.join(cmd[:3])}` failed (exit {done.returncode}). Is `uv` installed? https://docs.astral.sh/uv/")
            return 2
    return 0
