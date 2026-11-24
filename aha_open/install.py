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
    weights_dir,
    worker_python,
)
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
        # `model_index.json` is among the first files to arrive, so it proves nothing: after an interrupted download the big files are
        # missing. Unfinished = a file still marked incomplete, or hf began downloading here and no install.json says it ended.
        started_by_hf = (target / HF_DOWNLOAD_STATE).is_dir()
        unfinished = download_incomplete(target) or (started_by_hf and not (target / "install.json").exists())
        if unfinished or not (target / "model_index.json").exists():
            (target / "install.json").unlink(missing_ok=True)  # a download starts: an older record must not vouch for it
            code = _download(info["repo"], info["revision"], target)  # resumes: what is already complete is not fetched again
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
    extra = {"tqdm_class": _progress_lines_class()} if os.environ.get(PROGRESS_ENV) == "lines" else {}
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), **extra)
    return 0


PROGRESS_ENV = "AHA_OPEN_PROGRESS"
PROGRESS_PREFIX = "AHAOPEN_PROGRESS"


def _progress_lines_class():
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
                print(f"{PROGRESS_PREFIX} {int(self.n)} {int(self.total or 0)}", flush=True)
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
