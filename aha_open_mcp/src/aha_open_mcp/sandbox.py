"""A throwaway gateway (plan D3). It is a REAL `aha-open serve` child process so the code under test is the production code
path; it must not inherit this process's stdio (stdout is the MCP protocol channel, and a library that prints to an
inherited stdin/stdout is how the Qwen worker once hung on Windows)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from aha_open.catalog import CATALOG
from aha_open.config import Config, ModelConfig, default_data_dir, render
from aha_open.util import iso

from . import proc
from .targets import REPO, SANDBOX_FILE, SANDBOX_MARKER, STATE_DIR, TargetError, record_of

FAKE_MODELS = [ModelConfig(id="example-image", worker="fake"), ModelConfig(id="example-video", worker="fake")]


SIZED_VIDEO = ModelConfig(id="example-video-sized", worker="fake")  # the shape of what MiniMax H3 declares (sizes, frame grid, steps)


H3_MODELS = ("minimax-h3-fl2va", "minimax-h3-ref2va")  # one install folder serves both (the reference model adds its own transformer)


def real_h3_models(weights_dir: str = "", model: str = "minimax-h3-fl2va", variant: str = "") -> list[ModelConfig]:
    """The real MiniMax H3 model (build ``variant``, default "mixed") from an install folder (`aha-open models install minimax-h3-fl2va ...`) and the worker
    environment of this checkout. Nothing is downloaded or installed; raises with the command that fixes what is missing."""
    # The build name first: it is wrong whatever this machine has installed, and no download or environment would fix it.
    variant = variant or "mixed"
    offered = CATALOG[model].variants
    if variant not in offered:
        raise TargetError(f"{model} offers no build {variant!r}; choose one of: {', '.join(offered)}")
    weights = Path(weights_dir or os.environ.get("AHA_OPEN_H3_WEIGHTS") or default_data_dir() / "models" / "minimax-h3-fl2va" / "mixed")
    if not (weights / "model_index.json").exists():
        raise TargetError(f"no MiniMax H3 install at {weights} (no model_index.json): run `aha-open models install minimax-h3-fl2va "
                          "--accept-license minimax-h3 --confirm-territory`, or pass weights_dir / set AHA_OPEN_H3_WEIGHTS")
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = REPO / "workers" / "minimax_h3" / ".venv" / scripts
    if not python.exists():
        raise TargetError(f"no worker environment at {python}: run `cd workers/minimax_h3 && uv sync --frozen`")
    absent =[f for f in (*offered[variant]["files"], *CATALOG[model].required_files) if not (weights / f).is_file()]
    if absent:
        raise TargetError(f"{absent[0]} is not in {weights}: run `aha-open models install {model} --variant {variant} "
                          "--accept-license minimax-h3 --confirm-territory`")
    return [ModelConfig(id=model, worker="minimax_h3", variant=variant, weights_dir=str(weights), python=str(python))]


def real_models(weights_dir: str = "", model: str = "qwen-image-2.1", variant: str = "") -> list[ModelConfig]:
    """The real model ``model`` (Qwen INT8 by default), run from the weights the operator already installed into the data folder
    (`aha-open models install`, or the web page's Download button) and the worker environment that exists in this checkout (nothing
    is downloaded or installed). Raises with the exact command to fix what is missing."""
    if model in H3_MODELS:
        return real_h3_models(weights_dir, model, variant)
    if model != "qwen-image-2.1":
        raise TargetError(f"no real-model setup for {model!r}; known: qwen-image-2.1, {', '.join(H3_MODELS)}")
    weights = Path(weights_dir or os.environ.get("AHA_OPEN_QWEN_WEIGHTS") or default_data_dir() / "models" / "qwen-image-2.1" / "int8")
    if not (weights / "model_index.json").exists():
        raise TargetError(f"no Qwen INT8 weights at {weights} (no model_index.json): install the model first (`aha-open models install "
                          "qwen-image-2.1 --accept-license qwen-research`), or pass weights_dir / set AHA_OPEN_QWEN_WEIGHTS")
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = REPO / "workers" / "qwen_image" / ".venv" / scripts
    if not python.exists():
        raise TargetError(f"no worker environment at {python}: run `cd workers/qwen_image && uv sync --frozen`")
    return [ModelConfig(id="qwen-image-2.1", worker="qwen_image", variant="int8", weights_dir=str(weights), python=str(python))]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _info() -> dict | None:
    try:
        return json.loads((STATE_DIR / SANDBOX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def start(fake_step_s: float = 0.02, fake_load_s: float = 0.2, start_timeout_s: float = 60.0, crash_on_load: bool = False,
          real_model: bool = False, weights_dir: str = "", model: str = "qwen-image-2.1", sized_video: bool = False,
          variant: str = "", memory_limit_gib: float = 0.0) -> dict:
    if (current := _info()) and proc.pid_alive(int((record_of(Path(current["data_dir"])) or {}).get("pid", 0))):
        raise TargetError(f"a sandbox is already running at {current['base_url']} - call run_sandbox_stop first")
    if current:  # left by a server that was killed before it could clean up: its gateway is gone, its folder is not
        stop()
    models = real_models(weights_dir, model, variant) if real_model else [*FAKE_MODELS, *([SIZED_VIDEO] if sized_video else [])]  # before anything is created: a missing file leaves nothing behind
    data = Path(tempfile.mkdtemp(prefix="aha-open-sandbox-"))
    (data / SANDBOX_MARKER).write_text("created by aha-open-mcp; safe to delete\n", encoding="utf-8")
    port = _free_port()
    cfg = Config(data_dir=data, listen=f"127.0.0.1:{port}", models=models, gpu_memory_limit_gib=memory_limit_gib)
    (data / "config.toml").write_text(render(cfg), encoding="utf-8")
    (data / "logs").mkdir(exist_ok=True)
    out = (data / "logs" / "serve-stdio.txt").open("ab")
    env = dict(os.environ, AHA_FAKE_STEP_S=str(fake_step_s), AHA_FAKE_LOAD_S=str(fake_load_s), PYTHONUTF8="1")
    if crash_on_load:  # fault injection the fake worker already supports: it dies natively while loading the model
        env["AHA_FAKE_CRASH_ON_LOAD"] = "native"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    launcher = subprocess.Popen(
        [sys.executable, "-c", "from aha_open.cli import main; main()", "--data-dir", str(data), "serve"],
        stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=env, creationflags=flags,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + start_timeout_s
    while True:
        try:
            if httpx.get(base + "/v1/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if launcher.poll() is not None or time.monotonic() > deadline:
            proc.kill_pid(launcher.pid)
            raise TargetError(f"the sandbox gateway did not come up; see {data / 'logs' / 'serve-stdio.txt'}")
        time.sleep(0.1)
    # On Windows `launcher` is the venv's python.exe shim, not the gateway: the gateway's own record names the real pid.
    record = record_of(data) or {}
    info = {"data_dir": str(data), "base_url": base, "pid": record.get("pid"), "launcher_pid": launcher.pid, "started_at": iso(time.time())}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / SANDBOX_FILE).write_text(json.dumps(info), encoding="utf-8")
    return info


def stop(keep_data: bool = False) -> str:
    info = _info()
    if not info:
        return "no sandbox was started by this tool"
    data = Path(info["data_dir"])
    pids = {int(p) for p in (info.get("pid"), (record_of(data) or {}).get("pid"), info.get("launcher_pid")) if p}
    gone = all([proc.kill_pid(p) for p in pids])
    removed = False
    if gone and not keep_data and (data / SANDBOX_MARKER).exists() and data.name.startswith("aha-open-sandbox-"):
        # Only a directory this tool created, marked and named itself. A real worker is a CHILD of the gateway: it notices the closed pipe
        # and exits a moment after the gateway is killed, and until then it still holds its log open (observed 2026-11-24), so retry.
        for _ in range(40):
            shutil.rmtree(data, ignore_errors=True)
            if not data.exists():
                break
            time.sleep(0.5)
        removed = not data.exists()
    if gone:
        (STATE_DIR / SANDBOX_FILE).unlink(missing_ok=True)
    return f"stopped pids {sorted(pids)}: {'all gone' if gone else 'SOME STILL ALIVE'}; data dir {'removed' if removed else 'kept at ' + str(data)}"
