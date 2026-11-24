"""Install a model from the web page, for the person at this machine (`/admin/models/*`, docs/04_architecture.md).

The work is done by a CHILD PROCESS running the same `aha-open models install` the command line uses, so there is exactly
one way to install a model. That is also what makes cancelling real: the whole process tree is killed (a download resumes
the next time). The gateway never imports `huggingface_hub`, and no request waits for the download: the page polls.

When the child exits cleanly the gateway re-reads `config.toml` and replaces `cfg.models` in place; the model's state is
computed from disk on every request (catalog.model_state), so the manifest changes with no restart.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .catalog import CATALOG, detect_gpus, model_state, weights_dir
from .config import Config, ModelConfig, load
from .errors import ApiError, invalid, not_found
from .hardware import detect_ram_gib, free_disk_gib, recommend_variant
from .install import INSTALLABLE, PROGRESS_ENV, PROGRESS_PREFIX, uv_executable
from .util import iso, now

GIB = 2**30
TAIL_LINES = 40
SIZE_POLL_S = 1.0
MESSAGE_LIMIT = 300

# What the child prints (aha_open/install.py) -> the phase the page shows.
PHASES = (
    ("Installing ", "preparing"),
    ("Downloading ", "downloading"),
    ("Creating the ", "creating_environment"),
    ("Installed ", "finishing"),
)
# Output of a download or `uv` that could not reach the network. Matched on the tail of the child's output.
NETWORK_MARKERS = (
    "connectionerror", "max retries exceeded", "getaddrinfo", "name resolution", "timed out", "connection reset",
    "sslerror", "proxyerror", "network is unreachable", "error sending request", "dns error", "failed to fetch",
    "failed to download", "could not resolve", "connection refused", "connecterror", "readtimeout",
)

Command = Callable[[str, str], list[str]]


def default_command(cfg: Config) -> Command:
    """`python -m aha_open.cli --data-dir D models install MODEL --accept-license KEY` with the interpreter running the gateway."""

    def build(model_id: str, license_key: str) -> list[str]:
        return [
            sys.executable, "-m", "aha_open.cli", "--data-dir", str(cfg.data_dir),
            "models", "install", model_id, "--accept-license", license_key,
        ]

    return build


def has_download_support() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop the child AND what it started (`uv`, the download): a bare terminate() would leave them running."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _dir_bytes(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    else:
                        try:
                            total += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
        except OSError:
            continue
    return total


def classify_failure(returncode: int, tail: list[str], data_dir: Path) -> dict:
    """One sentence the person can act on, and a code for the cause (docs/04_architecture.md)."""
    text = "\n".join(tail).lower()
    last_error = next((line for line in reversed(tail) if line.startswith("error:")), "")
    if any(marker in text for marker in NETWORK_MARKERS):
        return {
            "code": "network_unreachable",
            "message": "The download could not reach the internet. Check the connection (and any proxy or firewall), then press "
                       "Download again; what was already downloaded is kept.",
        }
    line = last_error or next((ln for ln in reversed(tail) if ln.strip()), f"the installer stopped (exit {returncode})")
    line = line.replace(str(data_dir), "<data folder>")[:MESSAGE_LIMIT]
    return {"code": "install_failed", "message": f"The installation stopped: {line}"}


class _Run:
    def __init__(self, model_id: str, target: Path, expected_bytes: int) -> None:
        self.model_id = model_id
        self.state = "running"
        self.phase = "preparing"
        self.target = target
        self.expected_bytes = expected_bytes
        self.error: dict | None = None
        self.started_at = now()
        self.finished_at: float | None = None
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.size = (0.0, 0)  # (when it was measured, bytes of the weights folder): only a fallback, see `reported`
        self.reported: tuple[int, int] | None = None  # (bytes done, bytes total) as the downloader itself says


class InstallManager:
    def __init__(self, cfg: Config, command: Command | None = None) -> None:
        self.cfg = cfg
        self._command = command or default_command(cfg)
        self._lock = threading.Lock()
        self._run: _Run | None = None  # the running install, or the last one that ended

    # ------------------------------------------------------------------ what the page shows
    def models(self) -> list[dict]:
        out = []
        for model_id, plan in INSTALLABLE.items():
            spec = CATALOG[model_id]
            variant, _ = self._variant()
            mc = self.cfg.model(model_id) or ModelConfig(id=model_id, worker=spec.family, variant=variant or "")
            state, detail = model_state(self.cfg, mc, spec)
            if detail:  # "weights not found at <path>": the page shows it, and it must not spell out the data folder
                detail = detail.replace(str(self.cfg.data_dir), "<data folder>")
            blocker = None if state == "ready" else self._blocker(model_id)
            lic = spec.license
            out.append({
                "id": model_id,
                "display_name": spec.display_name,
                "state": state,
                "state_detail": detail,
                "license": {
                    "key": plan["license_key"], "name": lic.get("name"), "commercial_use": lic.get("commercial_use"),
                    "notes": lic.get("notes", ""), "url": lic.get("url", ""),
                },
                "disk_gib": next(iter(plan["variants"].values()))["disk_gib"],
                "blocker": blocker,
                "install": self.status(model_id),
            })
        return out

    def status(self, model_id: str) -> dict:
        with self._lock:
            run = self._run if self._run and self._run.model_id == model_id else None
            if run is None:
                return {"state": "idle", "phase": None, "downloaded_bytes": 0, "expected_bytes": 0, "error": None,
                        "started_at": None, "finished_at": None}
            done, expected = run.size[1], run.expected_bytes
            if run.reported is not None:  # the downloader's own count: the folder under-counts (hf-xet assembles files at the end)
                done = run.reported[0]
                if run.reported[1] >= expected * 0.9:  # the total grows as files are discovered; trust it once it is nearly complete
                    expected = run.reported[1]
            elif run.state == "running" and run.phase == "downloading" and time.monotonic() - run.size[0] >= SIZE_POLL_S:
                run.size = (time.monotonic(), _dir_bytes(run.target))
                done = run.size[1]
            return {
                "state": run.state, "phase": run.phase if run.state == "running" else None,
                "downloaded_bytes": done, "expected_bytes": expected, "error": run.error,
                "started_at": iso(run.started_at), "finished_at": iso(run.finished_at) if run.finished_at else None,
            }

    # ------------------------------------------------------------------ start / cancel
    def start(self, model_id: str, accept_license: object) -> dict:
        plan = INSTALLABLE.get(model_id)
        if plan is None or CATALOG.get(model_id) is None:
            raise not_found("this model cannot be installed from here")
        if accept_license != plan["license_key"]:
            raise invalid("the licence has not been accepted: send accept_license with the licence key shown on the page")
        with self._lock:
            if self._run and self._run.state == "running":
                raise ApiError(409, "install_running", "a model is being installed already; wait for it to finish or cancel it")
            spec = CATALOG[model_id]
            variant, _ = self._variant()
            mc = self.cfg.model(model_id) or ModelConfig(id=model_id, worker=spec.family, variant=variant or "")
            if model_state(self.cfg, mc, spec)[0] == "ready":
                raise ApiError(409, "already_installed", f"{spec.display_name} is already installed")
            blocker = self._blocker(model_id)
            if blocker:
                raise ApiError(409, blocker["code"], blocker["message"])
            variant = variant or next(iter(plan["variants"]))
            target = weights_dir(self.cfg, ModelConfig(id=model_id, worker=spec.family, variant=variant))
            run = _Run(model_id, target, int(plan["variants"][variant]["disk_gib"] * GIB))
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "AHA_OPEN_UV": uv_executable() or "",
                   PROGRESS_ENV: "lines"}
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            run.proc = subprocess.Popen(  # cwd: see launcher.start - `python -m` must not pick up an `aha_open` from the current folder
                self._command(model_id, plan["license_key"]), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(self.cfg.data_dir), **kwargs,
            )
            self._run = run
        threading.Thread(target=self._watch, args=(run,), name=f"install-{model_id}", daemon=True).start()
        return self.status(model_id)

    def cancel(self, model_id: str) -> bool:
        with self._lock:
            run = self._run
            if run is None or run.model_id != model_id or run.state != "running" or run.proc is None:
                return False
            run.cancelled = True
            proc = run.proc
        _kill_tree(proc)
        return True

    def shutdown(self) -> None:
        """The gateway is stopping: do not leave a download running with nobody to report to."""
        with self._lock:
            run = self._run
            proc = run.proc if run and run.state == "running" else None
            if run and proc:
                run.cancelled = True
        if proc:
            _kill_tree(proc)

    # ------------------------------------------------------------------ internals
    def _variant(self) -> tuple[str | None, str]:
        gpus = detect_gpus()
        return recommend_variant(gpus[0]["memory_total_mib"] if gpus else None, detect_ram_gib())

    def _blocker(self, model_id: str) -> dict | None:
        """Why an install would fail, checked BEFORE starting, with the same functions the command line uses."""
        plan, spec = INSTALLABLE[model_id], CATALOG[model_id]
        variant, reason = self._variant()
        if variant is None:
            return {"code": "unsupported_hardware", "message": reason[0].upper() + reason[1:] + "."}
        if uv_executable() is None:
            return {"code": "uv_missing", "message": "The tool that builds the model's environment (uv) was not found. "
                                                     "Reinstall AHA open, or install uv from https://docs.astral.sh/uv/."}
        if not has_download_support():
            return {"code": "download_support_missing", "message": "This installation cannot download models "
                    "(huggingface-hub is missing). Reinstall AHA open, or run: pip install \"aha-open[download]\"."}
        info = plan["variants"][variant]
        target = weights_dir(self.cfg, ModelConfig(id=model_id, worker=spec.family, variant=variant))
        free = free_disk_gib(target)
        if free < info["disk_gib"] * 1.1 and not (target / "model_index.json").exists():
            return {"code": "not_enough_disk", "message": f"{spec.display_name} needs about {info['disk_gib']} GB of free disk "
                    f"(plus room for its environment) but only {free:.0f} GB are free on that drive. Free some space, then try again."}
        return None

    def _watch(self, run: _Run) -> None:
        proc = run.proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith(PROGRESS_PREFIX + " "):  # machine-readable progress: not an output line, never part of an error message
                try:
                    _, done, total = line.split()
                    with self._lock:
                        run.reported = (int(done), int(total))
                except ValueError:
                    pass
                continue
            with self._lock:
                run.tail.append(line)
                for prefix, phase in PHASES:
                    if line.startswith(prefix):
                        run.phase = phase
        code = proc.wait()
        error = None
        state = "succeeded"
        if run.cancelled:
            state = "cancelled"
        elif code != 0:
            state, error = "failed", classify_failure(code, list(run.tail), self.cfg.data_dir)
        else:
            try:  # the installer wrote config.toml in another process; adopt it in place so the scheduler sees the model
                self.cfg.models[:] = load(self.cfg.data_dir).models
            except Exception as err:  # noqa: BLE001 - say so rather than report success with a stale configuration
                state = "failed"
                error = {"code": "install_failed", "message": f"The model was installed but its settings could not be read: {err}"}
        with self._lock:
            run.state, run.error, run.finished_at = state, error, now()
            run.size = (time.monotonic(), _dir_bytes(run.target))
