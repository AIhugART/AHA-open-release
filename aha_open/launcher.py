"""`aha-open start` / `aha-open stop`: what the Start-menu shortcuts of the Windows installer run (docs/product/06_deployment.md).

`start` brings the gateway up in the background (no console window) if it is not running, waits until `/v1/health`
answers, and leaves the opening of the page to the caller. It never starts a second gateway: it asks the port first.
`stop` ends the gateway AND what it started (an install in progress, a worker), using the pid the gateway wrote itself
(`run/gateway.json`, runtime.py): on Windows the venv's `python.exe` is a launcher, so the pid of the process we started is
not the gateway's.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config
from .runtime import record_path

START_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 15.0
LOG_TAIL_LINES = 8


def health_ok(cfg: Config, timeout: float = 2.0) -> bool:
    host = cfg.listen_host or "127.0.0.1"
    try:
        with urllib.request.urlopen(f"http://{host}:{cfg.listen_port}/v1/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":  # os.kill(pid, 0) would KILL the process on Windows
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def recorded_pid(cfg: Config) -> int | None:
    try:
        pid = json.loads(record_path(cfg).read_text(encoding="utf-8")).get("pid")
    except (OSError, ValueError):
        return None
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid_alive(pid) else None


def start_log(cfg: Config) -> Path:
    return cfg.data_dir / "logs" / "start.log"


def _tail(path: Path) -> str:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    return "\n".join(lines[-LOG_TAIL_LINES:])


def _port_in_use(cfg: Config) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((cfg.listen_host or "127.0.0.1", cfg.listen_port)) == 0


def start(cfg: Config) -> tuple[bool, str]:
    """(ok, one sentence for the person). Idempotent: a running gateway is reported, not started again."""
    if cfg.profile != "local":
        return False, f"`start` is for the local profile (this one is {cfg.profile}); run `aha-open serve` in a terminal instead."
    if health_ok(cfg):
        return True, "AHA open is already running."
    already_starting = recorded_pid(cfg) is not None
    if not already_starting and _port_in_use(cfg):
        return False, (f"Port {cfg.listen_port} is used by another program, so AHA open cannot start. "
                       f"Close that program, or change `listen` in {cfg.data_dir / 'config.toml'}.")
    proc = None
    log = start_log(cfg)
    if not already_starting:
        log.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        with log.open("ab") as out:
            out.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
            # cwd: `python -m` puts the CURRENT folder first on sys.path, so from a source checkout it would run that code instead of
            # the installed package (a real smoke run did exactly that); the data folder never holds an `aha_open`.
            proc = subprocess.Popen(
                [sys.executable, "-m", "aha_open.cli", "--data-dir", str(cfg.data_dir), "serve"],
                stdin=subprocess.DEVNULL, stdout=out, stderr=out, cwd=str(cfg.data_dir), **kwargs,
            )
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if health_ok(cfg):
            return True, "AHA open is running."
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(0.4)
    reason = _tail(log)
    return False, "AHA open did not start." + (f"\n\n{reason}" if reason else "")


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)


def stop(cfg: Config) -> tuple[bool, str]:
    pid = recorded_pid(cfg)
    if pid is None:
        if health_ok(cfg):
            return False, ("Something answers on the AHA open port, but it was not started by this installation "
                           "(there is no record of it), so it was left alone.")
        try:
            record_path(cfg).unlink()  # a crash leaves its record behind
        except OSError:
            pass
        return True, "AHA open is not running."
    _kill_tree(pid)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline and pid_alive(pid):
        time.sleep(0.2)
    if pid_alive(pid):
        return False, f"AHA open (process {pid}) did not stop. Close it from the Task Manager."
    try:
        record_path(cfg).unlink()
    except OSError:
        pass
    return True, "AHA open has stopped."
