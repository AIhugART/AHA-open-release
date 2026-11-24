"""Process helpers that are safe on Windows (where `os.kill(pid, 0)` would TERMINATE the process)."""

from __future__ import annotations

import os
import subprocess
import sys
import time


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def kill_pid(pid: int, wait_s: float = 15.0) -> bool:
    """Terminate `pid` and wait for it to go. Returns True when it is gone."""
    if pid_alive(pid):
        try:
            os.kill(pid, 15 if sys.platform == "win32" else 9)  # on Windows any signal terminates
        except OSError:
            pass
    deadline = time.monotonic() + wait_s
    while pid_alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True
