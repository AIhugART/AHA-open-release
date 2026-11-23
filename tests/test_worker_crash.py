"""The real Qwen worker leaves a Python stack in its log when it dies natively (needs the worker's own environment).

Root cause this guards: a native crash (access violation / SIGSEGV) skips every `except`, so the worker log held only
progress bars and the gateway only an exit status; nobody could tell which line was running. `install_crash_diagnostics`
turns that into a stack in the log. The test kills a real interpreter inside the worker's venv on purpose.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parents[1] / "workers" / "qwen_image"
PYTHON = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

pytestmark = pytest.mark.skipif(not PYTHON.exists(), reason="the qwen worker environment is not installed")

# NOT ctypes.string_at(0): ctypes catches the access violation and raises OSError, so the process would only exit with 1 and
# the test would not have the shape of the real defect. faulthandler._sigsegv() really kills the process without any Python
# exception handling (on Windows it aborts with exit status 3, not 0xC0000005; the mapping of the real status 3221225477 is
# covered by test_worker_exit.py, which takes that number from the crash seen on 2026-11-23).
CRASH = "import worker, faulthandler; worker.install_crash_diagnostics(); faulthandler._sigsegv()"


def run(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(PYTHON), "-c", snippet], cwd=WORKER_DIR, capture_output=True, text=True, timeout=120)


def test_a_native_crash_writes_the_python_stack_to_the_log():
    done = run(CRASH)
    assert done.returncode != 0
    assert "access violation" in done.stderr or "Segmentation fault" in done.stderr
    assert "most recent call first" in done.stderr and "<string>" in done.stderr  # the line that was running


def test_stage_lines_carry_time_and_memory_and_nothing_from_a_request():
    done = run("import worker; worker.stage('load transformer: reading weights')")
    assert done.returncode == 0
    line = done.stderr.strip()
    assert line.startswith("[worker +") and "rss=" in line and "ram_available=" in line and line.endswith("load transformer: reading weights")
