"""Regression: a worker hung forever on its first model load under the gateway.

Root cause: on Windows a thread parked in a synchronous ReadFile on the stdin pipe blocks other operations on that
handle, including those performed by `import numpy` / `import torch`. The gateway keeps stdin open for the whole life
of the worker, so the first heavy import never returned. The worker therefore reads stdin by polling
(`worker.stdin_lines`), never by blocking.

The tests run in the worker's own virtual environment (the one that has numpy) and are skipped when it is absent.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parents[1] / "workers" / "qwen_image"
VENV_PY = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

pytestmark = pytest.mark.skipif(not VENV_PY.exists(), reason="the qwen_image worker environment is not installed")

# What the worker does at startup: start the stdin reader thread, then import a heavy native library on the main thread.
SNIPPET = """
import sys, threading
sys.path.insert(0, {worker_dir!r})
import worker
{reader}
import numpy
print("IMPORTED", flush=True)
"""


def _imports_within(reader: str, timeout: float) -> bool:
    """True when the worker-like process finishes its heavy import while its stdin stays OPEN and silent (as under the
    gateway). `communicate()` would close stdin and hide the bug, so stdout is read from a thread instead."""
    code = SNIPPET.format(worker_dir=str(WORKER_DIR), reader=reader)
    proc = subprocess.Popen([str(VENV_PY), "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    seen = threading.Event()

    def watch() -> None:
        for line in proc.stdout:
            if "IMPORTED" in line:
                seen.set()

    threading.Thread(target=watch, daemon=True).start()
    try:
        return seen.wait(timeout)
    finally:
        proc.kill()
        proc.wait()


def test_heavy_imports_complete_while_the_stdin_reader_is_waiting():
    # stdin stays open and silent, exactly like a gateway that has not sent a message yet.
    assert _imports_within("threading.Thread(target=worker.reader, daemon=True).start()", timeout=60)


def test_the_reader_still_delivers_messages_and_signals_eof():
    code = SNIPPET.format(worker_dir=str(WORKER_DIR), reader="threading.Thread(target=worker.reader, daemon=True).start()")
    code += "\nprint(worker.inbox.get(timeout=10), flush=True)\nprint(worker.inbox.get(timeout=10), flush=True)\nprint(sorted(worker.cancelled), flush=True)\n"
    done = subprocess.run(
        [str(VENV_PY), "-c", code], input='{"op": "load", "variant": "int8"}\n{"op": "cancel", "job_id": "job_x"}\n',
        capture_output=True, text=True, timeout=60, check=False,
    )
    lines = done.stdout.strip().splitlines()
    assert lines[1] == "{'op': 'load', 'variant': 'int8'}", done.stderr[-500:]
    assert lines[2] == "{'op': 'unload'}"  # stdin closed: the worker is told to stop
    assert lines[3] == "['job_x']"  # cancel is handled by the reader itself, not queued behind a long job


@pytest.mark.skipif(sys.platform != "win32", reason="the blocking-read hang is a Windows pipe behaviour")
def test_control_a_blocking_reader_really_does_hang_on_this_platform():
    """Proves the regression test has teeth: the old reader (`for line in sys.stdin`) never gets past the import."""
    blocking = "threading.Thread(target=lambda: [None for _ in sys.stdin], daemon=True).start()"
    assert not _imports_within(blocking, timeout=20)
