"""A worker child process spoken to over stdin/stdout, one JSON object per line (docs/product/04_architecture.md section 2).

No network port is opened for workers. Pixels never travel over the pipe: the gateway puts inputs in the job directory
and the worker writes outputs there. The prompt is never sent over the pipe or logged either (it is in request.json).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path


# Windows reports a crashed process as an NTSTATUS (>= 0xC0000000); POSIX reports the killing signal as a negative number.
_NTSTATUS = {
    0xC0000005: ("crashed in native code (access violation)", False),
    0xC00000FD: ("crashed in native code (stack overflow)", False),
    0xC0000374: ("crashed in native code (heap corruption)", False),
    0xC0000409: ("crashed in native code (fail-fast abort)", False),
    0xC0000142: ("could not start because a DLL failed to initialise", False),
    0xC0000135: ("could not start because a DLL is missing", False),
    0xC000012D: ("ran out of memory (Windows commit limit reached)", True),
    0xC0000017: ("ran out of memory (not enough virtual memory)", True),
}
_SIGNALS = {
    11: ("crashed in native code (segmentation fault)", False),
    7: ("crashed in native code (bus error)", False),
    6: ("aborted", False),
    9: ("was killed (on Linux this is often the out-of-memory killer)", True),
    15: ("was terminated", False),
}


def describe_exit(code: int | None, log_name: str) -> str:
    """Turn a worker's exit status into a sentence the person running the server can act on.

    Different causes give different advice: running out of memory is fixed by freeing memory; a crash is not the request's
    fault and the log (stage lines + the Python stack faulthandler wrote) shows where it stopped. Only the log's file name
    is given, never a server path, because this text travels to the API client.
    """
    if code is None:
        return f"the worker exited without a status; see {log_name} in the server's logs folder"
    status = code & 0xFFFFFFFF
    memory = False
    if -256 < code < 0:  # POSIX: the signal that killed the process (checked first: -9 & 0xFFFFFFFF would look like an NTSTATUS)
        what, memory = _SIGNALS.get(-code, (f"was stopped by signal {-code}", False))
        shown = f"signal {-code}"
    elif status >= 0xC0000000:  # Windows NTSTATUS, unsigned from Popen or the same bits as a signed integer
        what, memory = _NTSTATUS.get(status, (f"crashed (Windows status 0x{status:08X})", False))
        shown = f"exit code {status} = 0x{status:08X}"
    else:
        what, shown = "exited", f"exit code {code}"
    advice = (
        "Free memory (close other programs) and try again."
        if memory
        else "The request is not the cause; try again, and if it repeats report it with that log."
    )
    return f"the worker {what} ({shown}). {advice} {log_name} in the server's logs folder shows where it stopped."


class WorkerProc:
    def __init__(self, cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self.log_name = log_path.name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("ab")
        self.events: queue.Queue[dict] = queue.Queue()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, cwd=str(cwd) if cwd else None,
            env=env, text=True, encoding="utf-8", bufsize=1,
        )
        self._write_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, name="worker-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a library printed to stdout; it is not protocol
            if isinstance(event, dict):
                self.events.put(event)
        self.events.put({"event": "exited", "code": self.proc.wait()})

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, message: dict) -> bool:
        with self._write_lock:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps(message) + "\n")
                self.proc.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

    def next_event(self, timeout: float) -> dict | None:
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self, grace_s: float = 5.0) -> None:
        """Ask politely, then terminate, then kill. Never raises."""
        if self.alive():
            self.send({"op": "unload"})
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                self.kill()
        self._close()

    def kill(self) -> None:
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._close()

    def _close(self) -> None:
        for stream in (self.proc.stdin, self._log):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
