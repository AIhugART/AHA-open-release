"""A worker child process spoken to over stdin/stdout, one JSON object per line (docs/04_architecture.md section 2).

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


class WorkerProc:
    def __init__(self, cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
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
