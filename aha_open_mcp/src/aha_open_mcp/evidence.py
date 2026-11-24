"""Every tool output starts with a header that says what the evidence is worth and whether the tool or the
server runs older code than the files on disk."""

from __future__ import annotations

import hashlib
import socket
import sys
from pathlib import Path

from aha_open import runtime

from . import proc
from .targets import Target, record_of

PACKAGE_DIR = Path(__file__).resolve().parent


def package_digest() -> str:
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts:
            h.update(path.relative_to(PACKAGE_DIR).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


LOADED_DIGEST = package_digest()  # taken when this module is first imported = when the MCP server process started


def harness_state() -> str:
    return "fresh" if package_digest() == LOADED_DIGEST else "STALE HARNESS (the files changed after this tool started: restart the agent session)"


def gateway_state(target: Target) -> str:
    record = record_of(target.data_dir)
    if record is None:
        return "NO RECORD (not started by `aha-open serve` of this version, or never started)"
    pid = int(record.get("pid", 0))
    if not proc.pid_alive(pid):
        return f"NOT RUNNING (the record names pid {pid}, which is gone)"
    if record.get("source_digest") != runtime.source_digest():
        return f"STALE GATEWAY (pid {pid} runs gateway code older than the files on disk: restart it)"
    return "fresh"


def machine() -> str:
    return f"{socket.gethostname()} ({sys.platform})"


def header(target: Target) -> str:
    return (
        f"machine: {machine()} | target: {target.name} | worker: {','.join(target.worker_kinds()) or 'none'} | "
        f"evidence: {target.evidence()}\nharness: {harness_state()} | gateway: {gateway_state(target)}"
    )
