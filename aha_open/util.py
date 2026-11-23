"""Small helpers shared by the gateway. Nothing here imports torch."""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def now() -> float:
    return time.time()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG IHDR, or None when the file is not a PNG. Lets the gateway check output size without PIL."""
    with path.open("rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
