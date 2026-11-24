"""Read logs passively: every read says which file, when it last changed and how old that is (a log older than the failure
being investigated is a hypothesis, not evidence). Rotated backups are followed so a read right after a rotation is not
mistaken for a complete tail."""

from __future__ import annotations

import time
from pathlib import Path

from aha_open.util import iso

STALE_AFTER_S = 2 * 24 * 3600
MAX_LINES_CAP = 2000


def _age(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def files_newest_first(path: Path) -> list[Path]:
    """`gateway.log`, then `gateway.log.1`, `.2`, ... (the rotation scheme of RotatingFileHandler)."""
    found = [path] if path.exists() else []
    n = 1
    while (backup := Path(f"{path}.{n}")).exists():
        found.append(backup)
        n += 1
    return found


def read_tail(path: Path, max_lines: int = 100, contains: str | None = None) -> str:
    max_lines = max(1, min(int(max_lines), MAX_LINES_CAP))
    files = files_newest_first(path)
    if not files:
        return f"log: {path}\nNOT FOUND - nothing wrote it yet (the server never ran here, or logs elsewhere)."
    now = time.time()
    lines: list[str] = []
    used: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace").splitlines()
        if contains:
            text = [ln for ln in text if contains in ln]
        used.append(f"{f.name} ({f.stat().st_size} bytes)")
        lines = text + lines
        if len(lines) >= max_lines:
            break
    mtime = files[0].stat().st_mtime
    age = now - mtime
    stale = "  STALE (older than 2 days)" if age > STALE_AFTER_S else ""
    head = f"log: {path}\nlast modified: {iso(mtime)} (age {_age(age)}){stale}\nread: {', '.join(used)}; showing the last {min(len(lines), max_lines)} of {len(lines)} matching lines"
    return head + "\n---\n" + "\n".join(lines[-max_lines:])
