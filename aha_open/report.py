"""The support report: one text a person can copy or save and send to whoever supports their installation.

Why it exists (plan 2026-11-25_ui_menu_report_about_plan.md, D6-D9): when something fails the supporter does not have this machine. The
report carries what they would otherwise ask for one question at a time: which code is running, what the machine is, what state each model
is in, and the end of the logs, with the age of every file so that an old log is not mistaken for the failure being reported.

Rules, in the order they matter:

* **Nothing leaves this machine from here.** This module builds a string. The person reads it, then copies it or saves it.
* **An allow-list, not a block-list.** The header is built from named fields. `config.toml`, the token files and the database are never
  opened. The logs are the only free text, and they are written by a gateway that never logs a token or a prompt (tests/test_runtime.py).
* **A second layer on top of that**: personal paths become `~` / `<data>` and anything shaped like a secret becomes `[redacted]`.
* **Passive**: it reads the logs and writes nothing, so it cannot create the failure it is asked about.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
from pathlib import Path

from . import __version__
from .catalog import CATALOG, describe_gpu, detect_gpus, model_state
from .config import Config
from .hardware import detect_ram_gib, free_disk_gib
from .runtime import LOG_NAME, record_path, source_digest
from .util import iso, now

DEFAULT_BUDGET = 256 * 1024  # bytes of log text in all; a report that cannot be pasted into a chat is sent as a file (plan D8)
NOTE_MAX = 4000

# Secrets: the shapes this project and its dependencies use. A line that holds one keeps its words and loses the secret.
_SECRETS = [
    (re.compile(r"\baho_[A-Za-z0-9_\-]{6,}"), "[redacted]"),                       # an AHA open client token
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/=]{6,}"), "Bearer [redacted]"),   # an Authorization header
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[redacted]"),                          # a Hugging Face token
    (re.compile(r"(?i)\b(token|secret|password|passwd|api[_-]?key|authorization)(\s*[=:]\s*)(?!\[redacted\])[^\s,;'\"]{4,}"), r"\1\2[redacted]"),
]
# Another person's home folder inside a path: keep the shape, drop the name.
_USER_DIRS = [
    (re.compile(r"(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/\s\"']+"), r"\1<user>"),
    (re.compile(r"(/home/|/Users/)[^/\s\"']+"), r"\1<user>"),
]


def _forms(path: Path) -> list[str]:
    """A path as it appears in text: native, with forward slashes, and doubled backslashes (JSON)."""
    s = str(path)
    return sorted({s, s.replace("\\", "/"), s.replace("\\", "\\\\")}, key=len, reverse=True)


def redact(text: str, cfg: Config | None = None) -> str:
    """Personal paths and secret-shaped values removed. The data folder first (it is the longer path), then the home folder."""
    if cfg is not None:
        for form in _forms(cfg.data_dir):
            text = re.sub(re.escape(form), "<data>", text, flags=re.I)
    try:
        home = Path.home()
    except RuntimeError:
        home = None
    if home is not None and str(home) not in ("", ".", "/"):
        for form in _forms(home):
            text = re.sub(re.escape(form), "~", text, flags=re.I)
    for pattern, repl in _USER_DIRS:
        text = pattern.sub(repl, text)
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s ago"
    if seconds < 5400:
        return f"{round(seconds / 60)} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.1f} days ago"


def split_budget(budget: int, sizes: dict[str, int]) -> dict[str, int]:
    """Water-filling: a small file keeps all of itself, the rest share what is left equally. No file is starved by a large one."""
    out = {name: 0 for name in sizes}
    left = {n: s for n, s in sizes.items() if s > 0}
    remaining = budget
    while left and remaining > 0:
        share = remaining // len(left)
        if share == 0:
            break
        small = {n: s for n, s in left.items() if s <= share}
        if not small:
            for name in left:
                out[name] = share
            break
        for name, size in small.items():
            out[name] = size
            remaining -= size
            del left[name]
    return out


_STAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[,.]\d{3} ")
_PEER = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b")
READ_CAP = 16 * 1024 * 1024  # the most log text read per file to look for the end of it; the gateway log is at most 4 x 5 MiB
REPEAT_MIN = 3 # a line that comes this many times in a row (ignoring its time and the caller's port) is shown once, with the count


def collapse_repeats(text: str) -> str:
    """Runs of the same line become the first line and a count.

    Measured on a real log (plan 2026-11-25_ui_menu_report_about_plan.md, U2): a client that waits for a job asks every 60 ms, and the
    last 128 KiB of the gateway log was nothing else. The line that explains a failure sat before them, outside the budget.
    """
    out: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        key = _PEER.sub("", _STAMP.sub("", lines[i]))
        j = i + 1
        while j < len(lines) and key and _PEER.sub("", _STAMP.sub("", lines[j])) == key:
            j += 1
        if j - i >= REPEAT_MIN:
            stamp = _STAMP.match(lines[j - 1])
            out.append(lines[i])
            out.append(f"    ... the same line {j - i - 1} more times" + (f", the last at {stamp.group(0).strip()}" if stamp else ""))
        else:
            out.extend(lines[i:j])
        i = j
    return "\n".join(out)


def _tail(paths: list[Path], nbytes: int) -> str:
    """The end of the files taken oldest to newest, repeated lines collapsed, cut to `nbytes` from a whole line on."""
    # The collapse frees room, and how much is unknowable before reading (a 60 ms poll fills a megabyte with one line), so read up to a hard cap.
    raw, cut = _read_tail(paths, READ_CAP)
    data = collapse_repeats(raw.replace("\r\n", "\n")).encode("utf-8")  # Windows writes the log with CRLF
    if len(data) > nbytes:
        data, cut = data[-nbytes:], True
    if cut:
        newline = data.find(b"\n")
        if newline >= 0:
            data = data[newline + 1:]
    return data.decode("utf-8", errors="replace")


def _read_tail(paths: list[Path], nbytes: int) -> tuple[str, bool]:
    """(the last `nbytes` of the files taken oldest to newest, whether older text was left out)."""
    chunks: list[bytes] = []
    need = nbytes
    cut = False  # the text starts in the middle of something older: its first line is a fragment
    for path in reversed(paths):
        if need <= 0:
            cut = True
            break
        try:
            size = path.stat().st_size
            start = max(0, size - need)
            with path.open("rb") as fh:
                fh.seek(start)
                data = fh.read()
        except OSError:
            continue
        chunks.append(data)
        need -= len(data)
        cut = start > 0
    return b"".join(reversed(chunks)).decode("utf-8", errors="replace"), cut


def _log_files(cfg: Config) -> list[tuple[str, list[Path]]]:
    """(label, files oldest to newest). The gateway log is read across its rotated backups."""
    logs = cfg.data_dir / "logs"
    found: list[tuple[str, list[Path]]] = []
    rotated = sorted((p for p in logs.glob(LOG_NAME + ".*") if p.suffix[1:].isdigit()), key=lambda p: int(p.suffix[1:]), reverse=True)
    chain = rotated + ([logs / LOG_NAME] if (logs / LOG_NAME).exists() else [])  # gateway.log.3 is the oldest
    if chain:
        found.append((LOG_NAME, chain))
    for path in sorted(logs.glob("worker-*.log")) + [logs / "start.log"]:
        if path.exists():
            found.append((path.name, [path]))
    return found


def _machine_lines(cfg: Config) -> list[str]:
    gpus = detect_gpus(ttl_s=0)
    ram = detect_ram_gib()
    lines = [
        f"OS: {platform.platform()}",
        f"Time zone: UTC{time.strftime('%z')[:3]}:{time.strftime('%z')[3:]} - the times inside the logs are local time, the times in this header are UTC",
        f"Python: {sys.version.split()[0]} ({platform.machine()})",
        "GPU: " + ("; ".join(describe_gpu(g, ", ") for g in gpus) if gpus else "no NVIDIA GPU found (nvidia-smi)"),
        f"GPU memory limit: {cfg.gpu_memory_limit_gib} GiB" if cfg.gpu_memory_limit_gib else "GPU memory limit: none",
        f"RAM: {ram:.0f} GiB" if ram else "RAM: not measured",
    ]
    try:
        lines.append(f"Free disk where models are stored: {free_disk_gib(cfg.data_dir):.0f} GiB")
    except OSError:
        lines.append("Free disk where models are stored: not measured")
    return lines


def _gateway_lines(cfg: Config) -> list[str]:
    lines = [f"Version on disk: {__version__}", f"Profile: {cfg.profile}", f"Listen: {cfg.listen}", f"Token required: {cfg.require_token}"]
    try:
        rec = json.loads(record_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        lines.append("Run record: none (the gateway was never started from this data folder, or has not written it yet)")
        return lines
    lines.append(f"Run record: pid {rec.get('pid')}, version {rec.get('version')}, started {rec.get('started_at')}")
    running, on_disk = rec.get("source_digest"), source_digest()
    if running and running == on_disk:
        lines.append("Code: the running gateway matches the files on disk")
    else:
        lines.append("Code: DIFFERENT - the gateway that wrote the run record was started from older (or other) files than the ones on disk now; "
                     "restart it before trusting a result")
    return lines


def _model_lines(cfg: Config) -> list[str]:
    lines = []
    for mc in cfg.models:
        spec = CATALOG.get(mc.id)
        if spec is None:
            lines.append(f"{mc.id}: not a known model")
            continue
        state, detail = model_state(cfg, mc, spec)
        lines.append(f"{mc.id} [{mc.variant or 'default'}]: {state}" + (f" - {detail}" if detail else ""))
    return lines or ["(no model configured)"]


def build_report(cfg: Config, note: str = "", budget: int = DEFAULT_BUDGET) -> str:
    """The whole report as text. `note` is the person's own words (at most NOTE_MAX characters)."""
    parts = [f"AHA open support report\nCreated: {iso(now())} (UTC)"]
    note = (note or "").strip()[:NOTE_MAX]
    if note:
        parts.append("== What happened (the person's words) ==\n" + note)
    parts.append("== Machine ==\n" + "\n".join(_machine_lines(cfg)))
    parts.append("== Gateway ==\n" + "\n".join(_gateway_lines(cfg)))
    parts.append("== Models ==\n" + "\n".join(_model_lines(cfg)))
    files = _log_files(cfg)
    sizes = {label: sum(p.stat().st_size for p in chain if p.exists()) for label, chain in files}
    shares = split_budget(budget, sizes)
    clock = time.time()
    if not files:
        parts.append("== Logs ==\n(no log file in the data folder yet)")
    for label, chain in files:
        newest = chain[-1].stat().st_mtime
        head = (f"== Log: {label} ==\nsize {sizes[label]} bytes, last written {iso(newest)} ({_age(clock - newest)}); "
                f"showing the end, at most {shares[label]} bytes, a line that repeats is shown once with its count")
        parts.append(head + "\n" + _tail(chain, shares[label]).rstrip())
    return redact("\n\n".join(parts) + "\n", cfg)
