"""What a running gateway leaves on disk so that an observer who did not start it can still tell what is running.

Two files in the data directory, both written by `aha-open serve` and never served over HTTP:

* `logs/gateway.log` - the gateway log (the same lines that go to the terminal), rotated by size.
* `run/gateway.json` - `pid`, `started_at`, `version`, `source_digest`. An observer compares `source_digest` with the
  digest of the files on disk: a difference means the process runs older code than the files it can read.

Neither file ever holds a token or a prompt: nothing in the gateway logs either, and a test searches for sample values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
from pathlib import Path

from . import __version__
from .config import Config
from .util import iso, now

LOG_NAME = "gateway.log"
RECORD_NAME = "gateway.json"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
# Loggers that do not propagate to the root logger (uvicorn configures its own handlers).
_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


def source_digest(root: Path | None = None) -> str:
    """sha256 over the relative path and bytes of every `.py` file of the gateway package, in sorted order."""
    root = root or Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel:
            continue
        h.update(rel.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_path(cfg: Config) -> Path:
    return cfg.data_dir / "logs" / LOG_NAME


def record_path(cfg: Config) -> Path:
    return cfg.data_dir / "run" / RECORD_NAME


def install_gateway_log(cfg: Config) -> logging.Handler:
    """Add a rotating file handler to the loggers the gateway writes through."""
    path = log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    handler.set_name(f"aha-open-gateway-log:{path}")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in _LOGGERS:
        logging.getLogger(name).addHandler(handler)
    return handler


def remove_gateway_log(handler: logging.Handler) -> None:
    for name in _LOGGERS:
        logging.getLogger(name).removeHandler(handler)
    handler.close()


def write_run_record(cfg: Config) -> Path:
    """Write `run/gateway.json` atomically (an observer never reads half a file)."""
    path = record_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": os.getpid(), "started_at": iso(now()), "version": __version__,
        "source_digest": source_digest(), "listen": cfg.listen, "profile": cfg.profile,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def remove_run_record(cfg: Config) -> None:
    """On a clean exit only; a record left behind by a crash is detected by its dead pid."""
    try:
        record = json.loads(record_path(cfg).read_text(encoding="utf-8"))
        if record.get("pid") == os.getpid():
            record_path(cfg).unlink()
    except (OSError, ValueError):
        pass
