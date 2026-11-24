"""Which server a call talks to: the operator's `live` server or a throwaway `sandbox` (plan D3).

A target is only a data directory and a base URL. Tokens are read from the data directory's token file when needed and
are never put into anything a tool returns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from aha_open import runtime
from aha_open.config import Config, ConfigError, load, split_listen

# Where the tool keeps its own state (gitignored). Tests point it elsewhere so they never touch a sandbox a person is using.
STATE_DIR = Path(os.environ.get("AHA_OPEN_MCP_STATE_DIR") or Path(__file__).resolve().parents[2] / "runs")
REPO = Path(__file__).resolve().parents[3]
SANDBOX_FILE = "sandbox.json"
SANDBOX_MARKER = ".aha-open-sandbox"  # only a directory carrying this file may be deleted by `run_sandbox_stop`
TARGETS = ("sandbox", "live")


class TargetError(Exception):
    """Said to the agent as is: it names what to do."""


@dataclass
class Target:
    name: str
    cfg: Config
    base_url: str

    @property
    def data_dir(self) -> Path:
        return self.cfg.data_dir

    @property
    def log_dir(self) -> Path:
        return self.cfg.data_dir / "logs"

    def token(self) -> str:
        path = self.cfg.local_token_path
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as err:
            raise TargetError(f"no local token at {path} ({err.__class__.__name__}): the server has not run on this data directory yet") from err

    def worker_kinds(self) -> list[str]:
        return sorted({m.worker for m in self.cfg.models if m.enabled})

    def evidence(self) -> str:
        kinds = self.worker_kinds()
        if not kinds:
            return "none (no model enabled)"
        return "🧪 (fake worker)" if kinds == ["fake"] else "✅ (real model)"


def base_url_of(cfg: Config) -> str:
    host, port = split_listen(cfg.listen)
    host = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    return f"{'https' if cfg.profile == 'lan' else 'http'}://{host}:{port}"


def record_of(data_dir: Path) -> dict | None:
    try:
        return json.loads((data_dir / "run" / runtime.RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resolve(name: str) -> Target:
    if name == "live":
        try:
            cfg = load()
        except ConfigError as err:
            raise TargetError(f"target=live: {err}") from err
        return Target("live", cfg, base_url_of(cfg))
    if name == "sandbox":
        try:
            info = json.loads((STATE_DIR / SANDBOX_FILE).read_text(encoding="utf-8"))
            cfg = load(Path(info["data_dir"]))
        except (OSError, ValueError, KeyError, ConfigError) as err:
            raise TargetError("target=sandbox: no sandbox is running - call run_sandbox_start first") from err
        return Target("sandbox", cfg, info["base_url"])
    raise TargetError(f"target must be one of {TARGETS}, got {name!r}")
