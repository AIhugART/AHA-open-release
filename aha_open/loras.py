"""LoRA files the operator has put on this server.

Design: LoRAs are *operator-managed*, never uploaded through /v1. A client can only pick one by id from the manifest.
Why: a LoRA is model weights - it can be hundreds of MB, carries its own licence, and a hostile file is a way into the
worker. Only `.safetensors` is accepted (no pickle), the id is restricted to a safe alphabet (it becomes a file name and
an adapter name), and nothing here imports torch.

Layout (docs/04_architecture.md section 4):  <data_dir>/loras/<id>.safetensors  [+ <id>.json]
Sidecar fields (all optional): display_name, default_scale, trigger, license, notes.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .util import sha256_file

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_FILE_BYTES = 4 * 2**30
MAX_HEADER_BYTES = 64 * 2**20
MAX_SIDECAR_BYTES = 64 * 1024
LORA_KEY_MARKERS = ("lora_A", "lora_B", "lora_down", "lora_up")

_hash_cache: dict[tuple[str, int, int], str] = {}


class LoraError(Exception):
    """A LoRA file or request the operator can fix; the message says how."""


@dataclass(frozen=True)
class Lora:
    id: str
    path: Path
    bytes: int
    display_name: str
    default_scale: float
    trigger: str = ""
    license: str = ""
    notes: str = ""

    def public(self) -> dict:
        """What the manifest tells clients. Never the path."""
        view = {"id": self.id, "display_name": self.display_name, "default_scale": self.default_scale, "bytes": self.bytes}
        for key in ("trigger", "license", "notes"):
            if getattr(self, key):
                view[key] = getattr(self, key)
        return view

    def sha256(self) -> str:
        stat = self.path.stat()
        key = (str(self.path), stat.st_mtime_ns, stat.st_size)
        if key not in _hash_cache:
            _hash_cache[key] = sha256_file(self.path)
        return _hash_cache[key]


def lora_dir(cfg: Config) -> Path:
    return cfg.data_dir / "loras"


def check_safetensors(path: Path) -> list[str]:
    """Tensor names of a safetensors file, read from its header only. Raises LoraError when it is not a LoRA."""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise LoraError(f"{path.name} is larger than {MAX_FILE_BYTES // 2**30} GiB")
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise LoraError(f"{path.name} is not a safetensors file (too short)")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
            raise LoraError(f"{path.name} is not a safetensors file (bad header)")
        try:
            header = json.loads(fh.read(header_len))
        except ValueError as err:
            raise LoraError(f"{path.name} is not a safetensors file (header is not JSON)") from err
    names = [k for k in header if k != "__metadata__"]
    if not names:
        raise LoraError(f"{path.name} contains no tensors")
    if not any(marker in name for name in names for marker in LORA_KEY_MARKERS):
        raise LoraError(f"{path.name} has no LoRA tensors (expected names containing lora_A/lora_B or lora_down/lora_up)")
    return names


def _sidecar(path: Path) -> dict:
    side = path.with_suffix(".json")
    if not side.is_file():
        return {}
    if side.stat().st_size > MAX_SIDECAR_BYTES:
        raise LoraError(f"{side.name} is larger than {MAX_SIDECAR_BYTES // 1024} KiB")
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except ValueError as err:
        raise LoraError(f"{side.name} is not valid JSON") from err
    if not isinstance(data, dict):
        raise LoraError(f"{side.name} must be a JSON object")
    return data


def check_meta(lora_id: str, meta: dict) -> tuple[float, dict]:
    """(default scale, the four texts) of a LoRA's sidecar. One rule for files an operator wrote and for the admin page."""
    scale = meta.get("default_scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 2:
        raise LoraError(f"{lora_id}.json: default_scale must be a number above 0 and at most 2")
    texts = {}
    for key in ("display_name", "trigger", "license", "notes"):
        value = meta.get(key, "")
        if not isinstance(value, str) or len(value) > 500:
            raise LoraError(f"{lora_id}.json: {key} must be a text of at most 500 characters")
        texts[key] = value
    return float(scale), texts


def _load(path: Path) -> Lora:
    lora_id = path.stem
    if not ID_RE.match(lora_id):
        raise LoraError(f"{path.name}: the file name (without .safetensors) must be lowercase letters, digits, '-' or '_' (at most 64)")
    check_safetensors(path)
    scale, texts = check_meta(lora_id, _sidecar(path))
    return Lora(
        id=lora_id, path=path.resolve(), bytes=path.stat().st_size, display_name=texts["display_name"] or lora_id,
        default_scale=float(scale), trigger=texts["trigger"], license=texts["license"], notes=texts["notes"],
    )


def scan(cfg: Config) -> tuple[dict[str, Lora], list[str]]:
    """(usable LoRAs by id, one problem sentence per file that was skipped). Cheap: reads headers, never tensors."""
    found: dict[str, Lora] = {}
    problems: list[str] = []
    folder = lora_dir(cfg)
    if not folder.is_dir():
        return found, problems
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() in (".json", ".tmp"):
            continue
        if path.suffix != ".safetensors":
            problems.append(f"{path.name}: only .safetensors files are accepted (pickle-based formats are refused)")
            continue
        try:
            lora = _load(path)
        except LoraError as err:
            problems.append(str(err))
            continue
        found[lora.id] = lora
    return found, problems


def add(cfg: Config, source: Path, lora_id: str | None = None, move: bool = False, **meta: object) -> Lora:
    """Validate and copy (or, with move=True, rename) a LoRA into the server's folder.

    Atomic: a half-copied file is never offered, and nothing is written when the file or its details are refused.
    """
    source = Path(source)
    if not source.is_file():
        raise LoraError(f"{source} is not a file")
    if source.suffix != ".safetensors":
        raise LoraError("only .safetensors files are accepted")
    lora_id = lora_id or re.sub(r"[^a-z0-9_-]+", "-", source.stem.lower()).strip("-")
    if not ID_RE.match(lora_id):
        raise LoraError(f"id {lora_id!r} is not allowed: use lowercase letters, digits, '-' or '_' (at most 64)")
    check_safetensors(source)
    clean = {k: v for k, v in meta.items() if v not in (None, "")}
    check_meta(lora_id, clean)  # refuse bad details before anything is written, not after
    folder = lora_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{lora_id}.safetensors"
    if target.exists():
        raise LoraError(f"a LoRA with id {lora_id!r} already exists; remove it first or choose another id")
    partial = target.with_suffix(".tmp")
    if move:
        source.replace(partial)
    else:
        shutil.copyfile(source, partial)
    if clean:
        target.with_suffix(".json").write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    partial.replace(target)
    return _load(target)


def remove(cfg: Config, lora_id: str) -> bool:
    if not ID_RE.match(lora_id):
        return False
    target = lora_dir(cfg) / f"{lora_id}.safetensors"
    if not target.exists():
        return False
    target.unlink()
    target.with_suffix(".json").unlink(missing_ok=True)
    return True
