"""Read a .safetensors file by its own header, tolerating bytes after the last tensor.

Why this exists: the community MiniMax H3 files carry a 61-83 byte trailer after the last tensor
(``\\nL2P_bypass_<file name>_<timestamp>\\n``, left by the publisher's upload tool). The published sha256 includes it,
so the file cannot be "fixed", and ``safetensors`` 0.8 refuses it ("file not fully covered"). The header is still exact:
every tensor is addressed by its own offsets, so reading by offset is correct and also lets us read one tensor at a time
without loading the file (a 27 GB text encoder must never be loaded whole into 31 GiB of RAM).
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import torch

_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "BOOL": torch.bool,
}
_MAX_HEADER = 100 * 1024 * 1024  # a header larger than this is a damaged file, not a model


class SafetensorsFile:
    """Header + memory map of one file. ``get`` copies a single tensor out (CPU); the file stays on disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        (header_len,) = struct.unpack("<Q", self._file.read(8))
        if not 0 < header_len < _MAX_HEADER:
            raise ValueError(f"{self.path.name}: header length {header_len} is not a safetensors header")
        header = json.loads(self._file.read(header_len))
        self.metadata: dict = header.pop("__metadata__", {})
        self._header = header
        self._base = 8 + header_len
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        available = len(self._map) - self._base
        end = max((t["data_offsets"][1] for t in header.values()), default=0)
        if end > available:  # a trailer is fine, a missing tail is not
            raise ValueError(f"{self.path.name}: truncated, header expects {end} data bytes and the file has {available}")

    def close(self) -> None:
        self._map.close()
        self._file.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def keys(self):
        return self._header.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._header

    def shape(self, key: str) -> list[int]:
        return list(self._header[key]["shape"])

    def dtype(self, key: str) -> torch.dtype:
        return _DTYPES[self._header[key]["dtype"]]

    def get(self, key: str) -> torch.Tensor:
        entry = self._header[key]
        start, end = entry["data_offsets"]
        dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
        if end == start:
            return torch.empty(shape, dtype=dtype)
        # bytearray copies out of the map, so the returned tensor owns writable memory and does not pin the file
        return torch.frombuffer(bytearray(self._map[self._base + start : self._base + end]), dtype=dtype).reshape(shape)

    def get_json(self, key: str) -> dict:
        """A ``*.comfy_quant`` entry: a UTF-8 JSON document stored as a uint8 tensor."""
        return json.loads(bytes(self.get(key).tolist()).decode("utf-8"))
