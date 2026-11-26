"""Read what is really inside an MP4 with the standard library only (plan 2026-11-24_minimax_h3_plan.md, M5).

Why not a media library: this tool runs in its own light environment, and a check that uses the library that WROTE the file would
agree with it by construction. The container is a tree of boxes (``size(4) type(4) payload``, big-endian); everything a check needs
is in ``moov``: per track the handler (video or sound), the sample entry (codec, picture size, channels, sample rate), the sample
count and durations. ``probe_mp4`` returns those numbers or raises ``Mp4Error`` saying which box is wrong.
"""

from __future__ import annotations

import struct


class Mp4Error(ValueError):
    """The bytes are not a well-formed MP4 (said as is to the agent)."""


def _boxes(data: bytes, start: int, end: int):
    """(type, payload_start, payload_end) of every box in data[start:end]; a box that runs past ``end`` is an error, not a stop."""
    pos = start
    while pos < end:
        if end - pos < 8:
            raise Mp4Error(f"{end - pos} stray bytes at offset {pos}, too few for a box header")
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1:
            if end - pos < 16:
                raise Mp4Error(f"box at offset {pos} announces a 64-bit size but the file ends")
            size, header = struct.unpack(">Q", data[pos + 8 : pos + 16])[0], 16
        elif size == 0:
            size = end - pos  # "extends to the end of the file"
        if size < header or pos + size > end:
            raise Mp4Error(f"box {kind!r} at offset {pos} is {size} bytes long and runs past the end ({end - pos} left): the file is cut short")
        yield kind, pos + header, pos + size
        pos += size


def _find(data: bytes, start: int, end: int, kind: bytes):
    return next(((s, e) for k, s, e in _boxes(data, start, end) if k == kind), None)


def _need(data: bytes, start: int, end: int, kind: bytes, where: str) -> tuple[int, int]:
    found = _find(data, start, end, kind)
    if found is None:
        raise Mp4Error(f"no {kind.decode('latin-1')} box in {where}")
    return found


def _u(data: bytes, offset: int, size: int) -> int:
    chunk = data[offset : offset + size]
    if len(chunk) != size:
        raise Mp4Error(f"a field at offset {offset} runs past the end of its box")
    return int.from_bytes(chunk, "big")


def _comment(data: bytes, moov: tuple[int, int]) -> str | None:
    """The ``\xa9cmt`` item of udta/meta/ilst (where FFmpeg writes a container ``comment``), or None."""
    udta = _find(data, *moov, b"udta")
    meta = _find(data, *udta, b"meta") if udta else None
    if not meta:
        return None
    for skip in (4, 0):  # ISO 'meta' is a full box (version+flags first); the QuickTime form is not
        try:
            ilst = _find(data, meta[0] + skip, meta[1], b"ilst")
        except Mp4Error:
            continue
        if ilst:
            item = _find(data, *ilst, b"\xa9cmt")
            value = _find(data, *item, b"data") if item else None
            if value:
                return data[value[0] + 8 : value[1]].decode("utf-8", "replace")  # 4 bytes type + 4 bytes locale, then the text
    return None


def _track(data: bytes, trak: tuple[int, int]) -> dict:
    mdia = _need(data, *trak, b"mdia", "a track")
    mdhd = _need(data, *mdia, b"mdhd", "a track's mdia")
    version = data[mdhd[0]]
    timescale = _u(data, mdhd[0] + (20 if version == 1 else 12), 4)
    hdlr = _need(data, *mdia, b"hdlr", "a track's mdia")
    handler = data[hdlr[0] + 8 : hdlr[0] + 12].decode("latin-1")
    stbl = _need(data, *_need(data, *mdia, b"minf", "a track's mdia"), b"stbl", "a track's minf")
    stsz = _need(data, *stbl, b"stsz", "a track's stbl")
    samples = _u(data, stsz[0] + 8, 4)
    stts = _need(data, *stbl, b"stts", "a track's stbl")
    entries = _u(data, stts[0] + 4, 4)
    ticks = sum(_u(data, stts[0] + 8 + 8 * i, 4) * _u(data, stts[0] + 12 + 8 * i, 4) for i in range(entries))
    stsd = _need(data, *stbl, b"stsd", "a track's stbl")
    entry = stsd[0] + 8  # version/flags + entry count, then the first sample entry
    codec = data[entry + 4 : entry + 8].decode("latin-1")
    out = {"handler": handler, "codec": codec, "samples": samples, "timescale": timescale, "duration_s": round(ticks / timescale, 4) if timescale else 0.0}
    if handler == "vide":
        out["width"], out["height"] = _u(data, entry + 32, 2), _u(data, entry + 34, 2)
        out["fps"] = round(samples / out["duration_s"], 3) if out["duration_s"] else 0.0
    elif handler == "soun":
        out["channels"], out["sample_rate"] = _u(data, entry + 24, 2), _u(data, entry + 32, 2)  # the rate is 16.16: its integer half
    return out


def probe_mp4(data: bytes) -> dict:
    """``{"brand", "video": {...}|None, "audio": {...}|None, "comment": str|None, "tracks": n}`` or ``Mp4Error``."""
    if len(data) < 16:
        raise Mp4Error(f"{len(data)} bytes: too short to be an MP4")
    top = {kind: (s, e) for kind, s, e in _boxes(data, 0, len(data))}
    if b"ftyp" not in top:
        raise Mp4Error("no ftyp box: this is not an MP4 (a video result must be one)")
    if b"moov" not in top:
        raise Mp4Error("no moov box: the file has pictures but no index of them (an interrupted write)")
    brand = data[top[b"ftyp"][0] : top[b"ftyp"][0] + 4].decode("latin-1")
    tracks = [_track(data, (s, e)) for kind, s, e in _boxes(data, *top[b"moov"]) if kind == b"trak"]
    video = next((t for t in tracks if t["handler"] == "vide"), None)
    audio = next((t for t in tracks if t["handler"] == "soun"), None)
    return {"brand": brand, "video": video, "audio": audio, "comment": _comment(data, top[b"moov"]), "tracks": len(tracks)}
