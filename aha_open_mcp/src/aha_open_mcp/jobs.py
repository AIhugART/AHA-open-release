"""Write-side `/v1` calls: uploads, jobs, cancel, result download. Everything that changes server state goes through here, so
the `SIDE EFFECTS:` banner of the tools has one place to be true to.

No retry anywhere: a failed job is the client's decision (the server's own rule, kept by its tools)."""

from __future__ import annotations

import hashlib
import struct
import time
import zlib

import httpx

from .client import project_job
from .targets import Target, TargetError

PNG_SIG = b"\x89PNG\r\n\x1a\n"
TERMINAL = ("succeeded", "failed", "cancelled")


def synthetic_png(width: int, height: int) -> bytes:
    """A valid solid-colour PNG, built in pure Python (an input for edit jobs that needs no file on disk)."""
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise TargetError("synthetic_png: width and height must be between 1 and 4096")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + b"\x20\x40\xa0" * width for _ in range(height))
    return PNG_SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _http(target: Target) -> httpx.Client:
    return httpx.Client(
        base_url=target.base_url, headers={"Authorization": f"Bearer {target.token()}"}, timeout=30, verify=target.cfg.profile != "lan",
    )


def _error_of(r: httpx.Response) -> dict:
    try:
        err = r.json().get("error", {})
    except ValueError:
        err = {}
    return {"code": err.get("code"), "message": err.get("message")}


def upload(target: Target, data: bytes, media_type: str = "image/png") -> str:
    sha = hashlib.sha256(data).hexdigest()
    try:
        with _http(target) as http:
            made = http.post("/v1/uploads", json={"bytes": len(data), "sha256": sha, "media_type": media_type})
            if made.status_code != 201:
                raise TargetError(f"upload refused: HTTP {made.status_code} {_error_of(made)}")
            info = made.json()
            size = int(info["part_size"])
            for n, start in enumerate(range(0, len(data), size)):
                put = http.put(f"/v1/uploads/{info['upload_id']}/parts/{n}", content=data[start : start + size])
                if put.status_code != 204:
                    raise TargetError(f"upload part {n} refused: HTTP {put.status_code} {_error_of(put)}")
            done = http.post(f"/v1/uploads/{info['upload_id']}/complete")
            if done.status_code != 200:
                raise TargetError(f"upload complete refused: HTTP {done.status_code} {_error_of(done)}")
            return info["upload_id"]
    except httpx.HTTPError as err:
        raise TargetError(f"cannot reach {target.base_url} ({err.__class__.__name__})") from err


def submit(target: Target, body: dict) -> dict:
    """POST /v1/jobs. A refusal is data (`http_status`, `error`), not an exception: a scenario may expect it."""
    try:
        with _http(target) as http:
            r = http.post("/v1/jobs", json=body)
    except httpx.HTTPError as err:
        raise TargetError(f"cannot reach {target.base_url} ({err.__class__.__name__})") from err
    out: dict = {"http_status": r.status_code}
    if r.status_code == 202:
        out["job_id"] = r.json()["id"]
    else:
        out["error"] = _error_of(r)
    return out


def view(target: Target, job_id: str) -> dict:
    try:
        with _http(target) as http:
            r = http.get(f"/v1/jobs/{job_id}")
    except httpx.HTTPError as err:
        raise TargetError(f"cannot reach {target.base_url} ({err.__class__.__name__})") from err
    if r.status_code != 200:
        raise TargetError(f"job {job_id}: HTTP {r.status_code} {_error_of(r)}")
    return project_job(r.json())


def wait(target: Target, job_id: str, timeout_s: float = 60.0, until: tuple[str, ...] = TERMINAL) -> dict:
    """Poll until the job reaches one of `until` (default: any end state). On timeout the last view is returned with
    `timed_out: true` - never an exception that hides how far the job got."""
    timeout_s = max(1.0, min(float(timeout_s), 600.0))
    deadline = time.monotonic() + timeout_s
    while True:
        job = view(target, job_id)
        if job["status"] in until:
            return job
        if time.monotonic() > deadline:
            return {**job, "timed_out": True}
        time.sleep(0.05)


def cancel(target: Target, job_id: str) -> dict:
    try:
        with _http(target) as http:
            r = http.post(f"/v1/jobs/{job_id}/cancel")
    except httpx.HTTPError as err:
        raise TargetError(f"cannot reach {target.base_url} ({err.__class__.__name__})") from err
    if r.status_code != 200:
        raise TargetError(f"cancel {job_id}: HTTP {r.status_code} {_error_of(r)}")
    return project_job(r.json())


def check_result_image(target: Target, job_id: str, index: int = 0) -> dict:
    """Judge the PIXELS, not the status: the PNG header must say what the job's metadata says."""
    job = view(target, job_id)
    try:
        with _http(target) as http:
            r = http.get(f"/v1/jobs/{job_id}/result", params={"index": index})
    except httpx.HTTPError as err:
        raise TargetError(f"cannot reach {target.base_url} ({err.__class__.__name__})") from err
    if r.status_code != 200:
        return {"fetched": False, "http_status": r.status_code, "error": _error_of(r)}
    data = r.content
    meta = job["results"][index] if index < len(job["results"]) else {}
    width = height = None
    if data[:8] == PNG_SIG and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
    sha = hashlib.sha256(data).hexdigest()
    return {
        "fetched": True, "signature_ok": data[:8] == PNG_SIG, "width": width, "height": height,
        "ihdr_matches": bool(meta) and (width, height) == (meta.get("width"), meta.get("height")),
        "sha256": sha, "sha256_matches": sha == meta.get("sha256"), "bytes_match": len(data) == meta.get("bytes"),
    }
