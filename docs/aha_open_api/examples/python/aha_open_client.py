"""A small AHA open `/v1` client. Python 3.9+, standard library only (copy this one file into your project).

    from aha_open_client import Client
    client = Client()                       # reads AHA_OPEN_URL / AHA_OPEN_TOKEN, or the local token file (see below)
    job, png = client.generate("a red square")

Where the token comes from (first match wins):
  1. the `token=` argument,
  2. the environment variable AHA_OPEN_TOKEN,
  3. the local token file `local_client.token` in the AHA open data directory (only the person at the machine can read it;
     this is what lets the `local` profile work without pasting anything).
The token is never printed or logged by this file. Do not add it to log lines in your own code either.

Specification: docs/03_api_v1.md. Everything here follows it: sizes are never stretched, parts are numbered from 0, a job is
a job + poll, and `busy` answers are waited out with `retry_after_s`.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8890"
FINAL = ("succeeded", "failed", "cancelled")


class AhaOpenError(Exception):
    """An error answer from the server (`{"error": {...}}`), or `code="unreachable"` when nothing answered."""

    def __init__(self, status: int, code: str, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.status, self.code, self.message, self.retry_after_s = status, code, message, retry_after_s


def default_data_dir() -> Path:
    env = os.environ.get("AHA_OPEN_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AHA-open"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "aha-open"


def find_token() -> str | None:
    token = os.environ.get("AHA_OPEN_TOKEN", "").strip()
    if token:
        return token
    path = default_data_dir() / "local_client.token"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


class Client:
    def __init__(self, base_url: str | None = None, token: str | None = None, *, timeout: float = 60.0,
                 cafile: str | None = None, extra_headers: dict | None = None) -> None:
        self.base_url = (base_url or os.environ.get("AHA_OPEN_URL") or DEFAULT_URL).rstrip("/")
        self._token = token or find_token()
        self.timeout = timeout
        # `lan` profile: pass the server's certificate (docs/06_deployment.md) as cafile.
        # `internet` profile: extra_headers={"CF-Access-Client-Id": ..., "CF-Access-Client-Secret": ...}.
        self._ssl = ssl.create_default_context(cafile=cafile) if self.base_url.startswith("https") else None
        self._extra = dict(extra_headers or {})

    # ---------------------------------------------------------------- transport
    def _request(self, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None,
                 headers: dict | None = None, auth: bool = True) -> tuple[int, dict, bytes]:
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        if content_type:
            request.add_header("Content-Type", content_type)
        if auth and self._token:
            request.add_header("Authorization", "Bearer " + self._token)
        for name, value in {**self._extra, **(headers or {})}.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self._ssl) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as err:
            raw = err.read()
            try:
                info = json.loads(raw)["error"]
            except (ValueError, KeyError, TypeError):
                raise AhaOpenError(err.code, "error", f"HTTP {err.code} without a JSON error body") from None
            raise AhaOpenError(err.code, info.get("code", "error"), info.get("message", ""), info.get("retry_after_s")) from None
        except urllib.error.URLError as err:
            raise AhaOpenError(0, "unreachable", f"cannot reach {self.base_url}: {err.reason}") from None

    def _json(self, method: str, path: str, payload: dict | None = None, **kw) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        _, _, raw = self._request(method, path, body=body, content_type="application/json" if payload is not None else None, **kw)
        return json.loads(raw) if raw else {}

    # ---------------------------------------------------------------- the routes
    def health(self) -> dict:
        return self._json("GET", "/v1/health", auth=False)

    def manifest(self) -> dict:
        return self._json("GET", "/v1/manifest")

    def upload(self, data: bytes | str | Path, media_type: str = "image/png") -> str:
        """Upload a file in parts (numbered from 0, each exactly `part_size` bytes, the last one the remainder)."""
        if not isinstance(data, bytes):
            data = Path(data).read_bytes()
        made = self._json("POST", "/v1/uploads", {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "media_type": media_type})
        size = made["part_size"]
        for n, start in enumerate(range(0, len(data), size)):
            self._request("PUT", f"/v1/uploads/{made['upload_id']}/parts/{n}", body=data[start : start + size],
                          content_type="application/octet-stream")
        self._json("POST", f"/v1/uploads/{made['upload_id']}/complete")
        return made["upload_id"]

    def create_job(self, body: dict, *, wait_when_busy: bool = True, max_busy_wait_s: float = 120.0) -> dict:
        """POST /v1/jobs. On `busy` it waits `retry_after_s` and tries again (same client_request_id, so it can never run twice)."""
        body = {"client_request_id": os.urandom(8).hex(), **body}
        deadline = time.monotonic() + max_busy_wait_s
        while True:
            try:
                return self._json("POST", "/v1/jobs", body)
            except AhaOpenError as err:
                if err.code != "busy" or not wait_when_busy or time.monotonic() >= deadline:
                    raise
                time.sleep(min(max(err.retry_after_s or 1.0, 0.5), 30.0))

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def wait(self, job_id: str, *, timeout: float = 900.0, on_progress=None) -> dict:
        """Poll until the job is final. The pause grows from 0.25 s to 2 s: cheap for long jobs, quick for short ones."""
        deadline, pause = time.monotonic() + timeout, 0.25
        while True:
            job = self.job(job_id)
            if on_progress:
                on_progress(job)
            if job["status"] in FINAL:
                return job
            if time.monotonic() >= deadline:
                raise AhaOpenError(0, "timeout", f"job {job_id} is still {job['status']} after {timeout:.0f} s "
                                   "(it keeps running; poll it again or cancel it)")
            time.sleep(pause)
            pause = min(pause * 1.5, 2.0)

    def cancel(self, job_id: str) -> dict:
        return self._json("POST", f"/v1/jobs/{job_id}/cancel")

    def result(self, job_id: str, index: int = 0) -> bytes:
        return self._request("GET", f"/v1/jobs/{job_id}/result?index={index}")[2]

    # ---------------------------------------------------------------- helpers built from the routes
    def pick_model(self, task: str, model: str | None = None) -> dict:
        """The manifest is the source of truth: choose a model that is `ready` and does `task` (or fail with a sentence a person can act on)."""
        models = self.manifest()["models"]
        if model:
            found = next((m for m in models if m["id"] == model), None)
            if found is None:
                offered = ", ".join(m["id"] for m in models) or "nothing"
                raise AhaOpenError(0, "unknown_model", f"this server does not offer {model!r}; it offers: {offered}")
            candidates = [found]
        else:
            candidates = [m for m in models if task in m["tasks"]]
        for m in candidates:
            if m["state"] == "ready" and task in m["tasks"]:
                return m
        states = ", ".join(f"{m['id']} is {m['state']}" for m in candidates) or f"no model does {task}"
        raise AhaOpenError(0, "no_ready_model", f"no model is ready for {task} ({states}). Install the model or wait for it to load; "
                           "see docs/aha_open_api/05_errors_and_troubleshooting.md")

    def generate(self, prompt: str, *, model: str | None = None, width: int | None = None, height: int | None = None,
                 on_progress=None, **params) -> tuple[dict, bytes]:
        """Text to image. With no size given, the first size the model declares is used (sizes are never stretched)."""
        chosen = self.pick_model("image.generate", model)
        if (width is None) != (height is None):
            raise ValueError("give width and height together")
        if width is None:
            first = chosen["capabilities"]["sizes"][0]
            width, height = first["width"], first["height"]
        job = self.create_job({"task": "image.generate", "model": chosen["id"], "prompt": prompt,
                               "params": {"width": width, "height": height, **params}})
        job = self.wait(job["id"], on_progress=on_progress)
        if job["status"] != "succeeded":
            err = job.get("error") or {}
            raise AhaOpenError(0, err.get("code", job["status"]), err.get("message", f"the job {job['status']}"))
        return job, self.result(job["id"])
