"""Conformance harness: a REAL gateway (uvicorn, real sockets, real worker subprocesses) driven over HTTP.

Nothing here reaches into gateway internals to make a test pass; helpers only read the files an operator could read.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from aha_open.config import Config, ModelConfig
from aha_open.server import make_server

FAKE_MODELS = [ModelConfig(id="example-image", worker="fake"), ModelConfig(id="example-video", worker="fake")]


@dataclass
class Running:
    cfg: Config
    server: object
    thread: threading.Thread
    base: str
    scheme: str = "http"
    tokens: dict = field(default_factory=dict)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30)


def start_gateway(cfg: Config) -> Running:
    server = make_server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the gateway did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    scheme = "https" if cfg.profile == "lan" else "http"
    return Running(cfg, server, thread, f"{scheme}://127.0.0.1:{port}", scheme)


class Api:
    """A tiny client: what any /v1 consumer would write."""

    def __init__(self, base: str, token: str | None = None, host: str | None = None, verify: bool = True) -> None:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if host:
            headers["Host"] = host
        self.http = httpx.Client(base_url=base, headers=headers, timeout=30, verify=verify)

    def __getattr__(self, name):
        return getattr(self.http, name)

    def job(self, body: dict) -> httpx.Response:
        return self.http.post("/v1/jobs", json=body)

    def wait(self, job_id: str, states=("succeeded", "failed", "cancelled"), timeout: float = 30) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            data = self.http.get(f"/v1/jobs/{job_id}").json()
            if data["status"] in states:
                return data
            if time.monotonic() > deadline:
                raise AssertionError(f"job {job_id} stuck in {data['status']}")
            time.sleep(0.02)

    def upload(self, data: bytes, media_type: str = "image/png", part_size: int | None = None) -> str:
        sha = hashlib.sha256(data).hexdigest()
        made = self.http.post("/v1/uploads", json={"bytes": len(data), "sha256": sha, "media_type": media_type})
        assert made.status_code == 201, made.text
        info = made.json()
        size = part_size or info["part_size"]
        for n, start in enumerate(range(0, len(data), size)):
            put = self.http.put(f"/v1/uploads/{info['upload_id']}/parts/{n}", content=data[start : start + size])
            assert put.status_code == 204, put.text
        done = self.http.post(f"/v1/uploads/{info['upload_id']}/complete")
        assert done.status_code == 200, done.text
        return info["upload_id"]


def image_job(prompt: str = "a red square", **params) -> dict:
    return {"task": "image.generate", "model": "example-image", "prompt": prompt, "params": {"seed": 1, **params}}


@pytest.fixture(autouse=True)
def fast_fake_worker(monkeypatch):
    monkeypatch.setenv("AHA_FAKE_STEP_S", "0.005")
    monkeypatch.setenv("AHA_FAKE_LOAD_S", "0.1")


@pytest.fixture
def make_gateway(tmp_path):
    started: list[Running] = []

    def make(**overrides) -> Running:
        data = tmp_path / f"data{len(started)}"
        cfg = Config(data_dir=data, listen="127.0.0.1:0", models=list(overrides.pop("models", FAKE_MODELS)))
        cfg.job_timeout_s = 60
        cfg.cancel_grace_s = 2
        for key, value in overrides.items():
            setattr(cfg, key, value)
        running = start_gateway(cfg)
        started.append(running)
        token_file = cfg.local_token_path
        running.tokens["local"] = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else None
        return running

    yield make
    for running in started:
        running.stop()


@pytest.fixture
def gw(make_gateway) -> Running:
    return make_gateway()


@pytest.fixture
def api(gw) -> Api:
    return Api(gw.base, gw.tokens["local"])


def new_client_token(gw: Running, name: str) -> str:
    """Create a second client the way an operator would: `aha-open clients create`."""
    from aha_open.auth import Auth
    from aha_open.store import Store

    store = Store(gw.cfg.data_dir)
    try:
        return Auth(gw.cfg, store).create_client(name)[1]
    finally:
        store.close()


def tree_bytes(root: Path, skip: tuple[str, ...] = ()) -> bytes:
    """Every byte under root, for the negative-search tests (the needle must appear 0 times)."""
    blob = b""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in skip:
            try:
                blob += path.read_bytes()
            except OSError:
                continue
    return blob
