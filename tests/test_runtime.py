"""`aha-open serve` leaves evidence on disk for an observer that did not start it (aha_open/runtime.py).

The server here is a REAL `aha-open serve` child process with the fake worker: the evidence is written by the production
code path, not by calling the helpers in-process.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from aha_open import runtime
from aha_open.config import Config, ModelConfig, render

PROMPT = "SENTINEL-PROMPT-b7d41e a lighthouse at dusk"


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":  # os.kill(pid, 0) would TERMINATE the process on Windows
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def kill_pid(pid: int) -> None:
    if pid_alive(pid):
        os.kill(pid, 9 if sys.platform != "win32" else 15)  # on Windows any signal terminates
    deadline = time.monotonic() + 15
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    port = free_port()
    cfg = Config(data_dir=data, listen=f"127.0.0.1:{port}", models=[ModelConfig(id="example-image", worker="fake")])
    (data / "config.toml").write_text(render(cfg), encoding="utf-8")
    env = dict(os.environ, AHA_FAKE_STEP_S="0.005", AHA_FAKE_LOAD_S="0.1", PYTHONUTF8="1")
    proc = subprocess.Popen(
        [sys.executable, "-c", "from aha_open.cli import main; main()", "--data-dir", str(data), "serve"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if httpx.get(base + "/v1/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("aha-open serve did not come up")
            time.sleep(0.1)
        yield cfg, proc, base
    finally:
        # On Windows the venv's python.exe is a launcher: `proc` is the launcher, the gateway is its child. The gateway's own
        # record names the real pid; kill it too or the server outlives the test.
        try:
            kill_pid(json.loads(runtime.record_path(cfg).read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError):
            pass
        proc.kill()
        proc.wait(timeout=30)


def run_one_job(cfg: Config, base: str) -> str:
    token = cfg.local_token_path.read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {token}"}
    body = {"task": "image.generate", "model": "example-image", "prompt": PROMPT, "params": {"seed": 1}}
    made = httpx.post(base + "/v1/jobs", json=body, headers=headers, timeout=10)
    assert made.status_code == 202, made.text
    job_id = made.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = httpx.get(f"{base}/v1/jobs/{job_id}", headers=headers, timeout=10).json()["status"]
        if status in ("succeeded", "failed", "cancelled"):
            assert status == "succeeded"
            return token
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def test_the_running_gateway_records_its_pid_and_the_digest_of_the_code_it_runs(served):
    cfg, _, _ = served
    record = json.loads(runtime.record_path(cfg).read_text(encoding="utf-8"))
    assert pid_alive(record["pid"])  # not proc.pid: the Windows venv launcher is the parent of the gateway
    assert record["source_digest"] == runtime.source_digest()
    assert record["version"] and record["started_at"].endswith("Z")


def test_the_digest_changes_when_the_gateway_code_changes(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    before = runtime.source_digest(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert runtime.source_digest(tmp_path) != before


def test_the_gateway_log_is_a_file_and_names_the_worker_it_started(served):
    cfg, _, base = served
    run_one_job(cfg, base)
    text = runtime.log_path(cfg).read_text(encoding="utf-8")
    assert "worker started: fake pid=" in text and "worker_sha256=" in text
    assert "/v1/jobs" in text  # the request line of the access log: the file is a real copy of what the terminal shows


def test_neither_file_holds_a_token_or_a_prompt(served):
    cfg, _, base = served
    token = run_one_job(cfg, base)
    for path in (runtime.log_path(cfg), runtime.record_path(cfg)):
        text = path.read_text(encoding="utf-8")
        assert token not in text, path.name
        assert "SENTINEL-PROMPT" not in text, path.name


def test_a_record_left_by_a_killed_gateway_names_a_pid_that_is_gone(served):
    cfg, _, _ = served
    record = json.loads(runtime.record_path(cfg).read_text(encoding="utf-8"))
    kill_pid(record["pid"])
    assert not pid_alive(record["pid"])
    assert json.loads(runtime.record_path(cfg).read_text(encoding="utf-8")) == record  # a crash cannot clean up


def test_a_clean_exit_removes_only_its_own_record(tmp_path):
    cfg = Config(data_dir=tmp_path)
    runtime.write_run_record(cfg)
    runtime.remove_run_record(cfg)
    assert not runtime.record_path(cfg).exists()
    runtime.record_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    runtime.record_path(cfg).write_text(json.dumps({"pid": -1}), encoding="utf-8")  # another process's record
    runtime.remove_run_record(cfg)
    assert runtime.record_path(cfg).exists()


def test_the_log_file_rotates(tmp_path, monkeypatch):
    import logging

    monkeypatch.setattr(runtime, "LOG_MAX_BYTES", 2000)
    cfg = Config(data_dir=tmp_path)
    handler = runtime.install_gateway_log(cfg)
    try:
        for i in range(200):
            logging.getLogger("aha_open.test").warning("line %d %s", i, "x" * 50)
    finally:
        runtime.remove_gateway_log(handler)
    assert Path(str(runtime.log_path(cfg)) + ".1").exists()


def test_the_support_report_of_a_real_gateway_that_ran_a_job_has_neither_the_token_nor_the_prompt(served):
    """The gateway never logs either, and the report adds a second layer. This is the one place both are checked against the real thing."""
    from aha_open import report

    cfg, _, base = served
    token = run_one_job(cfg, base)
    text = report.build_report(cfg)
    assert "worker started: fake pid=" in text and "== Log: gateway.log ==" in text and "the running gateway matches the files on disk" in text
    assert token not in text and "SENTINEL-PROMPT" not in text
    assert str(cfg.data_dir) not in text  # the data folder is `<data>` in a report


def test_every_line_of_the_server_log_is_written_once(served):
    """Seen in a real log (plan 2026-11-25_ui_menu_report_about_plan.md, U2): `Started server process` twice, because the file handler sat on
    `uvicorn.error` AND on its parent `uvicorn`, which `uvicorn.error` propagates to."""
    cfg, _, _ = served
    text = runtime.log_path(cfg).read_text(encoding="utf-8")
    assert text.count("Application startup complete.") == 1 and text.count("Started server process") == 1
