"""The case that started this tool: a worker that dies natively WHILE LOADING the model, leaving only an exit status.

The sandbox is started with the fake worker's own fault injection (it dies the way the Qwen worker did on 2026-11-23)."""

from __future__ import annotations

import json

import pytest

from aha_open_mcp import sandbox, tools


@pytest.fixture(scope="module")
def crashing_sandbox():
    sandbox.start(crash_on_load=True)
    try:
        yield
    finally:
        sandbox.stop()


def test_a_native_crash_while_loading_is_diagnosed_with_the_worker_log_and_an_action(crashing_sandbox):
    body = {"task": "image.generate", "model": "example-image", "prompt": "anything", "params": {"seed": 1}}
    started = tools.run_job(json.dumps(body))
    job_id = json.loads(started[started.index("{"):])["job_id"]
    tools.wait_job(job_id)
    out = tools.diagnose_job(job_id)
    assert "while loading the model:" in out and "crashed in native code" in out
    assert "died in native code" in out  # the CAUSE line
    assert "worker-fake.log" in out and "\nACTION: " in out
