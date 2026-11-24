"""`diagnose_job` against a real gateway, with the failures the fake worker can provoke on demand."""

from __future__ import annotations

import json
import re
import time

import pytest

from aha_open_mcp import targets, tools

NEEDLE = "SENTINEL-PROMPT-5d0e3a"


def submit(prompt: str, **params) -> str:
    body = {"task": "image.generate", "model": "example-image", "prompt": prompt, "params": {"seed": 1, **params}}
    started = tools.run_job(json.dumps(body))
    return json.loads(started[started.index("{"):])["job_id"]


def failed_job(marker: str, **params) -> str:
    job_id = submit(f"{NEEDLE} {marker}", **params)
    tools.wait_job(job_id)
    return job_id


@pytest.mark.parametrize(
    ("marker", "code", "cause"),
    [
        ("__fail_oom", "gpu_out_of_memory", "ran out of VRAM"),
        ("__crash_native", "generation_failed", "died in native code"),
        ("__fail", "generation_failed", "The pipeline failed for a reason that is not memory"),
    ],
)
def test_a_failure_gets_its_cause_and_its_action(sandbox_up, marker, code, cause):
    out = tools.diagnose_job(failed_job(marker))
    assert f"error: {code}:" in out and cause in out
    assert "\nACTION: " in out


def test_a_worker_output_of_the_wrong_size_is_named_as_a_worker_defect(sandbox_up):
    out = tools.diagnose_job(failed_job("__wrongsize", width=256, height=256))
    assert "another size than the one accepted" in out and "worker defect, not a client mistake" in out


def test_the_log_window_hides_the_pollers_own_lines_and_says_how_many(sandbox_up):
    out = tools.diagnose_job(failed_job("__fail"))
    hidden = int(re.search(r"\((\d+) poll lines hidden\)", out).group(1))
    assert hidden > 0  # wait_job polled while the job ran
    assert "GET /v1/jobs/" not in out and "GET /v1/health" not in out


def test_the_window_starts_at_the_job_not_at_the_start_of_the_log(sandbox_up):
    older = failed_job("__fail")
    time.sleep(3.5)  # created_at has a one-second resolution
    newer = failed_job("__fail")
    out = tools.diagnose_job(newer)
    assert older not in out


def test_the_worker_log_section_names_its_file_age_and_last_stage(sandbox_up):
    out = tools.diagnose_job(failed_job("__fail"))
    assert "== worker-fake.log: last stage reported: none (the worker wrote no stage lines)" in out
    assert "last modified: " in out and "(age " in out


def test_neither_the_token_nor_the_prompt_is_ever_printed(sandbox_up):
    out = tools.diagnose_job(failed_job("__crash_native"))
    assert targets.resolve("sandbox").token() not in out
    assert "SENTINEL-PROMPT" not in out


def test_a_succeeded_job_has_nothing_to_diagnose(sandbox_up):
    job_id = submit("fine")
    tools.wait_job(job_id)
    assert "Nothing to diagnose: the job succeeded" in tools.diagnose_job(job_id)


def test_a_job_still_running_is_not_diagnosed_as_a_failure(sandbox_up):
    job_id = submit("p __slow")
    try:
        tools.wait_job(job_id, until="running")
        out = tools.diagnose_job(job_id)
        assert "Still working" in out and "CAUSE" not in out
    finally:
        tools.run_cancel(job_id)


def test_an_unknown_job_says_so(sandbox_up):
    assert tools.diagnose_job("job_does_not_exist").startswith("ERROR: job job_does_not_exist: HTTP 404")
