"""Core /v1 behaviour over real HTTP: manifest, jobs, results, cancel, limits, ordering."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
import time

import httpx
from conftest import Api, image_job, new_client_token

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def test_health_needs_no_token_and_manifest_does(gw, api):
    health = httpx.get(f"{gw.base}/v1/health").json()
    assert health["api_version"] == "1.0" and health["status"] in ("ok", "degraded")
    assert set(health["queue"]) == {"queued", "running"}
    assert httpx.get(f"{gw.base}/v1/manifest").status_code == 401
    manifest = api.get("/v1/manifest").json()
    assert manifest["profile"] == "local" and manifest["optional"] == {"enroll": False}
    ids = {m["id"]: m for m in manifest["models"]}
    assert set(ids) == {"example-image", "example-video"}
    image = ids["example-image"]
    assert image["state"] == "ready" and "image.generate" in image["tasks"]
    assert image["license"]["commercial_use"] == "allowed"
    for key in ("part_size", "max_upload_bytes", "max_queue_per_client", "result_ttl_s"):
        assert key in manifest["limits"]


def test_image_job_end_to_end_with_exact_size_and_range(gw, api):
    created = api.job(image_job(width=320, height=180))
    assert created.status_code == 202
    job = api.wait(created.json()["id"])
    assert job["status"] == "succeeded" and job["error"] is None
    result = job["results"][0]
    assert (result["kind"], result["media_type"], result["width"], result["height"]) == ("image", "image/png", 320, 180)
    full = api.get(f"/v1/jobs/{job['id']}/result")
    assert full.status_code == 200 and full.headers["content-type"] == "image/png"
    assert full.content[:8] == PNG_SIG
    assert struct.unpack(">II", full.content[16:24]) == (320, 180)  # pixels, not just metadata
    assert hashlib.sha256(full.content).hexdigest() == result["sha256"] and len(full.content) == result["bytes"]
    part = api.get(f"/v1/jobs/{job['id']}/result", headers={"Range": "bytes=0-7"})
    assert part.status_code == 206 and part.content == PNG_SIG
    assert api.get(f"/v1/jobs/{job['id']}/result?index=5").status_code == 404


def test_progress_is_reported(gw, api):
    job_id = api.job(image_job(steps=50)).json()["id"]
    while True:
        job = api.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.01)
    assert job["status"] == "succeeded"
    assert job["progress"] == {"step": 50, "total_steps": 50, "fraction": 1.0}


def test_params_outside_capabilities_are_refused_before_queueing(gw, api):
    bad = [
        image_job(width=300, height=300),  # never silently resized
        image_job(width=256),  # width without height
        image_job(steps=999),
        image_job(unknown_param=1),
        {**image_job(), "task": "video.generate"},
        {**image_job(), "prompt": ""},
        {**image_job(), "model": "no-such-model"},
    ]
    for body in bad:
        response = api.job(body)
        assert response.status_code == 400, (body, response.text)
        error = response.json()["error"]
        assert error["code"] in ("invalid_request", "unsupported_task") and error["message"]
    assert api.get("/v1/health").json()["queue"] == {"queued": 0, "running": 0}


def test_worker_output_of_the_wrong_size_is_rejected(gw, api):
    job = api.wait(api.job(image_job("x __wrongsize", width=256, height=256)).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "generation_failed"
    assert "256" in job["error"]["message"]
    assert api.get(f"/v1/jobs/{job['id']}/result").status_code == 409


def test_failure_codes_come_from_the_worker(gw, api):
    oom = api.wait(api.job(image_job("p __fail_oom")).json()["id"])
    assert oom["status"] == "failed" and oom["error"]["code"] == "gpu_out_of_memory"
    crash = api.wait(api.job(image_job("p __crash")).json()["id"])
    assert crash["status"] == "failed" and crash["error"]["code"] == "generation_failed"
    again = api.wait(api.job(image_job("fine")).json()["id"])  # the lane recovered: a new worker is started
    assert again["status"] == "succeeded"
    assert api.get(f"/v1/jobs/{oom['id']}").json()["status"] == "failed"  # nothing re-ran the failed job on its own


def test_a_native_worker_crash_gives_the_operator_something_to_act_on(gw, api):
    """A real crash leaves no Python exception, only an exit status. The error must say what happened and where to look,
    without revealing a server path to the client, and the lane must recover."""
    crash = api.wait(api.job(image_job("p __crash_native")).json()["id"])
    message = crash["error"]["message"]
    assert crash["status"] == "failed" and crash["error"]["code"] == "generation_failed"
    assert message.startswith("while generating:") and "crashed in native code" in message
    assert "worker-fake.log" in message and str(gw.cfg.data_dir) not in message
    assert api.wait(api.job(image_job("fine")).json()["id"])["status"] == "succeeded"


def test_a_native_crash_while_loading_the_model_is_described_too(monkeypatch, make_gateway):
    monkeypatch.setenv("AHA_FAKE_CRASH_ON_LOAD", "native")
    gw = make_gateway()
    api = Api(gw.base, gw.tokens["local"])
    failed = api.wait(api.job(image_job()).json()["id"])
    message = failed["error"]["message"]
    assert failed["status"] == "failed" and message.startswith("while loading the model:")
    assert "crashed in native code" in message and "worker-fake.log" in message


def test_cancel_queued_and_running_and_never_a_result(gw, api):
    running_id = api.job(image_job("p __slow")).json()["id"]
    queued_id = api.job(image_job("later")).json()["id"]
    api.wait(running_id, states=("running",))
    assert api.get(f"/v1/jobs/{queued_id}").json()["queue_position"] == 1
    assert api.post(f"/v1/jobs/{queued_id}/cancel").json()["status"] == "cancelled"
    cancelled_r = api.post(f"/v1/jobs/{running_id}/cancel")
    assert cancelled_r.status_code == 200 and cancelled_r.json()["status"] == "cancelled"
    time.sleep(0.5)  # a late result would have landed by now
    for job_id in (running_id, queued_id):
        job = api.get(f"/v1/jobs/{job_id}").json()
        assert job["status"] == "cancelled" and job["results"] == []
        assert api.get(f"/v1/jobs/{job_id}/result").status_code == 409
    assert api.post(f"/v1/jobs/{running_id}/cancel").json()["status"] == "cancelled"  # idempotent
    assert api.wait(api.job(image_job("after")).json()["id"])["status"] == "succeeded"  # the GPU was given back


def test_a_worker_that_ignores_cancel_is_killed_and_the_job_still_ends_cancelled(make_gateway):
    gw = make_gateway(cancel_grace_s=1)
    api = Api(gw.base, gw.tokens["local"])
    job_id = api.job(image_job("p __slow __ignorecancel")).json()["id"]
    api.wait(job_id, states=("running",))
    assert api.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert api.wait(api.job(image_job("next")).json()["id"])["status"] == "succeeded"


def test_a_result_that_arrives_after_cancel_is_dropped(gw, api):
    # The worker ignores the cancel and finishes anyway: 'done' reaches the gateway for a job the client cancelled.
    created = api.job(image_job("p __ignorecancel", steps=50))
    assert created.status_code == 202, created.text  # a refused job would make this test pass for the wrong reason
    job_id = created.json()["id"]
    api.wait(job_id, states=("running",))
    assert api.post(f"/v1/jobs/{job_id}/cancel").json()["status"] in ("cancelled", "running")
    job = api.wait(job_id, states=("succeeded", "cancelled", "failed"))
    time.sleep(0.8)  # long enough for the worker's late 'done' to be processed
    final = api.get(f"/v1/jobs/{job_id}").json()
    assert job["status"] == final["status"] == "cancelled" and final["results"] == []
    assert api.get(f"/v1/jobs/{job_id}/result").status_code == 409


def test_client_request_id_returns_the_same_job(gw, api):
    body = {**image_job(), "client_request_id": "req-1"}
    first = api.job(body).json()
    second = api.job(body).json()
    assert first["id"] == second["id"]
    assert api.job({**body, "client_request_id": "req-2"}).json()["id"] != first["id"]


def test_queue_limits_answer_429_with_retry_after(make_gateway):
    gw = make_gateway(max_queue_per_client=2)
    api = Api(gw.base, gw.tokens["local"])
    ids = [api.job(image_job("p __slow")).json()["id"] for _ in range(2)]
    refused = api.job(image_job("one more"))
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "busy" and refused.json()["error"]["retry_after_s"] > 0
    assert int(refused.headers["retry-after"]) > 0
    other = Api(gw.base, new_client_token(gw, "second"))  # the limit is per client
    assert other.job(image_job("p __slow")).status_code == 202
    for job_id in ids:
        api.post(f"/v1/jobs/{job_id}/cancel")


def test_rate_limit_per_minute(make_gateway):
    gw = make_gateway(max_jobs_per_minute=3)
    api = Api(gw.base, gw.tokens["local"])
    for _ in range(3):
        assert api.job(image_job()).status_code == 202
    assert api.job(image_job()).status_code == 429


def test_clients_are_served_round_robin_not_first_come_first_served(gw, api):
    other = Api(gw.base, new_client_token(gw, "bee"))
    blocker = api.job(image_job("p __slow")).json()["id"]
    api.wait(blocker, states=("running",))
    a1 = api.job(image_job("a1")).json()["id"]
    b1 = other.job(image_job("b1")).json()["id"]
    api.post(f"/v1/jobs/{blocker}/cancel")
    api.wait(a1)
    other.wait(b1)
    db = sqlite3.connect(str(gw.cfg.data_dir / "jobs.sqlite"))
    finished = dict(db.execute("SELECT id, finished_at FROM jobs WHERE id IN (?,?)", (a1, b1)).fetchall())
    db.close()
    assert finished[b1] < finished[a1], "client B waited behind client A although A had just been served"


def test_model_swap_loads_the_other_model_and_health_follows(gw, api):
    assert api.wait(api.job(image_job()).json()["id"])["status"] == "succeeded"
    video = {"task": "video.generate", "model": "example-video", "prompt": "waves", "params": {"duration_s": 2, "seed": 3}}
    done = api.wait(api.job(video).json()["id"])
    assert done["status"] == "succeeded"
    result = done["results"][0]
    assert result["kind"] == "video" and result["media_type"] == "video/mp4" and result["duration_s"] == 2
    gpus = httpx.get(f"{gw.base}/v1/health").json()["gpus"]
    if gpus:  # this machine has an NVIDIA GPU
        assert gpus[0]["active_model"] == "example-video"


def test_job_dirs_are_cleaned_when_results_expire(make_gateway):
    gw = make_gateway(result_ttl_s=1)
    api = Api(gw.base, gw.tokens["local"])
    job = api.wait(api.job(image_job()).json()["id"])
    assert api.get(f"/v1/jobs/{job['id']}/result").status_code == 200
    time.sleep(1.2)
    gw.server.config.app.state.scheduler.sweep()
    assert api.get(f"/v1/jobs/{job['id']}/result").status_code == 404
    assert not (gw.cfg.data_dir / "jobs" / job["id"]).exists()


def test_unknown_routes_and_methods_use_the_error_shape(gw, api):
    for response in (api.get("/v1/nope"), api.delete("/v1/jobs/x")):
        assert response.status_code in (404, 405)
        assert set(response.json()["error"]) >= {"code", "message"}
