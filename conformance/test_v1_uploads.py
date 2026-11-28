"""Chunked uploads and input handling (docs/product/03_api_v1.md sections 2 and 5)."""

from __future__ import annotations

import hashlib
import os
import time

from conftest import Api, image_job, new_client_token


def _edit(upload_id: str, **params) -> dict:
    return {
        "task": "image.edit", "model": "example-image", "prompt": "make it blue",
        "inputs": [{"role": "source", "upload_id": upload_id}], "params": {"seed": 2, **params},
    }


def test_multipart_upload_feeds_an_edit_job(make_gateway):
    gw = make_gateway(part_size=1024)
    api = Api(gw.base, gw.tokens["local"])
    data = os.urandom(3500)  # 4 parts, the last one short
    assert api.get("/v1/manifest").json()["limits"]["part_size"] == 1024
    upload_id = api.upload(data)
    job = api.wait(api.job(_edit(upload_id, width=256, height=256)).json()["id"])
    assert job["status"] == "succeeded" and (job["results"][0]["width"], job["results"][0]["height"]) == (256, 256)


def test_upload_with_the_wrong_sha256_is_refused_and_discarded(make_gateway):
    gw = make_gateway(part_size=1024)
    api = Api(gw.base, gw.tokens["local"])
    data = os.urandom(2000)
    made = api.post("/v1/uploads", json={"bytes": len(data), "sha256": "0" * 64, "media_type": "image/png"}).json()
    for n, start in enumerate(range(0, len(data), 1024)):
        assert api.put(f"/v1/uploads/{made['upload_id']}/parts/{n}", content=data[start : start + 1024]).status_code == 204
    done = api.post(f"/v1/uploads/{made['upload_id']}/complete")
    assert done.status_code == 400 and done.json()["error"]["code"] == "upload_incomplete"
    assert api.post(f"/v1/uploads/{made['upload_id']}/complete").status_code == 404  # the bad upload is gone


def test_missing_parts_and_wrong_part_sizes_are_refused(make_gateway):
    gw = make_gateway(part_size=1024)
    api = Api(gw.base, gw.tokens["local"])
    data = os.urandom(2500)
    sha = hashlib.sha256(data).hexdigest()
    made = api.post("/v1/uploads", json={"bytes": len(data), "sha256": sha, "media_type": "image/png"}).json()
    uid = made["upload_id"]
    assert api.put(f"/v1/uploads/{uid}/parts/0", content=data[:100]).status_code == 400  # short non-final part
    assert api.put(f"/v1/uploads/{uid}/parts/0", content=data[:2000]).status_code == 400  # over part_size
    assert api.put(f"/v1/uploads/{uid}/parts/9", content=data[:1024]).status_code == 400  # out of range
    assert api.put(f"/v1/uploads/{uid}/parts/0", content=data[:1024]).status_code == 204
    incomplete = api.post(f"/v1/uploads/{uid}/complete")
    assert incomplete.status_code == 400 and incomplete.json()["error"]["code"] == "upload_incomplete"


def test_upload_limits_and_validation(make_gateway):
    gw = make_gateway(max_upload_bytes=5000)
    api = Api(gw.base, gw.tokens["local"])
    ok_sha = "a" * 64
    for body in (
        {"bytes": 6000, "sha256": ok_sha, "media_type": "image/png"},
        {"bytes": 0, "sha256": ok_sha, "media_type": "image/png"},
        {"bytes": 10, "sha256": "XYZ", "media_type": "image/png"},
        {"bytes": 10, "sha256": ok_sha, "media_type": "application/zip"},
    ):
        assert api.post("/v1/uploads", json=body).status_code == 400, body


def test_another_client_cannot_use_or_touch_my_upload(gw, api):
    upload_id = api.upload(b"\x89PNG-ish bytes")
    other = Api(gw.base, new_client_token(gw, "mallory"))
    assert other.put(f"/v1/uploads/{upload_id}/parts/0", content=b"x").status_code == 404
    assert other.post(f"/v1/uploads/{upload_id}/complete").status_code == 404
    refused = other.job(_edit(upload_id, width=256, height=256))
    assert refused.status_code == 404  # the upload does not exist for this token


def test_a_job_cannot_use_an_upload_that_was_never_completed(gw, api):
    data = b"hello world"
    made = api.post("/v1/uploads", json={"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "media_type": "image/png"}).json()
    refused = api.job(_edit(made["upload_id"], width=256, height=256))
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "upload_incomplete"


def test_input_roles_and_media_types_are_checked(gw, api):
    video_upload = api.upload(b"not really a video", "video/mp4")
    image_upload = api.upload(b"image bytes", "image/png")
    assert api.job(_edit(video_upload, width=256, height=256)).status_code == 400  # a video as the source image
    assert api.job({**_edit(image_upload), "inputs": []}).status_code == 400  # edit needs a source
    generate_with_input = api.job({**image_job(), "inputs": [{"role": "source", "upload_id": image_upload}]})
    assert generate_with_input.status_code == 400  # image.generate takes no inputs
    two_sources = api.job({**_edit(image_upload), "inputs": [{"role": "source", "upload_id": image_upload}] * 2})
    assert two_sources.status_code == 400


def test_part_numbers_run_from_zero_to_ceil_minus_one_and_a_part_can_be_resent(make_gateway):
    """docs/product/03_api_v1.md section 2 'Upload rules': numbering, exact sizes, re-sending, complete twice, parts after complete."""
    gw = make_gateway(part_size=1024)
    api = Api(gw.base, gw.tokens["local"])
    data = os.urandom(2500)  # ceil(2500 / 1024) = 3 parts: 1024 + 1024 + 452
    sha = hashlib.sha256(data).hexdigest()
    uid = api.post("/v1/uploads", json={"bytes": len(data), "sha256": sha, "media_type": "image/png"}).json()["upload_id"]
    assert api.put(f"/v1/uploads/{uid}/parts/3", content=data[:452]).status_code == 400  # one past the last number
    assert api.put(f"/v1/uploads/{uid}/parts/-1", content=data[:1024]).status_code in (400, 404, 422)
    assert api.put(f"/v1/uploads/{uid}/parts/2", content=data[2048:]).status_code == 204  # the last part is the remainder
    assert api.put(f"/v1/uploads/{uid}/parts/2", content=data[:1024]).status_code == 400  # ... not a full part
    assert api.put(f"/v1/uploads/{uid}/parts/0", content=b"\0" * 1024).status_code == 204
    assert api.put(f"/v1/uploads/{uid}/parts/0", content=data[:1024]).status_code == 204  # re-sent: the new bytes win
    assert api.put(f"/v1/uploads/{uid}/parts/1", content=data[1024:2048]).status_code == 204
    done = api.post(f"/v1/uploads/{uid}/complete")
    assert done.status_code == 200, done.text  # the sha256 matches only if part 0 was really replaced
    assert api.post(f"/v1/uploads/{uid}/complete").status_code == 200  # again: same answer
    late = api.put(f"/v1/uploads/{uid}/parts/0", content=data[:1024])
    assert late.status_code == 400 and late.json()["error"]["code"] == "invalid_request"  # no parts after complete


def test_a_queued_job_keeps_its_inputs_when_the_upload_expires(make_gateway, monkeypatch):
    """docs/product/03_api_v1.md section 2: the upload id expires, a job already created does not lose what it was given.

    The worker refuses a job whose input file is gone, so `succeeded` proves the file was still there when it ran.
    """
    monkeypatch.setenv("AHA_FAKE_STEP_S", "0.05")  # the blocker (200 steps) must still be running after the sweep
    gw = make_gateway(upload_ttl_s=1)
    api = Api(gw.base, gw.tokens["local"])
    blocker = api.job(image_job("p __slow")).json()["id"]
    api.wait(blocker, states=("running",))
    upload_id = api.upload(b"the source picture")
    queued = api.job(_edit(upload_id, width=256, height=256)).json()["id"]
    time.sleep(1.2)
    gw.server.config.app.state.scheduler.sweep()  # what the janitor does every 20 s
    assert not (gw.cfg.data_dir / "uploads" / upload_id).exists()  # the upload itself did expire
    assert api.job(_edit(upload_id, width=256, height=256)).status_code == 404  # and its id is gone
    assert api.get(f"/v1/jobs/{queued}").json()["status"] == "queued"
    api.post(f"/v1/jobs/{blocker}/cancel")
    done = api.wait(queued)
    assert done["status"] == "succeeded", done["error"]


def test_a_finished_job_keeps_no_copy_of_its_inputs(gw, api):
    needle = b"needle-private-picture-" + os.urandom(16)
    upload_id = api.upload(needle)
    done = api.wait(api.job(_edit(upload_id, width=256, height=256)).json()["id"])
    assert done["status"] == "succeeded"
    jobdir = gw.cfg.data_dir / "jobs" / done["id"]
    deadline = time.monotonic() + 5  # the lane removes the inputs just after it records the result
    # Wait by NAMES only. Opening a file the lane is removing fails here, and an open file cannot be removed there (Windows).
    while any(jobdir.glob("in_*")) or (jobdir / "request.json").exists():
        assert time.monotonic() < deadline, "the job folder still holds the client's input after the job ended"
        time.sleep(0.02)
    assert not any(needle in p.read_bytes() for p in jobdir.iterdir() if p.is_file()), "a copy of the input stayed under another name"


def test_an_upload_that_vanishes_while_the_job_is_created_is_404_and_leaves_no_job_folder(gw, api):
    upload_id = api.upload(b"gone in a moment")
    (gw.cfg.data_dir / "uploads" / upload_id / "data").unlink()  # the row is still there: the shape of the janitor winning the race
    refused = api.job(_edit(upload_id, width=256, height=256))
    assert refused.status_code == 404 and refused.json()["error"]["code"] == "not_found", refused.text
    assert not any((gw.cfg.data_dir / "jobs").iterdir())


def test_inputs_are_copied_where_the_file_system_has_no_hard_links(gw, api, monkeypatch):
    def no_hard_links(*_args, **_kwargs):
        raise OSError("this file system has no hard links")

    monkeypatch.setattr("aha_open.app.os.link", no_hard_links)  # the gateway runs in this process
    upload_id = api.upload(b"a picture on exFAT")
    done = api.wait(api.job(_edit(upload_id, width=256, height=256)).json()["id"])
    assert done["status"] == "succeeded", done["error"]


def test_too_many_stored_uploads_answer_busy_with_retry_after(gw, api):
    body = {"bytes": 10, "sha256": "a" * 64, "media_type": "image/png"}
    answers = [api.post("/v1/uploads", json=body) for _ in range(33)]
    assert [r.status_code for r in answers[:32]] == [201] * 32
    refused = answers[32]
    assert refused.status_code == 429 and refused.json()["error"]["code"] == "busy"
    assert int(refused.headers["retry-after"]) == 30 and refused.json()["error"]["retry_after_s"] == 30
    other = Api(gw.base, new_client_token(gw, "second"))  # the cap is per client
    assert other.post("/v1/uploads", json=body).status_code == 201
