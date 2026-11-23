"""Chunked uploads and input handling (docs/03_api_v1.md sections 2 and 5)."""

from __future__ import annotations

import hashlib
import os

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
