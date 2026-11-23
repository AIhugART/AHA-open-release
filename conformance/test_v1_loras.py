"""LoRA selection over real HTTP with the synthetic worker: clients pick by id, the server resolves the file."""

from __future__ import annotations

import hashlib
import json
import struct

from conftest import Api, image_job, new_client_token

from aha_open import loras


def make_safetensors(path):
    names = ("transformer.blocks.0.attn.to_q.lora_A.weight", "transformer.blocks.0.attn.to_q.lora_B.weight")
    header = {n: {"dtype": "F32", "shape": [2, 2], "data_offsets": [i * 16, (i + 1) * 16]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 32)
    return path


def _install(gw, name: str, **meta):
    folder = loras.lora_dir(gw.cfg)
    folder.mkdir(parents=True, exist_ok=True)
    path = make_safetensors(folder / f"{name}.safetensors")
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _model(api):
    return next(m for m in api.get("/v1/manifest").json()["models"] if m["id"] == "example-image")


def test_manifest_lists_loras_without_a_restart_and_without_paths(gw, api):
    assert _model(api)["capabilities"]["loras"]["available"] == []
    _install(gw, "ink", display_name="Ink", default_scale=0.8, trigger="inkpaint")
    caps = _model(api)["capabilities"]["loras"]
    assert [x["id"] for x in caps["available"]] == ["ink"] and caps["available"][0]["trigger"] == "inkpaint"
    assert caps["max"] == 3 and caps["scale"]["max"] == 2.0
    assert str(gw.cfg.data_dir) not in json.dumps(api.get("/v1/manifest").json())


def test_a_lora_changes_the_result_and_is_recorded_with_its_hash(gw, api):
    path = _install(gw, "ink")
    plain = api.wait(api.job(image_job(width=256, height=256, seed=5)).json()["id"])
    styled_resp = api.job(image_job(width=256, height=256, seed=5, loras=[{"id": "ink", "scale": 0.7}]))
    assert styled_resp.status_code == 202, styled_resp.text
    styled = api.wait(styled_resp.json()["id"])
    assert plain["status"] == styled["status"] == "succeeded"
    assert "loras" not in plain["results"][0]
    assert plain["results"][0]["sha256"] != styled["results"][0]["sha256"]  # the adapter really reached the worker
    used = styled["results"][0]["loras"]
    assert used == [{"id": "ink", "scale": 0.7, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    assert str(path) not in json.dumps(styled)  # never the server's path
    again = api.wait(api.job(image_job(width=256, height=256, seed=5, loras=[{"id": "ink", "scale": 0.7}])).json()["id"])
    assert again["results"][0]["sha256"] == styled["results"][0]["sha256"]  # same job, same picture
    after = api.wait(api.job(image_job(width=256, height=256, seed=5)).json()["id"])
    assert after["results"][0]["sha256"] == plain["results"][0]["sha256"]  # nothing left over from the LoRA job


def test_unknown_or_malformed_loras_are_refused_before_anything_is_queued(gw, api):
    _install(gw, "ink")
    bodies = [
        [{"id": "nope"}], [{"id": "ink", "scale": 5}], [{"id": "ink"}, {"id": "ink"}], [{"id": "../ink"}],
        [{"id": "ink", "path": "C:/Windows/system.ini"}], "ink",
    ]
    for loras_param in bodies:
        response = api.job(image_job(width=256, height=256, loras=loras_param))
        assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request", loras_param
    assert api.get("/v1/health").json()["queue"] == {"queued": 0, "running": 0}


def test_a_lora_removed_after_the_job_was_accepted_fails_the_job_cleanly(gw, api):
    path = _install(gw, "ink")
    blocker = api.job(image_job("p __slow")).json()["id"]
    api.wait(blocker, states=("running",))
    queued = api.job(image_job(width=256, height=256, loras=[{"id": "ink"}])).json()["id"]
    path.unlink()  # the operator deletes it while the job waits
    api.post(f"/v1/jobs/{blocker}/cancel")
    job = api.wait(queued)
    assert job["status"] == "failed" and "LoRA" in job["error"]["message"]


def test_a_tampered_safetensors_header_is_not_offered(gw, api):
    folder = loras.lora_dir(gw.cfg)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evil.safetensors").write_bytes(struct.pack("<Q", 2**40) + b"{}")
    assert _model(api)["capabilities"]["loras"]["available"] == []
    assert api.job(image_job(width=256, height=256, loras=[{"id": "evil"}])).status_code == 400


def test_other_clients_see_the_same_server_wide_list_but_choose_independently(gw, api):
    _install(gw, "ink")
    _install(gw, "paper")
    other = Api(gw.base, new_client_token(gw, "second"))
    assert [x["id"] for x in _model(other)["capabilities"]["loras"]["available"]] == ["ink", "paper"]
    a = api.wait(api.job(image_job(width=256, height=256, seed=1, loras=[{"id": "ink"}])).json()["id"])
    b = other.wait(other.job(image_job(width=256, height=256, seed=1, loras=[{"id": "paper"}])).json()["id"])
    assert a["results"][0]["sha256"] != b["results"][0]["sha256"]
