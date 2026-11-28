"""A video model that declares exact sizes: the accepted params are complete, and the size of the result is checked (docs/product/03_api_v1.md).

Uses the internal ``example-video-sized`` fake model, which has the shape of what MiniMax H3 declares (sizes, a frame grid, steps).
"""

from __future__ import annotations

from aha_open.config import ModelConfig
from conftest import Api

SIZED = [ModelConfig(id="example-video-sized", worker="fake")]


def video_job(prompt: str = "waves", **params) -> dict:
    return {"task": "video.generate", "model": "example-video-sized", "prompt": prompt, "params": {"seed": 1, **params}}


def test_the_manifest_lists_the_exact_sizes_the_frame_grid_and_the_steps(make_gateway):
    gw = make_gateway(models=SIZED)
    api = Api(gw.base, gw.tokens["local"])
    [model] = api.get("/v1/manifest").json()["models"]
    caps = model["capabilities"]
    assert [(s["short_edge"], s["width"], s["height"]) for s in caps["sizes"]] == [(64, 128, 64), (32, 64, 32)]
    assert caps["frame_grid"] == {"step": 4, "base": 1} and caps["steps"] == {"min": 2, "max": 2, "default": 2}


def test_the_job_carries_the_complete_accepted_params_and_the_result_has_exactly_that_size(make_gateway):
    gw = make_gateway(models=SIZED)
    api = Api(gw.base, gw.tokens["local"])
    made = api.job(video_job(short_edge=32))
    assert made.status_code in (200, 201, 202), made.text
    job = api.wait(made.json()["id"])
    assert job["status"] == "succeeded", job
    # The fake worker builds the size it reports from the width/height it was TOLD, so a result of exactly 64x32 proves the gateway sent
    # the complete accepted params for the 32 short edge (a worker that was only told short_edge could not have known them).
    [out] = job["results"]
    assert out["kind"] == "video" and (out["width"], out["height"]) == (64, 32)
    default = api.wait(api.job(video_job()).json()["id"])  # no short_edge: the first declared one
    assert (default["results"][0]["width"], default["results"][0]["height"]) == (128, 64)


def test_a_size_that_was_not_declared_is_refused_with_what_is_offered(make_gateway):
    gw = make_gateway(models=SIZED)
    api = Api(gw.base, gw.tokens["local"])
    for params in ({"short_edge": 48}, {"aspect_ratio": "1:1"}, {"duration_s": 3}, {"steps": 9}, {"width": 64, "height": 32}):
        refused = api.job(video_job(**params))
        assert refused.status_code == 400, (params, refused.text)
        assert refused.json()["error"]["code"] == "invalid_request"


def test_a_worker_that_returns_another_size_than_accepted_is_a_failure_not_a_result(make_gateway):
    gw = make_gateway(models=SIZED)
    api = Api(gw.base, gw.tokens["local"])
    job = api.wait(api.job(video_job("waves __wrongsize")).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "generation_failed"
    assert "was accepted" in job["error"]["message"]
    assert api.get(f"/v1/jobs/{job['id']}/result").status_code == 409  # a wrong-sized video is never offered to the client


def test_a_worker_that_does_not_say_what_size_it_made_cannot_be_checked_and_fails(make_gateway):
    gw = make_gateway(models=SIZED)
    api = Api(gw.base, gw.tokens["local"])
    job = api.wait(api.job(video_job("waves __nosize")).json()["id"])
    assert job["status"] == "failed" and "did not report the size" in job["error"]["message"]
