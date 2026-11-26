"""Every shipped scenario runs against a real sandbox gateway and passes; the safety rules of the active tools hold."""

from __future__ import annotations

import json

import pytest
from aha_open.config import Config, ModelConfig

from aha_open_mcp import jobs, scenarios, targets, tools
from aha_open_mcp.targets import Target, TargetError

NEEDLE = "SENTINEL-PROMPT-91c2f7 a quiet harbour"


def json_of(out: str) -> dict:
    return json.loads(out[out.index("{"):])


FAKE_SCENARIOS = sorted(k for k, v in scenarios.load_all()[0].items() if v.requires_worker == "fake" and "sized_video_sandbox" not in v.preconditions)
SIZED_SCENARIOS = sorted(k for k, v in scenarios.load_all()[0].items() if "sized_video_sandbox" in v.preconditions)


@pytest.fixture
def sized_sandbox_up():
    """Function scope on purpose: a module-scoped one would still be up when the default `sandbox_up` of the next tests starts."""
    from aha_open_mcp import sandbox

    sandbox.stop()  # one sandbox at a time
    info = sandbox.start(sized_video=True)
    try:
        yield info
    finally:
        sandbox.stop()


def test_there_are_scenarios_that_need_the_sized_video_sandbox():
    assert SIZED_SCENARIOS == ["cancel_running_video", "video_size_mismatch_rejected"]


@pytest.mark.parametrize("scenario_id", SIZED_SCENARIOS)
def test_each_video_scenario_passes_on_a_sandbox_with_the_sized_video_model(sized_sandbox_up, scenario_id):
    out = tools.run_scenario(scenario_id, record=False)
    assert "VERDICT: PASS" in out and "evidence 🧪 (fake worker)" in out, out


@pytest.mark.parametrize("scenario_id", FAKE_SCENARIOS)
def test_each_shipped_scenario_passes_on_the_sandbox(sandbox_up, scenario_id):
    out = tools.run_scenario(scenario_id, record=False)
    assert out.startswith("SIDE EFFECTS:"), out  # said before it acts
    assert "VERDICT: PASS" in out and "evidence 🧪 (fake worker)" in out, out
    assert "harness: fresh | gateway: fresh" in out


def test_a_scenario_that_cannot_pass_says_fail_with_the_failing_check(sandbox_up):
    out = tools.run_scenario("generate_exact_size", overrides_json=json.dumps({"prompt": "x __wrongsize"}), record=False)
    assert "VERDICT: FAIL" in out and "FAIL  job.status eq 'succeeded'  (actual 'failed')" in out


def test_an_override_for_a_variable_the_scenario_does_not_have_is_refused(sandbox_up):
    out = tools.run_scenario("generate_exact_size", overrides_json='{"nope": 1}', record=False)
    assert out.startswith("ERROR: unknown override(s) ['nope']")


def test_an_unknown_scenario_lists_what_exists(sandbox_up):
    assert tools.run_scenario("nothing_here").startswith("ERROR: no scenario 'nothing_here'; have [")


def test_a_scenario_whose_job_is_refused_stops_and_does_not_invent_a_job(sandbox_up):
    out = tools.run_scenario("params_refused_before_queueing", record=False)
    assert "refused: invalid_request" in out and "VERDICT: PASS" in out


def test_the_prompt_never_appears_in_any_output_or_in_the_recorded_files(sandbox_up):
    body = json.dumps({"task": "image.generate", "model": "example-image", "prompt": NEEDLE, "params": {"seed": 1}})
    started = tools.run_job(body)
    job_id = json_of(started)["job_id"]
    outputs = [started, tools.wait_job(job_id), tools.assert_result_image(job_id), tools.full_diagnostics()]
    outputs.append(tools.run_scenario("generate_exact_size", overrides_json=json.dumps({"prompt": NEEDLE})))
    for out in outputs:
        assert "SENTINEL-PROMPT" not in out
    recorded = "".join(p.read_text(encoding="utf-8") for p in targets.STATE_DIR.rglob("*.json"))
    assert "SENTINEL-PROMPT" not in recorded
    record = next((targets.STATE_DIR / "ledger" / "generate_exact_size").glob("*.json"))
    assert json.loads(record.read_text("utf-8"))["prompt_sha256_12"]


def test_assert_result_image_judges_the_pixels(sandbox_up):
    body = json.dumps({"task": "image.generate", "model": "example-image", "prompt": "square", "params": {"seed": 3, "width": 256, "height": 256}})
    job_id = json_of(tools.run_job(body))["job_id"]
    tools.wait_job(job_id)
    out = tools.assert_result_image(job_id)
    assert "\nPASS\n" in out and '"ihdr_matches": true' in out


def test_a_real_model_target_is_refused_without_allow_gpu_and_names_why():
    cfg = Config(data_dir=targets.STATE_DIR, listen="127.0.0.1:1", models=[ModelConfig(id="qwen-image-2.1", worker="qwen_image")])
    t = Target("live", cfg, "http://127.0.0.1:1")
    with pytest.raises(TargetError, match="17.6 GiB.*allow_gpu=true"):
        scenarios.gpu_gate(t, False)
    scenarios.gpu_gate(t, True)  # explicit consent passes


def test_a_job_the_server_refused_is_not_retried_by_the_tool(sandbox_up):
    t = targets.resolve("sandbox")
    before = jobs.submit(t, {"task": "image.generate", "model": "example-image", "prompt": "p", "params": {"width": 300, "height": 300}})
    assert before["http_status"] == 400 and "job_id" not in before
    assert "HTTP 400" in tools.run_scenario("params_refused_before_queueing", record=False)
