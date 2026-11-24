"""The real-model and `live` paths, tested WITHOUT loading a model: the gates that decide whether a real model may be loaded,
the target resolution, and the sandbox UI address. (Loading the real model is M5's recorded run, not a unit test.)"""

from __future__ import annotations

import json

import httpx
import pytest
from aha_open.config import Config, ModelConfig

from aha_open_mcp import ram, sandbox, scenarios, targets, tools
from aha_open_mcp.targets import Target, TargetError

REAL = Config(data_dir=targets.STATE_DIR, listen="127.0.0.1:1", models=[ModelConfig(id="qwen-image-2.1", worker="qwen_image")])
FAKE = Config(data_dir=targets.STATE_DIR, listen="127.0.0.1:1", models=[ModelConfig(id="example-image", worker="fake")])


def target_of(cfg: Config) -> Target:
    return Target("sandbox", cfg, "http://127.0.0.1:1")  # nothing listens there: model_resident() must answer False, not crash


def test_free_ram_is_measured_on_this_machine():
    free = ram.free_ram_gib()
    assert free is not None and free > 0.1


def test_the_ram_gate_refuses_a_real_model_when_too_little_is_free_and_says_the_numbers():
    with pytest.raises(TargetError, match=r"free RAM is 5\.0 GiB but loading the real model needs about 20 GiB.*allow_low_ram=true"):
        ram.gate(target_of(REAL), False, free=5.0)


def test_the_ram_gate_passes_with_enough_free_ram_or_explicit_consent_or_a_resident_model(monkeypatch):
    ram.gate(target_of(REAL), False, free=25.0)
    ram.gate(target_of(REAL), True, free=1.0)  # the person accepted the risk
    monkeypatch.setattr(ram, "model_resident", lambda t: True)  # already loaded: low free RAM is the model itself
    ram.gate(target_of(REAL), False, free=1.0)


def test_the_ram_gate_never_blocks_the_fake_worker():
    ram.gate(target_of(FAKE), False, free=0.5)


def test_an_unmeasurable_ram_is_refused_not_assumed_fine(monkeypatch):
    monkeypatch.setattr(ram, "free_ram_gib", lambda: None)
    with pytest.raises(TargetError, match="cannot measure free RAM"):
        ram.gate(target_of(REAL), False)


def test_a_scenario_for_the_fake_worker_is_refused_on_a_real_target_and_the_reverse():
    ok, _ = scenarios.load_all()
    with pytest.raises(TargetError, match="needs worker 'fake'"):
        scenarios.worker_gate(ok["cancel_running_job"], target_of(REAL))
    with pytest.raises(TargetError, match="needs worker 'qwen_image'"):
        scenarios.worker_gate(ok["real_generate_1k_seed_42"], target_of(FAKE))
    scenarios.worker_gate(ok["real_generate_1k_seed_42"], target_of(REAL))


def test_starting_a_real_sandbox_needs_allow_gpu_and_enough_ram(stopped, monkeypatch):
    assert "allow_gpu=true" in tools.run_sandbox_start(real_model=True)
    monkeypatch.setattr(ram, "free_ram_gib", lambda: 6.5)
    refused = tools.run_sandbox_start(real_model=True, allow_gpu=True)
    assert refused.startswith("ERROR: free RAM is 6.5 GiB but the real model needs about 20 GiB")
    assert not (targets.STATE_DIR / targets.SANDBOX_FILE).exists()  # nothing was started or left behind


def test_a_missing_weights_folder_is_named_and_leaves_nothing(stopped, tmp_path):
    out = tools.run_sandbox_start(real_model=True, weights_dir=str(tmp_path / "nowhere"), allow_gpu=True, allow_low_ram=True)
    assert out.startswith("ERROR: no Qwen INT8 weights at ")
    assert not (targets.STATE_DIR / targets.SANDBOX_FILE).exists()


def test_a_real_sandbox_starts_with_the_real_worker_configured_and_loads_nothing_until_a_job(stopped):
    try:
        models = sandbox.real_models()
    except TargetError as err:
        pytest.skip(str(err))
    out = tools.run_sandbox_start(real_model=True, allow_gpu=True, allow_low_ram=True)  # no job is submitted: no model is loaded
    try:
        assert "evidence: ✅ (real model)" in out and "worker: qwen_image" in out
        manifest = json.loads(tools.probe_v1("/v1/manifest").split("\n\n", 1)[1])["manifest"]
        assert [m["id"] for m in manifest["models"]] == [models[0].id]
        assert "worker started" not in tools.read_gateway_log("sandbox", 200)  # the worker process does not exist yet
    finally:
        tools.run_sandbox_stop()


def test_live_resolves_from_the_operators_data_directory_and_reads_it_like_the_sandbox(sandbox_up, monkeypatch):
    monkeypatch.setenv("AHA_OPEN_DATA_DIR", str(targets.resolve("sandbox").data_dir))  # stand in for an operator's server
    out = tools.check_health("live")
    assert "| target: live |" in out.splitlines()[0] and '"status": "ok"' in out
    assert "gateway.log" in tools.read_gateway_log("live", 5)


def test_live_without_a_configuration_says_what_to_run(monkeypatch, tmp_path):
    monkeypatch.setenv("AHA_OPEN_DATA_DIR", str(tmp_path))
    assert "run `aha-open init` first" in tools.check_health("live")


def test_the_ui_address_carries_the_sandbox_token_and_dies_with_the_sandbox(stopped):
    tools.run_sandbox_start()
    out = tools.run_ui_url()
    address = out.rsplit("\n\n", 1)[1]
    base, token = address.split("/ui/#token=")
    base = base.strip()
    assert out.startswith("SIDE EFFECTS:") and base.startswith("http://127.0.0.1:")
    assert httpx.get(base + "/v1/manifest", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert httpx.get(base + "/ui/").status_code == 200
    tools.run_sandbox_stop()
    with pytest.raises(httpx.HTTPError):
        httpx.get(base + "/v1/health", timeout=2)


def test_the_ui_address_exists_only_for_a_running_sandbox(stopped):
    assert tools.run_ui_url().startswith("ERROR: target=sandbox: no sandbox is running")


def test_model_resident_reads_the_real_server_before_and_after_a_model_is_loaded(stopped):
    """The gate must not block the next job of a model that is already loaded (its RAM is why free RAM is low)."""
    tools.run_sandbox_start()
    try:
        t = targets.resolve("sandbox")
        assert ram.model_resident(t) is False
        body = {"task": "image.generate", "model": "example-image", "prompt": "x", "params": {"seed": 1}}
        started = tools.run_job(json.dumps(body))
        tools.wait_job(json.loads(started[started.index("{"):])["job_id"])
        assert ram.model_resident(t) is True
    finally:
        tools.run_sandbox_stop()
