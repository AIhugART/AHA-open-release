"""Negative tests from docs/07_security.md section 4. Every security rule has a test that proves the refusal.

Facts: real sockets, real worker subprocesses; the gateway under test is the one `aha-open serve` would start.
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest
from conftest import Api, image_job, new_client_token, start_gateway, tree_bytes

from aha_open import auth as auth_module
from aha_open.auth import Auth
from aha_open.config import Config, ConfigError, ModelConfig, validate_for_serve
from aha_open.errors import ApiError
from aha_open.server import make_server
from aha_open.store import Store
from aha_open.tlsutil import fingerprint, init_tls

# ---------------------------------------------------------------------------------------------- tokens


def test_missing_or_wrong_token_is_401_everywhere(gw):
    anon = Api(gw.base)
    wrong = Api(gw.base, "aho_not-a-real-token")
    for client in (anon, wrong):
        assert client.get("/v1/manifest").status_code == 401
        assert client.post("/v1/jobs", json=image_job()).status_code == 401
        assert client.get("/v1/jobs/job_x").status_code == 401
        assert client.post("/v1/uploads", json={"bytes": 1, "sha256": "0" * 64, "media_type": "image/png"}).status_code == 401
        assert client.get("/v1/jobs/job_x/result").status_code == 401
    body = anon.get("/v1/manifest").json()["error"]
    assert body["code"] == "unauthorized"


def test_a_token_cannot_see_cancel_or_download_another_tokens_job(gw, api):
    job = api.wait(api.job(image_job()).json()["id"])
    other = Api(gw.base, new_client_token(gw, "mallory"))
    for response in (
        other.get(f"/v1/jobs/{job['id']}"),
        other.post(f"/v1/jobs/{job['id']}/cancel"),
        other.get(f"/v1/jobs/{job['id']}/result"),
    ):
        assert response.status_code == 404  # not 403: the job's existence is not revealed
        assert response.json()["error"]["code"] == "not_found"
    assert other.get("/v1/jobs/job_does_not_exist").status_code == 404
    assert api.get(f"/v1/jobs/{job['id']}").status_code == 200  # the owner still can


def test_a_revoked_token_stops_working_on_the_very_next_call(gw, api):
    token = new_client_token(gw, "temp")
    temp = Api(gw.base, token)
    assert temp.get("/v1/manifest").status_code == 200
    store = Store(gw.cfg.data_dir)
    assert store.revoke_client("temp", time.time()) == 1
    store.close()
    assert temp.get("/v1/manifest").status_code == 401


def test_tokens_are_stored_hashed_and_prompts_never_hit_disk_or_logs(gw, api, caplog):
    caplog.set_level("DEBUG")
    needle = "ZEBRA-SECRET-PROMPT-7731"
    job = api.wait(api.job(image_job(f"a photo of {needle}")).json()["id"])
    assert job["status"] == "succeeded"
    cancelled = api.job(image_job(f"{needle} __slow")).json()["id"]
    api.wait(cancelled, states=("running",))
    api.post(f"/v1/jobs/{cancelled}/cancel")
    failed = api.wait(api.job(image_job(f"{needle} __fail")).json()["id"])
    assert failed["status"] == "failed"
    time.sleep(0.3)
    tokens = [gw.tokens["local"], new_client_token(gw, "second")]
    blob = tree_bytes(gw.cfg.data_dir, skip=("local_client.token",))
    for secret in (needle, *tokens):
        assert secret.encode() not in blob, f"{secret[:12]}... was found in the data directory"
        assert secret not in caplog.text, f"{secret[:12]}... was found in the log"


# ---------------------------------------------------------------------------------------------- browser rules


def test_foreign_origin_and_foreign_host_are_403(gw, api):
    evil_origin = api.get("/v1/manifest", headers={"Origin": "https://evil.example"})
    assert evil_origin.status_code == 403 and evil_origin.json()["error"]["code"] == "forbidden"
    assert api.post("/v1/jobs", json=image_job(), headers={"Origin": "https://evil.example"}).status_code == 403
    assert api.get("/v1/manifest", headers={"Origin": "null"}).status_code == 403
    rebinding = Api(gw.base, gw.tokens["local"], host="evil.example")
    assert rebinding.get("/v1/manifest").status_code == 403
    assert httpx.get(f"{gw.base}/v1/health", headers={"Host": "evil.example"}).status_code == 403
    same_origin = api.get("/v1/manifest", headers={"Origin": gw.base})  # the bundled /ui page
    assert same_origin.status_code == 200


def test_operator_can_allow_one_origin_explicitly(make_gateway):
    gw = make_gateway(allowed_origins=["https://studio.example"])
    api = Api(gw.base, gw.tokens["local"])
    assert api.get("/v1/manifest", headers={"Origin": "https://studio.example"}).status_code == 200
    assert api.get("/v1/manifest", headers={"Origin": "https://evil.example"}).status_code == 403


def test_responses_are_not_cacheable(gw, api):
    response = api.get("/v1/manifest")
    assert response.headers["cache-control"] == "no-store" and response.headers["x-content-type-options"] == "nosniff"


# ---------------------------------------------------------------------------------------------- profiles


def test_lan_refuses_wildcard_and_missing_tls_and_tokenless(tmp_path):
    base = Config(data_dir=tmp_path, profile="lan", models=[])
    for listen in ("0.0.0.0:8890", ":8890", "[::]:8890"):
        base.listen = listen
        with pytest.raises(ConfigError, match="interface"):
            validate_for_serve(base)
    base.listen = "127.0.0.1:8890"
    with pytest.raises(ConfigError, match="HTTPS"):
        validate_for_serve(base)
    init_tls(base)
    base.require_token = False
    with pytest.raises(ConfigError, match="token"):
        validate_for_serve(base)
    base.require_token = True
    validate_for_serve(base)
    with pytest.raises(ConfigError, match="interface"):
        make_server(Config(data_dir=tmp_path, profile="lan", listen="0.0.0.0:8890"))


def test_lan_serves_https_with_a_pinnable_fingerprint_and_no_local_token(tmp_path, monkeypatch):
    monkeypatch.setenv("AHA_FAKE_LOAD_S", "0.1")
    cfg = Config(data_dir=tmp_path / "lan", profile="lan", listen="127.0.0.1:0", models=[ModelConfig("example-image", "fake")])
    cert = init_tls(cfg)
    assert len(fingerprint(cert).split(":")) == 32
    running = start_gateway(cfg)
    try:
        assert not cfg.local_token_path.exists()  # rule 12: only the local profile writes it
        assert httpx.get(f"{running.base}/v1/health", verify=str(cert)).status_code == 200
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"{running.base}/v1/health")  # not trusted without the pinned certificate
        no_token = Api(running.base, verify=str(cert))
        assert no_token.get("/v1/manifest").status_code == 401
        token = new_client_token(running, "laptop")
        client = Api(running.base, token, verify=str(cert))
        assert client.get("/v1/manifest").json()["profile"] == "lan"
        assert client.wait(client.job(image_job()).json()["id"])["status"] == "succeeded"
    finally:
        running.stop()


def test_internet_needs_loopback_access_confirmation_and_hosts(tmp_path):
    cfg = Config(data_dir=tmp_path, profile="internet", listen="0.0.0.0:8890")
    with pytest.raises(ConfigError, match="loopback"):
        validate_for_serve(cfg)
    cfg.listen = "127.0.0.1:8890"
    with pytest.raises(ConfigError, match="Cloudflare Access"):
        validate_for_serve(cfg)
    cfg.access_confirmed = True
    with pytest.raises(ConfigError, match="hosts"):
        validate_for_serve(cfg)
    cfg.hosts = ["aha.example.com"]
    validate_for_serve(cfg)
    assert cfg.enroll is False  # pairing is off by default on the internet profile


def test_internet_profile_answers_only_to_its_hostname_and_hosted_models_stay_off(make_gateway):
    gw = make_gateway(
        profile="internet", access_confirmed=True, hosts=["aha.example.com"],
        models=[ModelConfig("example-image", "fake"), ModelConfig("minimax-h3-fl2va", "minimax_h3", variant="mixed")],
    )
    assert not gw.cfg.local_token_path.exists()
    token = new_client_token(gw, "remote")
    remote = Api(gw.base, token, host="aha.example.com")
    models = {m["id"]: m for m in remote.get("/v1/manifest").json()["models"]}
    assert models["minimax-h3-fl2va"]["state"] == "error"  # rule 13: hosted-service duties not accepted
    assert models["minimax-h3-fl2va"]["state_detail"] == "hosted_service_obligations_not_accepted"  # for that reason, not for another
    assert models["example-image"]["state"] == "ready"
    refused = remote.job({"task": "video.generate", "model": "minimax-h3-fl2va", "prompt": "waves", "params": {}})
    assert refused.status_code == 403
    assert Api(gw.base, token, host="127.0.0.1").get("/v1/manifest").status_code == 200  # loopback is the tunnel's own path


def test_hosted_service_models_become_available_only_after_explicit_acceptance(make_gateway):
    gw = make_gateway(
        profile="internet", access_confirmed=True, hosts=["aha.example.com"], hosted_service_obligations="accepted",
        models=[ModelConfig("minimax-h3-fl2va", "minimax_h3", variant="mixed")],
    )
    remote = Api(gw.base, new_client_token(gw, "remote"), host="aha.example.com")
    state = remote.get("/v1/manifest").json()["models"][0]["state"]
    assert state != "error"  # now it is merely not installed on this machine


def test_local_token_file_disappears_when_the_profile_changes(gw, api):
    assert gw.cfg.local_token_path.exists() and api.get("/v1/manifest").status_code == 200
    other = Config(data_dir=gw.cfg.data_dir, profile="lan", listen="127.0.0.1:0")
    store = Store(gw.cfg.data_dir)
    Auth(other, store).sync_local_token()
    store.close()
    assert not gw.cfg.local_token_path.exists()
    assert api.get("/v1/manifest").status_code == 401  # and the old local token is revoked, not just unlinked


# ---------------------------------------------------------------------------------------------- pairing


def test_pairing_is_off_unless_enabled(gw):
    assert httpx.post(f"{gw.base}/v1/enroll", json={"client_name": "x"}).status_code == 404


def test_pairing_hands_the_token_over_exactly_once(make_gateway):
    gw = make_gateway(enroll=True)
    anon = Api(gw.base)
    made = anon.post("/v1/enroll", json={"client_name": "Laptop An", "client_kind": "aha-studio"})
    assert made.status_code == 201
    enroll = made.json()
    assert len(enroll["enroll_code"]) == 7 and enroll["enroll_code"][4] == "-"
    assert anon.get(f"/v1/enroll/{enroll['id']}").json() == {"status": "pending"}
    store = Store(gw.cfg.data_dir)
    Auth(gw.cfg, store).approve_enroll(enroll["enroll_code"].lower())
    first = anon.get(f"/v1/enroll/{enroll['id']}").json()
    assert first["status"] == "approved" and first["token"].startswith("aho_")
    assert "token" not in anon.get(f"/v1/enroll/{enroll['id']}").json()  # a second poll gets nothing
    assert Api(gw.base, first["token"]).get("/v1/manifest").status_code == 200
    with pytest.raises(ApiError, match="no pending"):
        Auth(gw.cfg, store).approve_enroll(enroll["enroll_code"])  # a used code does not mint a second token
    store.close()


def test_pairing_codes_expire_and_cannot_be_approved_or_guessed(make_gateway, monkeypatch):
    monkeypatch.setattr(auth_module, "ENROLL_TTL_S", 1)
    gw = make_gateway(enroll=True)
    anon = Api(gw.base)
    enroll = anon.post("/v1/enroll", json={"client_name": "slow"}).json()
    time.sleep(1.3)
    store = Store(gw.cfg.data_dir)
    with pytest.raises(ApiError, match="no pending"):
        Auth(gw.cfg, store).approve_enroll(enroll["enroll_code"])
    store.close()
    assert anon.get(f"/v1/enroll/{enroll['id']}").json() == {"status": "expired"}
    assert anon.get("/v1/enroll/enr_unknown").status_code == 404


def test_pairing_requests_are_capped(make_gateway):
    gw = make_gateway(enroll=True)
    anon = Api(gw.base)
    codes = [anon.post("/v1/enroll", json={"client_name": f"c{i}"}).status_code for i in range(7)]
    assert codes[:5] == [201] * 5 and codes[5:] == [429, 429]


# ---------------------------------------------------------------------------------------------- invariants


def test_the_gateway_never_imports_torch():
    code = (
        "import sys, aha_open.app, aha_open.cli, aha_open.server, aha_open.scheduler, aha_open.tlsutil; "
        "sys.exit(1 if any(m == 'torch' or m.startswith('torch.') for m in sys.modules) else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, "the gateway imported torch: " + done.stderr
