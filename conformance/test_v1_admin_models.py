"""Install a model from the web page (`/admin/models/*`): for the person at this machine, in the `local` profile, and nobody else.

Real HTTP against a real gateway. The install command is swapped for a small script that behaves like the real one
(prints the same phase lines, writes files, exits with a code) so no 18 GB download happens; the child process, the pipe, the
cancel of a whole process tree and the in-place reload of the configuration are all real. Every security rule has a negative test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import FAKE_MODELS, Api, new_client_token

from aha_open import model_install
from aha_open.config import ModelConfig
from aha_open.model_install import InstallManager

KEY = "qwen-research"
FAKE_INSTALLER = '''
import os, subprocess, sys, time
from pathlib import Path
from aha_open.config import Config, ModelConfig, render

mode, data_dir, model = sys.argv[1:4]
data = Path(data_dir)
target = data / "models" / model / "convrot"  # the folder the best variant on this machine installs into
print("Installing int8-convrot: test machine", flush=True)
if mode in ("progress", "progress_early", "progress_fail", "progress_exit"):
    # hf-xet keeps the bytes in its own cache, so the weights folder stays EMPTY while the download runs: only the count the
    # downloader reports can tell how far it got
    if os.environ.get("AHA_OPEN_PROGRESS") != "lines":
        print("error: the gateway did not ask for machine-readable progress", flush=True)
        sys.exit(3)
    print("Downloading repo @ abcd1234 to " + str(target), flush=True)
    target.mkdir(parents=True, exist_ok=True)
    # the downloader's total starts small and grows as it finds files; "early" is that moment, the others are nearly complete
    print("AHAOPEN_PROGRESS 5000 " + ("20000000" if mode == "progress_early" else "18000000000"), flush=True)
    if mode == "progress_fail":
        print("error: something broke", flush=True)
        sys.exit(2)
    if mode == "progress_exit":
        sys.exit(1)  # dies right after a progress line, with no error line of its own
    time.sleep(120)
if mode == "network":
    print("Downloading repo @ abcd1234 to " + str(target), flush=True)
    print("requests.exceptions.ConnectionError: HTTPSConnectionPool: Max retries exceeded with url: /x", flush=True)
    sys.exit(1)
if mode == "crash":
    print("error: something broke in " + str(data), flush=True)
    sys.exit(2)
print("Downloading repo @ abcd1234 to " + str(target), flush=True)
target.mkdir(parents=True, exist_ok=True)
(target / "part.bin").write_bytes(b"x" * 1024 * 1024)
if mode == "hang":
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (data / "grandchild.pid").write_text(str(grandchild.pid))
    time.sleep(120)
print("Creating the qwen_image worker environment in V", flush=True)
cfg = Config(data_dir=data, models=[ModelConfig(id="example-image", worker="fake"), ModelConfig(id="example-video", worker="fake"),
                                    ModelConfig(id=model, worker="qwen_image", variant="int8-convrot")])
(data / "config.toml").write_text(render(cfg), encoding="utf-8")
print("Installed Qwen-Image-2.1 (int8-convrot).", flush=True)
'''


@pytest.fixture
def machine(monkeypatch):
    """A machine that can install: a big GPU, `uv`, the download library and plenty of disk. Tests then break one thing."""
    state = {"gpus": [{"index": 0, "name": "Test GPU", "memory_total_mib": 24564}], "uv": "uv", "download": True, "free": 500.0}
    monkeypatch.setattr(model_install, "detect_gpus", lambda *a, **k: state["gpus"])
    monkeypatch.setattr(model_install, "uv_executable", lambda: state["uv"])
    monkeypatch.setattr(model_install, "has_download_support", lambda: state["download"])
    monkeypatch.setattr(model_install, "free_disk_gib", lambda path: state["free"])
    return state


@pytest.fixture
def fake_install(tmp_path):
    """(installer factory, runs): `runs` lists every command the gateway actually started."""
    script = tmp_path / "fake_install.py"
    script.write_text(FAKE_INSTALLER, encoding="utf-8")
    mode = {"value": "ok"}
    runs: list[list[str]] = []

    def factory(cfg):
        def command(model_id: str, key: str) -> list[str]:
            argv = [sys.executable, str(script), mode["value"], str(cfg.data_dir), model_id]
            runs.append(argv)
            return argv

        return InstallManager(cfg, command=command)

    factory.mode = mode
    return factory, runs


def start_page(make_gateway, fake_install):
    factory, runs = fake_install
    gw = make_gateway(installer=factory)
    return gw, Api(gw.base, gw.tokens["local"]), runs


def install_url(model_id: str = "qwen-image-2.1") -> str:
    return f"/admin/models/{model_id}/install"


def wait_install(api: Api, wanted: tuple[str, ...], timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        data = api.get(install_url()).json()
        if data["state"] in wanted:
            return data
        assert time.monotonic() < deadline, f"install stuck in {data}"
        time.sleep(0.05)


def alive(pid: int) -> bool:
    if sys.platform == "win32":  # os.kill(pid, 0) would KILL the process on Windows
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_grandchild(data_dir: Path) -> int:
    pid_file = data_dir / "grandchild.pid"
    deadline = time.monotonic() + 30
    while not pid_file.exists():
        assert time.monotonic() < deadline, "the install did not start its own child"
        time.sleep(0.1)
    time.sleep(0.2)
    return int(pid_file.read_text())


def wait_gone(pid: int, why: str) -> None:
    deadline = time.monotonic() + 15
    while alive(pid):
        assert time.monotonic() < deadline, why
        time.sleep(0.2)


# ---------------------------------------------------------------------------------------------- the page's view


def test_the_list_offers_qwen_with_its_licence_and_nothing_blocking(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    models = api.get("/admin/models").json()["models"]
    assert [m["id"] for m in models] == ["qwen-image-2.1", "minimax-h3-fl2va", "minimax-h3-ref2va"]
    qwen = models[0]
    assert qwen["state"] == "not_installed" and qwen["blocker"] is None and qwen["disk_gib"] == 17  # INT8 ConvRot: the best variant that fits a 24 GiB GPU
    assert qwen["requires_territory_confirmation"] is False and qwen["license"]["excluded_territories"] == []
    assert qwen["license"]["key"] == KEY and qwen["license"]["commercial_use"] == "not_allowed"
    assert qwen["license"]["name"] and qwen["license"]["url"]
    assert qwen["install"]["state"] == "idle"


def test_the_reason_a_model_is_not_ready_never_spells_out_the_data_folder(make_gateway, fake_install, machine):
    """A worker exists but the weights are not there yet: the catalog says "weights not found at <path>"."""
    factory, _ = fake_install
    mine = ModelConfig(id="qwen-image-2.1", worker="qwen_image", variant="int8", python=sys.executable)
    gw = make_gateway(installer=factory, models=[*FAKE_MODELS, mine])
    listed = Api(gw.base, gw.tokens["local"]).get("/admin/models")
    detail = listed.json()["models"][0]["state_detail"]
    assert detail.startswith("weights not found at ") and "<data folder>" in detail
    assert str(gw.cfg.data_dir) not in listed.text


# ---------------------------------------------------------------------------------------------- the install itself


def test_an_install_runs_in_a_child_process_and_the_manifest_changes_without_a_restart(make_gateway, fake_install, machine):
    gw, api, runs = start_page(make_gateway, fake_install)
    assert [m["id"] for m in api.get("/v1/manifest").json()["models"]] == ["example-image", "example-video"]
    started = api.post(install_url(), json={"accept_license": KEY})
    assert started.status_code == 202, started.text
    done = wait_install(api, ("succeeded", "failed"))
    assert done["state"] == "succeeded" and done["error"] is None and done["finished_at"]
    assert len(runs) == 1
    models = api.get("/v1/manifest").json()["models"]
    qwen = next((m for m in models if m["id"] == "qwen-image-2.1"), None)
    assert qwen is not None  # config.toml was re-read into the running gateway
    assert qwen["state"] == "not_installed"  # the fake wrote no worker environment: the state is computed from the disk, not claimed
    assert str(gw.cfg.data_dir) not in started.text


def test_progress_reports_bytes_that_are_really_on_disk(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "hang"
    assert api.post(install_url(), json={"accept_license": KEY}).status_code == 202
    deadline = time.monotonic() + 30
    while True:
        data = api.get(install_url()).json()
        if data["phase"] == "downloading" and data["downloaded_bytes"] >= 1024 * 1024:
            break
        assert time.monotonic() < deadline, data
        time.sleep(0.2)
    assert data["expected_bytes"] == 17 * 2**30 and data["state"] == "running"
    assert api.delete(install_url()).status_code == 204
    wait_install(api, ("cancelled",))


def test_progress_is_the_count_the_downloader_reports_not_the_size_of_the_folder(make_gateway, fake_install, machine):
    """On a real 18 GB download the folder stayed at 0.03 GiB for minutes while 1.3 GiB had arrived (hf-xet's own cache)."""
    gw, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "progress"
    assert api.post(install_url(), json={"accept_license": KEY}).status_code == 202
    deadline = time.monotonic() + 30
    while True:
        data = api.get(install_url()).json()
        if data["downloaded_bytes"] == 5000:
            break
        assert time.monotonic() < deadline, data
        time.sleep(0.1)
    assert data["phase"] == "downloading" and data["expected_bytes"] == 18_000_000_000  # the downloader's total, once it is near the plan
    assert not any(f.is_file() for f in (gw.cfg.data_dir / "models").rglob("*"))  # the folder holds no file: the number did not come from it
    assert api.delete(install_url()).status_code == 204
    wait_install(api, ("cancelled",))


def test_a_total_that_is_still_small_does_not_make_the_bar_jump_ahead(make_gateway, fake_install, machine):
    """Early on the downloader's total is only the files it has met so far; 5000 of 20 MB would show 0.02% of the wrong thing."""
    _, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "progress_early"
    api.post(install_url(), json={"accept_license": KEY})
    deadline = time.monotonic() + 30
    while True:
        data = api.get(install_url()).json()
        if data["downloaded_bytes"] == 5000:
            break
        assert time.monotonic() < deadline, data
        time.sleep(0.1)
    assert data["expected_bytes"] == 17 * 2**30  # the plan's size, not the early total of 20 MB
    api.delete(install_url())
    wait_install(api, ("cancelled",))


def test_a_progress_line_is_never_part_of_an_error_message(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    for mode, expect in (("progress_fail", "something broke"), ("progress_exit", "stopped")):
        fake_install[0].mode["value"] = mode
        api.post(install_url(), json={"accept_license": KEY})
        done = wait_install(api, ("failed",))
        assert expect in done["error"]["message"] and "AHAOPEN_PROGRESS" not in done["error"]["message"], (mode, done["error"])


def test_cancel_kills_the_whole_process_tree_and_leaves_the_machine_unchanged(make_gateway, fake_install, machine):
    gw, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "hang"
    api.post(install_url(), json={"accept_license": KEY})
    grandchild = wait_for_grandchild(Path(gw.cfg.data_dir))
    assert alive(grandchild)
    assert api.post(install_url(), json={"accept_license": KEY}).status_code == 409  # one install at a time
    assert api.delete(install_url()).status_code == 204
    done = wait_install(api, ("cancelled",))
    assert done["error"] is None
    wait_gone(grandchild, "cancelling left the install's child process running")
    assert [m["id"] for m in api.get("/v1/manifest").json()["models"]] == ["example-image", "example-video"]  # no result adopted
    assert api.delete(install_url()).status_code == 404  # nothing is running any more


def test_a_download_that_cannot_reach_the_internet_says_so_not_a_stack_trace(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "network"
    api.post(install_url(), json={"accept_license": KEY})
    done = wait_install(api, ("failed",))
    assert done["error"]["code"] == "network_unreachable" and "internet" in done["error"]["message"]
    assert "Traceback" not in done["error"]["message"] and "HTTPSConnectionPool" not in done["error"]["message"]


def test_any_other_failure_is_reported_without_the_server_path(make_gateway, fake_install, machine):
    gw, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "crash"
    api.post(install_url(), json={"accept_license": KEY})
    done = wait_install(api, ("failed",))
    assert done["error"]["code"] == "install_failed" and "something broke" in done["error"]["message"]
    assert str(gw.cfg.data_dir) not in done["error"]["message"]  # the data folder is never spelled out


def test_an_install_does_not_survive_the_gateway_stopping(make_gateway, fake_install, machine):
    gw, api, _ = start_page(make_gateway, fake_install)
    fake_install[0].mode["value"] = "hang"
    api.post(install_url(), json={"accept_license": KEY})
    grandchild = wait_for_grandchild(Path(gw.cfg.data_dir))
    gw.stop()
    wait_gone(grandchild, "stopping the gateway left a download running with nobody to report to")


# ---------------------------------------------------------------------------------------------- refused BEFORE anything starts


def _small_gpu(m):
    m.update(gpus=[{"index": 0, "name": "Small", "memory_total_mib": 4096}])


@pytest.mark.parametrize(
    ("break_it", "code"),
    [
        (lambda m: m.update(gpus=[]), "unsupported_hardware"),
        (_small_gpu, "unsupported_hardware"),
        (lambda m: m.update(uv=None), "uv_missing"),
        (lambda m: m.update(download=False), "download_support_missing"),
        (lambda m: m.update(free=3.0), "not_enough_disk"),
    ],
)
def test_each_cause_is_refused_with_its_own_code_and_nothing_is_started(make_gateway, fake_install, machine, break_it, code):
    _, api, runs = start_page(make_gateway, fake_install)
    break_it(machine)
    shown = api.get("/admin/models").json()["models"][0]["blocker"]
    assert shown["code"] == code  # the page learns it before the person presses anything
    refused = api.post(install_url(), json={"accept_license": KEY})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == code
    assert refused.json()["error"]["message"] and not runs  # no child process was ever started


def test_the_causes_give_different_sentences(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    healthy = dict(machine)
    seen = set()
    for break_it in (lambda: machine.update(gpus=[]), lambda: machine.update(uv=None),
                     lambda: machine.update(download=False), lambda: machine.update(free=3.0)):
        machine.update(healthy)
        break_it()
        seen.add(api.get("/admin/models").json()["models"][0]["blocker"]["message"])
    assert len(seen) == 4


# ---------------------------------------------------------------------------------------------- negative tests


def test_the_licence_must_be_accepted_with_the_exact_key(make_gateway, fake_install, machine):
    _, api, runs = start_page(make_gateway, fake_install)
    for body in ({}, {"accept_license": ""}, {"accept_license": True}, {"accept_license": "yes"}, {"accept_license": KEY.upper()}):
        assert api.post(install_url(), json=body).status_code == 400, body
    assert api.post(install_url("example-image"), json={"accept_license": KEY}).status_code == 404  # the fake model has no install
    assert api.post("/admin/models/..%2Fconfig/install", json={"accept_license": KEY}).status_code in (400, 404)
    assert not runs


def test_no_token_or_a_wrong_token_is_401(make_gateway, fake_install, machine):
    gw, _, runs = start_page(make_gateway, fake_install)
    for token in (None, "aho_not-a-real-token"):
        anon = Api(gw.base, token)
        assert anon.get("/admin/models").status_code == 401
        assert anon.post(install_url(), json={"accept_license": KEY}).status_code == 401
        assert anon.get(install_url()).status_code == 401
        assert anon.delete(install_url()).status_code == 401
    assert not runs


def test_a_foreign_host_or_origin_is_403_before_anything_else(make_gateway, fake_install, machine):
    gw, _, runs = start_page(make_gateway, fake_install)
    token = gw.tokens["local"]
    assert Api(gw.base, token, host="evil.example").get("/admin/models").status_code == 403  # DNS rebinding
    hostile = Api(gw.base, token).post(install_url(), headers={"Origin": "https://evil.example"}, json={"accept_license": KEY})
    assert hostile.status_code == 403  # a web page on another site cannot start a download of 18 GB
    assert not runs


def test_the_internet_profile_has_no_install_page_even_through_the_tunnel_path(make_gateway, fake_install, machine):
    factory, runs = fake_install
    gw = make_gateway(profile="internet", access_confirmed=True, hosts=["aha.example.com"], installer=factory, models=FAKE_MODELS)
    token = new_client_token(gw, "remote")
    for host in ("aha.example.com", "127.0.0.1"):
        tunnel = Api(gw.base, token, host=host)
        for method, path in (("GET", "/admin/models"), ("POST", install_url()), ("GET", install_url()), ("DELETE", install_url())):
            sent = tunnel.request(method, path, json={"accept_license": KEY} if method == "POST" else None)
            assert sent.status_code == 404, (host, method, path)
    assert not runs


def use_url(model_id: str = "qwen-image-2.1") -> str:
    return f"/admin/models/{model_id}/use"


def variant_url(variant: str, model_id: str = "qwen-image-2.1") -> str:
    return f"/admin/models/{model_id}/variants/{variant}"


def test_the_list_shows_the_four_qwen_variants_with_cost_fit_and_what_is_installed(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    qwen = next(m for m in api.get("/admin/models").json()["models"] if m["id"] == "qwen-image-2.1")
    names = [v["name"] for v in qwen["variants"]]
    assert names == ["int8", "int8-convrot", "mixed-p010", "mixed-p015"]
    assert [v["name"] for v in qwen["variants"] if v["recommended"]] == ["int8-convrot"]
    assert all(v["fit"] == "fits" and v["installed"] is False and v["disk_gib"] > 0 for v in qwen["variants"]), qwen["variants"]
    assert not {"folder", "files", "runtime"} & set(qwen["variants"][0])


def test_the_variant_routes_need_the_token_the_local_host_and_the_local_profile(make_gateway, fake_install, machine):
    """Negative security tests for choosing and deleting a variant: the same guard as installing."""
    gw, _, runs = start_page(make_gateway, fake_install)
    for token in (None, "aho_not-a-real-token"):
        anon = Api(gw.base, token)
        assert anon.post(use_url(), json={"variant": "int8"}).status_code == 401
        assert anon.delete(variant_url("int8")).status_code == 401
    token = gw.tokens["local"]
    assert Api(gw.base, token, host="evil.example").post(use_url(), json={"variant": "int8"}).status_code == 403  # DNS rebinding
    assert Api(gw.base, token, host="evil.example").delete(variant_url("int8")).status_code == 403
    hostile = {"Origin": "https://evil.example"}
    assert Api(gw.base, token).post(use_url(), headers=hostile, json={"variant": "int8"}).status_code == 403
    assert Api(gw.base, token).delete(variant_url("int8"), headers=hostile).status_code == 403  # a web page on another site cannot delete 18 GB
    assert not runs

    factory, _ = fake_install
    net = make_gateway(profile="internet", access_confirmed=True, hosts=["aha.example.com"], installer=factory, models=FAKE_MODELS)
    remote = Api(net.base, new_client_token(net, "remote"), host="aha.example.com")
    assert remote.post(use_url(), json={"variant": "int8"}).status_code == 404
    assert remote.delete(variant_url("int8")).status_code == 404


def test_choosing_a_variant_that_is_not_offered_or_not_installed_changes_nothing(make_gateway, fake_install, machine):
    gw, api, runs = start_page(make_gateway, fake_install)
    for body in ({}, {"variant": ""}, {"variant": "bf16"}, {"variant": 3}, {"variant": ["int8"]}):
        assert api.post(use_url(), json=body).status_code == 400, body
    missing = api.post(use_url(), json={"variant": "mixed-p015"})
    assert missing.status_code == 409 and missing.json()["error"]["code"] == "variant_not_installed"
    assert api.post(use_url("example-image"), json={"variant": "int8"}).status_code == 404  # the fake model has none
    assert api.delete(variant_url("bf16")).status_code == 400
    assert api.delete(variant_url("mixed-p015")).status_code == 404  # nothing there to delete
    assert api.post(install_url(), json={"accept_license": KEY, "variant": "bf16"}).status_code == 400
    assert not runs


def test_classify_failure_never_includes_the_data_folder():
    data = Path("C:/secret/data")
    got = model_install.classify_failure(2, [f"error: cannot write {data}/x"], data)
    assert str(data) not in got["message"] and got["code"] == "install_failed"
    assert model_install.classify_failure(1, ["...", "getaddrinfo failed"], data)["code"] == "network_unreachable"


# ---------------------------------------------------------------------------------------------- MiniMax H3 on the page


H3 = "minimax-h3-fl2va"
H3_REF = "minimax-h3-ref2va"
H3_KEY = "minimax-h3"
REF_FILE_GIB = 15093774276 / 2**30  # the one file the reference model adds to the text model's folder


def test_the_list_says_which_licences_need_a_territory_confirmation_and_which_territories(make_gateway, fake_install, machine):
    _, api, _ = start_page(make_gateway, fake_install)
    models = {m["id"]: m for m in api.get("/admin/models").json()["models"]}
    for model_id in (H3, H3_REF):
        entry = models[model_id]
        assert entry["requires_territory_confirmation"] is True
        assert entry["license"]["key"] == H3_KEY and entry["license"]["excluded_territories"] == ["EU", "GB", "KR", "US"]
        assert entry["blocker"] is None and entry["state"] == "not_installed"


def test_the_disk_a_model_needs_counts_only_the_files_that_are_not_there_yet(make_gateway, fake_install, machine, monkeypatch):
    """The reference model shares the text model's folder: with the text model installed it needs the one 14 GiB file, not 61 GB."""
    _, api, _ = start_page(make_gateway, fake_install)
    alone = {m["id"]: m["disk_gib"] for m in api.get("/admin/models").json()["models"]}
    assert 44 <= alone[H3] <= 46 and 44 <= alone[H3_REF] <= 46  # nothing on disk: both need the encoder and the VAEs
    real = model_install._missing_gib
    monkeypatch.setattr(model_install, "_missing_gib",
                        lambda parts, target: REF_FILE_GIB if any("Ref2VA" in n for p in parts for n in p["files"]) else 0.0)
    shared = {m["id"]: m["disk_gib"] for m in api.get("/admin/models").json()["models"]}
    assert shared[H3_REF] == 15 and shared[H3] == 1 and real is not model_install._missing_gib


@pytest.mark.parametrize("model_id", [H3, H3_REF])
@pytest.mark.parametrize("body", [
    {"accept_license": H3_KEY},                                  # no confirmation at all
    {"accept_license": H3_KEY, "confirm_territory": False},      # unticked
    {"accept_license": H3_KEY, "confirm_territory": "yes"},      # only the boolean true counts
    {"accept_license": H3_KEY, "confirm_territory": 1},
])
def test_a_licence_that_excludes_territories_is_refused_without_the_confirmation_and_nothing_starts(make_gateway, fake_install, machine, model_id, body):
    _, api, runs = start_page(make_gateway, fake_install)
    refused = api.post(install_url(model_id), json=body)
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "invalid_request"
    assert "territory has not been confirmed" in refused.json()["error"]["message"] and "EU, GB, KR, US" in refused.json()["error"]["message"]
    assert runs == [] and api.get(install_url(model_id)).json()["state"] == "idle"


def test_the_licence_is_still_checked_first_and_a_model_without_territories_needs_no_confirmation(make_gateway, fake_install, machine):
    _, api, runs = start_page(make_gateway, fake_install)
    wrong = api.post(install_url(H3), json={"accept_license": "qwen-research", "confirm_territory": True})
    assert wrong.status_code == 400 and "licence has not been accepted" in wrong.json()["error"]["message"] and runs == []
    assert api.post(install_url(), json={"accept_license": KEY}).status_code == 202  # Qwen: no territory question


def test_a_confirmed_install_passes_the_confirmation_on_to_the_command_that_installs(make_gateway, fake_install, machine, monkeypatch):
    """One way to install: the page runs the same command as the terminal, which asks for --confirm-territory itself."""
    real_popen, started = subprocess.Popen, []

    def spy(argv, *args, **kwargs):
        if any(str(a).endswith("fake_install.py") for a in argv):  # the install command, not the gateway's helpers (icacls, taskkill)
            started.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(model_install.subprocess, "Popen", spy)
    _, api, _ = start_page(make_gateway, fake_install)
    assert api.post(install_url(H3), json={"accept_license": H3_KEY, "confirm_territory": True}).status_code == 202
    assert api.post(install_url(), json={"accept_license": KEY}).status_code == 409  # one install at a time, whichever model
    assert "--confirm-territory" in started[0] and H3 in started[0]
    api.delete(install_url(H3))
    assert api.post(install_url(), json={"accept_license": KEY}).status_code in (202, 409)
    assert all("--confirm-territory" not in argv for argv in started[1:])  # a model without territories never gets the flag


def test_a_gpu_too_small_for_minimax_blocks_it_with_its_own_sentence_and_leaves_qwen_alone(make_gateway, fake_install, machine):
    machine["gpus"] = [{"index": 0, "name": "Small GPU", "memory_total_mib": 12288}]  # 12 GiB: enough for Qwen INT8, not for MiniMax H3
    _, api, runs = start_page(make_gateway, fake_install)
    models = {m["id"]: m for m in api.get("/admin/models").json()["models"]}
    assert models["qwen-image-2.1"]["blocker"] is None
    for model_id, needs in ((H3, "at least 16 GiB"), (H3_REF, "at least 18 GiB")):  # the reference model ran only on bigger cards
        assert models[model_id]["blocker"]["code"] == "unsupported_hardware" and needs in models[model_id]["blocker"]["message"]
        refused = api.post(install_url(model_id), json={"accept_license": H3_KEY, "confirm_territory": True})
        assert refused.status_code == 409 and refused.json()["error"]["code"] == "unsupported_hardware"
    assert runs == []


def test_not_enough_disk_names_the_size_still_to_download_not_the_whole_model(make_gateway, fake_install, machine, monkeypatch):
    monkeypatch.setattr(model_install, "_missing_gib",
                        lambda parts, target: REF_FILE_GIB if any("Ref2VA" in n for p in parts for n in p["files"]) else 0.0)
    machine["free"] = 10.0
    _, api, runs = start_page(make_gateway, fake_install)
    models = {m["id"]: m for m in api.get("/admin/models").json()["models"]}
    assert models[H3_REF]["blocker"]["code"] == "not_enough_disk" and "about 15 GB" in models[H3_REF]["blocker"]["message"]
    machine["free"] = 20.0  # 15 GB x 1.1 fits in 20 GB
    again = {m["id"]: m for m in api.get("/admin/models").json()["models"]}
    assert again[H3_REF]["blocker"] is None
    assert runs == []


@pytest.mark.parametrize("engine", [None, Path(sys.executable)])
def test_the_page_is_told_when_the_model_is_on_disk_and_only_its_engine_is_missing(make_gateway, fake_install, machine, monkeypatch, engine):
    """Root cause: the page offered Download only for weights that are absent, so a machine with weights and no engine had no button."""
    monkeypatch.setattr(model_install, "worker_python", lambda *_: engine)
    monkeypatch.setattr(model_install, "variant_installed", lambda cfg, spec, name: name == "int8-convrot")
    _, api, _ = start_page(make_gateway, fake_install)
    qwen = api.get("/admin/models").json()["models"][0]
    assert qwen["engine_missing"] is (engine is None)


def test_nothing_on_disk_is_a_download_not_a_missing_engine(make_gateway, fake_install, machine, monkeypatch):
    monkeypatch.setattr(model_install, "worker_python", lambda *_: None)
    monkeypatch.setattr(model_install, "variant_installed", lambda *_: False)
    _, api, _ = start_page(make_gateway, fake_install)
    assert [m["engine_missing"] for m in api.get("/admin/models").json()["models"]] == [False, False, False]
