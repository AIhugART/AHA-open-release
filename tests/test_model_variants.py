"""Choosing, installing and deleting variants from the web page, and what the scheduler does about it (plan Q3, D9).

No GPU, no network, no real worker: files are made on the fly and the worker and the store are small fakes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aha_open import model_install
from aha_open import scheduler as scheduler_module
from aha_open.catalog import CATALOG, weights_dir
from aha_open.config import Config, ModelConfig, load
from aha_open.errors import ApiError
from aha_open.model_install import InstallManager, classify_failure
from aha_open.scheduler import Entry, Lane, Scheduler

MODEL = "qwen-image-2.1"
QWEN = CATALOG[MODEL]


def _cfg(tmp_path: Path, variant: str = "") -> Config:
    cfg = Config(data_dir=tmp_path, models=[ModelConfig(MODEL, "qwen_image", variant=variant, python=sys.executable)])
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    return cfg


def _place(cfg: Config, variant: str) -> Path:
    folder = weights_dir(cfg, ModelConfig(MODEL, "qwen_image", variant=variant))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_index.json").write_text("{}", encoding="utf-8")
    for name in QWEN.variants[variant]["files"]:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 64)
    (folder / "shared.bin").write_bytes(b"y" * 64)
    (folder / "install.json").write_text(json.dumps({"variants": [variant]}), encoding="utf-8")
    return folder


class FakeScheduler:
    def __init__(self, busy: bool = False, on_release=None) -> None:
        self._busy, self.released, self._on_release = busy, [], on_release

    def busy(self, model_id: str) -> bool:
        return self._busy

    def release(self, model_id: str) -> bool:
        self.released.append(model_id)
        if self._on_release:
            self._on_release()
        return True


@pytest.fixture(autouse=True)
def machine(monkeypatch):
    gpus = [{"index": 0, "name": "g", "memory_total_mib": 24564}]
    monkeypatch.setattr(model_install, "detect_gpus", lambda *a, **k: gpus)
    monkeypatch.setattr("aha_open.catalog.detect_gpus", lambda *a, **k: gpus)
    monkeypatch.setattr(model_install, "uv_executable", lambda: "uv")
    monkeypatch.setattr(model_install, "has_download_support", lambda: True)
    monkeypatch.setattr(model_install, "free_disk_gib", lambda path: 500.0)


def _manager(cfg: Config, scheduler=None) -> InstallManager:
    manager = InstallManager(cfg, command=lambda model_id, key: [sys.executable, "-c", "pass"])
    if scheduler is not None:
        manager.attach_scheduler(scheduler)
    return manager


# ---------------------------------------------------------------------------------------------- what the page shows


def test_the_page_lists_every_variant_with_what_it_costs_and_what_is_on_disk(tmp_path):
    cfg = _cfg(tmp_path, "mixed-p015")
    _place(cfg, "mixed-p015")
    entry = next(m for m in _manager(cfg).models() if m["id"] == MODEL)
    by_name = {v["name"]: v for v in entry["variants"]}
    assert list(by_name) == list(QWEN.variants)
    assert by_name["mixed-p015"]["installed"] and by_name["mixed-p015"]["selected"] and by_name["mixed-p015"]["download_gib"] == 0
    assert not by_name["int8-convrot"]["installed"] and by_name["int8-convrot"]["download_gib"] >= 1
    assert [n for n, v in by_name.items() if v["recommended"]] == ["int8-convrot"]
    assert all(v["fit"] == "fits" and v["quality_note"] and v["disk_gib"] for v in by_name.values())
    assert not {"folder", "files", "runtime"} & set(by_name["int8"])
    # what a delete frees: a ConvRot build's own transformer file (the text encoder and VAE are shared), the whole folder for int8
    assert by_name["int8"]["own_gib"] == 18.0
    assert by_name["int8-convrot"]["own_gib"] == 6.8 and by_name["mixed-p010"]["own_gib"] == 6.1 and by_name["mixed-p015"]["own_gib"] == 4.3
    assert all(by_name[n]["own_gib"] < by_name[n]["disk_gib"] for n in ("int8-convrot", "mixed-p010", "mixed-p015"))


# ---------------------------------------------------------------------------------------------- install a chosen variant


def _capture_command(monkeypatch) -> list[list[str]]:
    seen: list[list[str]] = []
    real = subprocess.Popen

    def popen(command, *args, **kwargs):
        seen.append(list(command))
        return real(command, *args, **kwargs)

    monkeypatch.setattr(model_install.subprocess, "Popen", popen)
    return seen


def test_a_chosen_variant_travels_to_the_installer_and_none_chosen_sends_nothing(tmp_path, monkeypatch):
    seen = _capture_command(monkeypatch)
    manager = _manager(_cfg(tmp_path))
    manager.start(MODEL, "qwen-research", False, "mixed-p015")
    assert seen[-1][-2:] == ["--variant", "mixed-p015"]
    manager._run.proc.wait(timeout=30)
    manager._run.state = "succeeded"
    manager.start(MODEL, "qwen-research")
    assert "--variant" not in seen[-1]  # the installer picks the best one for this machine, as the command line does


@pytest.mark.parametrize("bad", ["bf16", "", 3, ["int8"], {"a": 1}])
def test_a_variant_that_is_not_offered_is_a_400_and_nothing_starts(tmp_path, monkeypatch, bad):
    seen = _capture_command(monkeypatch)
    with pytest.raises(ApiError) as caught:
        _manager(_cfg(tmp_path)).start(MODEL, "qwen-research", False, bad)
    assert caught.value.status == 400 and not seen


def test_a_variant_that_is_on_disk_is_not_downloaded_again_but_a_second_one_is_allowed(tmp_path, monkeypatch):
    seen = _capture_command(monkeypatch)
    cfg = _cfg(tmp_path, "int8-convrot")
    _place(cfg, "int8-convrot")
    manager = _manager(cfg)
    with pytest.raises(ApiError) as caught:
        manager.start(MODEL, "qwen-research", False, "int8-convrot")
    assert caught.value.status == 409 and caught.value.code == "already_installed" and not seen
    manager.start(MODEL, "qwen-research", False, "mixed-p010")  # another variant: allowed while the first one works
    assert seen[-1][-2:] == ["--variant", "mixed-p010"]
    manager._run.proc.wait(timeout=30)


# ---------------------------------------------------------------------------------------------- use


def test_use_selects_an_installed_variant_and_releases_the_worker_that_holds_the_old_one(tmp_path):
    cfg = _cfg(tmp_path, "int8-convrot")
    _place(cfg, "int8-convrot")
    _place(cfg, "mixed-p015")
    fake = FakeScheduler()
    entry = _manager(cfg, fake).use(MODEL, "mixed-p015")
    assert cfg.models[0].variant == "mixed-p015" and load(tmp_path).model(MODEL).variant == "mixed-p015"
    assert fake.released == [MODEL] and any(v["selected"] and v["name"] == "mixed-p015" for v in entry["variants"])


def test_use_is_refused_while_a_job_of_the_model_waits_or_runs(tmp_path):
    cfg = _cfg(tmp_path, "int8-convrot")
    _place(cfg, "int8-convrot")
    _place(cfg, "mixed-p015")
    fake = FakeScheduler(busy=True)
    with pytest.raises(ApiError) as caught:
        _manager(cfg, fake).use(MODEL, "mixed-p015")
    assert caught.value.status == 409 and caught.value.code == "model_busy"
    assert cfg.models[0].variant == "int8-convrot" and not fake.released  # nothing changed, nothing stopped


def test_use_is_refused_while_an_install_runs_and_for_a_variant_that_is_not_installed(tmp_path):
    cfg = _cfg(tmp_path, "int8-convrot")
    _place(cfg, "int8-convrot")
    manager = _manager(cfg, FakeScheduler())
    with pytest.raises(ApiError) as caught:
        manager.use(MODEL, "mixed-p015")
    assert caught.value.status == 409 and caught.value.code == "variant_not_installed"
    with pytest.raises(ApiError) as caught:
        manager.use(MODEL, "bf16")
    assert caught.value.status == 400
    manager._run = model_install._Run(MODEL, tmp_path, 1)  # an install is running
    with pytest.raises(ApiError) as caught:
        manager.use(MODEL, "int8-convrot")
    assert caught.value.code == "install_running"


# ---------------------------------------------------------------------------------------------- remove


def test_remove_releases_the_worker_first_then_deletes(tmp_path):
    cfg = _cfg(tmp_path, "int8-convrot")
    folder = _place(cfg, "mixed-p015")
    target = folder / QWEN.variants["mixed-p015"]["files"][0]
    seen = {}
    fake = FakeScheduler(on_release=lambda: seen.setdefault("file_still_there_at_release", target.exists()))
    result = _manager(cfg, fake).remove(MODEL, "mixed-p015")
    assert seen["file_still_there_at_release"] is True and not target.exists()  # a loaded worker holds the file open: stop it BEFORE deleting
    assert "removed mixed-p015" in result["message"] and str(tmp_path) not in result["message"]


def test_remove_is_refused_while_a_job_of_the_model_waits_or_runs_and_deletes_nothing(tmp_path):
    cfg = _cfg(tmp_path, "mixed-p015")
    folder = _place(cfg, "mixed-p015")
    with pytest.raises(ApiError) as caught:
        _manager(cfg, FakeScheduler(busy=True)).remove(MODEL, "mixed-p015")
    assert caught.value.status == 409 and caught.value.code == "model_busy"
    assert (folder / QWEN.variants["mixed-p015"]["files"][0]).exists()


def test_remove_of_a_variant_that_is_not_there_is_a_404_and_an_unknown_variant_a_400(tmp_path):
    manager = _manager(_cfg(tmp_path), FakeScheduler())
    with pytest.raises(ApiError) as caught:
        manager.remove(MODEL, "mixed-p015")
    assert caught.value.status == 404
    with pytest.raises(ApiError) as caught:
        manager.remove(MODEL, "../../etc")
    assert caught.value.status == 400  # never a path: only names the catalog knows


def test_the_shared_files_stay_unless_asked_for(tmp_path):
    cfg = _cfg(tmp_path)
    folder = _place(cfg, "mixed-p015")
    result = _manager(cfg, FakeScheduler()).remove(MODEL, "mixed-p015")
    assert (folder / "shared.bin").exists() and result["shared_offer"] is True
    assert "--include-shared" not in result["message"] and "were kept" in result["message"]  # the page offers a button, not a flag


# ---------------------------------------------------------------------------------------------- failure wording


def test_an_untrusted_certificate_is_not_reported_as_no_internet():
    seen = ["httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1032)"]
    out = classify_failure(1, seen, Path("C:/data"))
    assert out["code"] == "certificate_untrusted" and "SSL_CERT_FILE" in out["message"]


def test_no_internet_is_still_no_internet():
    out = classify_failure(1, ["httpx.ConnectError: [Errno 11001] getaddrinfo failed"], Path("C:/data"))
    assert out["code"] == "network_unreachable"


# ---------------------------------------------------------------------------------------------- scheduler


class FakeWorker:
    def __init__(self, *a, **k) -> None:
        self.proc = type("P", (), {"pid": 1})()
        self.sent: list[dict] = []
        self.stopped = False

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def next_event(self, timeout: float):
        return {"event": "loaded"}

    def alive(self) -> bool:
        return True

    def stop(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        self.stopped = True


def _scheduler(tmp_path: Path, variant: str) -> tuple[Scheduler, Config, Lane]:
    cfg = _cfg(tmp_path, variant)
    sched = Scheduler(cfg, store=None)  # busy/release/load never touch the store
    lane = Lane(0)
    sched.lanes[0] = lane
    return sched, cfg, lane


def test_the_load_message_names_the_variants_own_file(tmp_path, monkeypatch):
    sched, cfg, lane = _scheduler(tmp_path, "mixed-p010")
    monkeypatch.setattr(scheduler_module, "WorkerProc", FakeWorker)
    monkeypatch.setattr(scheduler_module, "worker_command", lambda cfg, mc: ["python", "worker.py"])
    sched._ensure_model(lane, cfg.models[0], QWEN)
    message = lane.worker.sent[0]
    assert message["op"] == "load" and message["variant"] == "mixed-p010"
    assert message["files"] == list(QWEN.variants["mixed-p010"]["files"])
    assert lane.model == MODEL and lane.variant == "mixed-p010"


def test_the_bitsandbytes_variant_sends_no_file(tmp_path, monkeypatch):
    sched, cfg, lane = _scheduler(tmp_path, "int8")
    monkeypatch.setattr(scheduler_module, "WorkerProc", FakeWorker)
    monkeypatch.setattr(scheduler_module, "worker_command", lambda cfg, mc: ["python", "worker.py"])
    sched._ensure_model(lane, cfg.models[0], QWEN)
    assert lane.worker.sent[0]["files"] == []


class FakeStore:
    def job(self, job_id):
        return {"status": "queued", "task": "image.generate"}

    def advance_job(self, *a, **k):
        return True


def test_another_variant_of_the_resident_model_is_another_worker(tmp_path, monkeypatch):
    """The model id alone used to decide residency: switching the variant in the configuration would have kept serving the old one."""
    sched, _, lane = _scheduler(tmp_path, "mixed-p015")
    sched.store = FakeStore()
    lane.worker, lane.model, lane.variant = FakeWorker(), MODEL, "int8-convrot"
    job_dir = sched.job_dir("j1")
    job_dir.mkdir(parents=True)
    request = json.dumps({"params": {}, "loras": []})
    (job_dir / "request.json").write_text(request, encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(sched, "_ensure_model", lambda lane, mc, spec: calls.append(mc.variant))
    monkeypatch.setattr(sched, "_drive", lambda *a, **k: None)
    sched._run_entry(lane, Entry("j1", "c1", MODEL, 0.0))
    assert calls == ["mixed-p015"]  # reloaded
    lane.worker, lane.model, lane.variant = FakeWorker(), MODEL, "mixed-p015"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "request.json").write_text(request, encoding="utf-8")
    sched._run_entry(lane, Entry("j1", "c1", MODEL, 0.0))
    assert calls == ["mixed-p015"]  # the same variant stays resident: no second load


def test_busy_sees_queued_loading_and_running_jobs_of_that_model_only(tmp_path):
    sched, _, lane = _scheduler(tmp_path, "int8")
    assert not sched.busy(MODEL)
    lane.queue.append(Entry("j", "c", MODEL, 0.0))
    assert sched.busy(MODEL) and not sched.busy("minimax-h3-fl2va")
    lane.queue.clear()
    lane.loading = MODEL
    assert sched.busy(MODEL)
    lane.loading, lane.model, lane.running = None, MODEL, "j"
    assert sched.busy(MODEL)


def test_release_stops_an_idle_worker_and_never_one_that_is_working(tmp_path):
    sched, _, lane = _scheduler(tmp_path, "int8")
    worker = FakeWorker()
    lane.worker, lane.model, lane.variant = worker, MODEL, "int8"
    lane.running = "j"
    assert sched.release(MODEL) is False and not worker.stopped
    lane.running = None
    assert sched.release(MODEL) is True and worker.stopped and lane.worker is None and lane.variant is None
    assert sched.release(MODEL) is False  # nothing resident any more
