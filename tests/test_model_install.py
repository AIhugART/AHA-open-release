"""The pieces of the web installer that need no gateway: finding `uv`, and refusing to install what is already there."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aha_open import install, model_install
from aha_open.config import Config, ModelConfig
from aha_open.errors import ApiError
from aha_open.model_install import InstallManager


def test_uv_comes_from_the_installer_variable_before_the_path(monkeypatch, tmp_path):
    shipped = tmp_path / "uv.exe"
    shipped.write_bytes(b"x")
    monkeypatch.setenv("AHA_OPEN_UV", str(shipped))
    monkeypatch.setattr(install.shutil, "which", lambda name: "C:/elsewhere/uv.exe")
    assert install.uv_executable() == str(shipped)


def test_a_named_uv_that_is_missing_is_not_silently_replaced_by_another(monkeypatch, tmp_path):
    monkeypatch.setenv("AHA_OPEN_UV", str(tmp_path / "gone.exe"))
    monkeypatch.setattr(install.shutil, "which", lambda name: "C:/elsewhere/uv.exe")
    assert install.uv_executable() is None  # the installer said which uv to use; a different one would hide a broken install


def test_without_the_variable_uv_is_found_on_the_path(monkeypatch):
    monkeypatch.delenv("AHA_OPEN_UV", raising=False)
    monkeypatch.setattr(install.shutil, "which", lambda name: "C:/tools/uv.exe")
    assert install.uv_executable() == "C:/tools/uv.exe"


def test_uv_next_to_the_interpreter_is_found_when_the_path_has_none(monkeypatch, tmp_path):
    """The Windows installer puts uv in <app>\\venv\\Scripts, beside python.exe, and does not touch PATH."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "python.exe").write_bytes(b"x")
    uv = scripts / ("uv.exe" if sys.platform == "win32" else "uv")
    uv.write_bytes(b"x")
    monkeypatch.delenv("AHA_OPEN_UV", raising=False)
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    monkeypatch.setattr(install.sys, "executable", str(scripts / "python.exe"))
    assert install.uv_executable() == str(uv)
    uv.unlink()
    assert install.uv_executable() is None  # nothing anywhere: the caller says so, it does not guess


def test_the_progress_class_prints_machine_readable_lines_and_draws_nothing(capsys):
    pytest.importorskip("tqdm")
    cls = install._progress_lines_class()
    bar = cls(total=1000, unit="B", desc="Downloading bytes")
    bar.update(100)
    out = capsys.readouterr()
    assert out.out == f"{install.PROGRESS_PREFIX} 100 1000\n"
    assert out.err == ""  # nothing is drawn: the page does not want terminal bars in its pipe
    bar.update(100)  # less than a second later: throttled
    assert capsys.readouterr().out == ""


def test_only_the_network_bytes_bar_is_reported(capsys):
    pytest.importorskip("tqdm")
    cls = install._progress_lines_class()
    cls._last = 0.0
    other = cls(total=1000, unit="B", desc="Reconstructing (incomplete total...)")  # disk writes, not what has arrived
    other.update(500)
    files = cls(total=48, desc="Fetching 48 files")  # a count of files, not bytes
    files.update(3)
    assert capsys.readouterr().out == ""


def test_the_command_line_keeps_its_own_progress_bars(monkeypatch):
    """The machine-readable form is only for the gateway's child process: `aha-open models install` in a terminal draws bars."""
    monkeypatch.delenv(install.PROGRESS_ENV, raising=False)
    seen = {}
    fake = type("Hub", (), {"snapshot_download": staticmethod(lambda **kw: seen.update(kw))})
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    install._download("o/r", "rev", Path("x"))
    assert "tqdm_class" not in seen
    monkeypatch.setenv(install.PROGRESS_ENV, "lines")
    pytest.importorskip("tqdm")
    install._download("o/r", "rev", Path("x"))
    assert "tqdm_class" in seen


# ---------------------------------------------------------------------------------------------- an interrupted download


def _weights_for_install(tmp_path):
    from aha_open.catalog import weights_dir

    return weights_dir(Config(data_dir=tmp_path), ModelConfig("qwen-image-2.1", "qwen_image", variant="int8"))


def _run_install(tmp_path, monkeypatch, record_record: list | None = None) -> list:
    """`aha-open models install qwen-image-2.1` with no GPU, no network and no worker environment; returns the downloads it started."""
    from aha_open.cli import main

    monkeypatch.setattr(install, "detect_gpus", lambda *a, **k: [{"index": 0, "name": "g", "memory_total_mib": 24564}])
    monkeypatch.setattr(install, "free_disk_gib", lambda path: 500.0)
    started: list = []

    def fake_download(repo, revision, target):
        started.append(target)
        if record_record is not None:
            record_record.append((target / "install.json").exists())
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_index.json").write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(install, "_download", fake_download)
    assert main(["--data-dir", str(tmp_path), "init"]) == 0
    assert main(["--data-dir", str(tmp_path), "models", "install", "qwen-image-2.1", "--variant", "int8", "--accept-license", "qwen-research", "--skip-env"]) == 0
    return started


def _state_dir(target: Path) -> Path:
    path = target / ".cache" / "huggingface" / "download" / "transformer"
    path.mkdir(parents=True)
    return path


def test_a_first_install_downloads(tmp_path, monkeypatch):
    assert len(_run_install(tmp_path, monkeypatch)) == 1


def test_an_install_that_was_cut_off_downloads_again_not_skips(tmp_path, monkeypatch):
    """model_index.json arrives among the first files. Seeing it must not mean 'done': the big files were missing."""
    target = _weights_for_install(tmp_path)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    (_state_dir(target) / "shard-00001.safetensors.xyz.incomplete").write_bytes(b"x")
    assert _run_install(tmp_path, monkeypatch) == [target]  # downloaded again (huggingface_hub continues what is incomplete)


def test_a_download_that_began_but_has_no_record_of_ending_is_resumed(tmp_path, monkeypatch):
    target = _weights_for_install(tmp_path)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    (_state_dir(target) / "shard.metadata").write_text("etag", encoding="utf-8")  # hf was here, but there is no install.json
    assert _run_install(tmp_path, monkeypatch) == [target]


def test_a_finished_install_is_not_downloaded_again(tmp_path, monkeypatch):
    target = _weights_for_install(tmp_path)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    (target / "install.json").write_text("{}", encoding="utf-8")
    (_state_dir(target) / "shard.metadata").write_text("etag", encoding="utf-8")
    assert _run_install(tmp_path, monkeypatch) == []


def test_weights_placed_by_hand_are_left_alone(tmp_path, monkeypatch):
    """No hf bookkeeping at all: somebody put the files there. That has always been accepted and still is."""
    target = _weights_for_install(tmp_path)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    assert _run_install(tmp_path, monkeypatch) == []


def test_a_model_whose_download_was_cut_off_is_not_ready(tmp_path):
    from aha_open.catalog import model_state

    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model_index.json").write_text("{}", encoding="utf-8")
    cfg = Config(data_dir=tmp_path / "data")
    mc = ModelConfig(id="qwen-image-2.1", worker="qwen_image", variant="int8", python=sys.executable, weights_dir=str(weights))
    assert model_state(cfg, mc, install.CATALOG["qwen-image-2.1"]) == ("ready", None)
    (_state_dir(weights) / "shard.incomplete").write_bytes(b"x")
    state, detail = model_state(cfg, mc, install.CATALOG["qwen-image-2.1"])
    assert state == "not_installed" and "did not finish" in detail and str(weights) not in detail
    (weights / "install.json").write_text("{}", encoding="utf-8")  # the install ended: stray .incomplete files of earlier attempts do not matter
    assert model_state(cfg, mc, install.CATALOG["qwen-image-2.1"]) == ("ready", None)


def test_stray_incomplete_files_after_a_finished_install_do_not_trigger_another_download(tmp_path, monkeypatch):
    """Seen on the real run: three .incomplete files of cut-off attempts stayed after the resume succeeded."""
    target = _weights_for_install(tmp_path)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    (target / "install.json").write_text("{}", encoding="utf-8")
    (_state_dir(target) / "old-attempt.incomplete").write_bytes(b"x")
    assert _run_install(tmp_path, monkeypatch) == []


def test_a_new_download_removes_the_record_of_the_previous_one(tmp_path, monkeypatch):
    """install.json vouches for 'finished'. If the files are gone and a download starts, the old record must not vouch for it."""
    target = _weights_for_install(tmp_path)
    target.mkdir(parents=True)
    (target / "install.json").write_text("{}", encoding="utf-8")  # a record from an earlier install; model_index.json is missing
    seen: list = []
    assert len(_run_install(tmp_path, monkeypatch, record_record=seen)) == 1
    assert seen == [False]  # it was gone when the download began


# ---------------------------------------------------------------------------------------------- which code runs


def test_the_gateway_is_started_from_the_data_folder_not_from_wherever_the_user_stood(tmp_path, monkeypatch):
    """`python -m` puts the current folder first on sys.path: from a source checkout that would run the checkout, not the install."""
    from aha_open import launcher

    cfg = Config(data_dir=tmp_path / "data", listen="127.0.0.1:1")
    cfg.data_dir.mkdir()
    seen = {}

    class Dead:
        def poll(self):
            return 1

    def popen(argv, **kwargs):
        seen.update(kwargs)
        return Dead()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "health_ok", lambda cfg, timeout=2.0: False)
    monkeypatch.setattr(launcher, "_port_in_use", lambda cfg: False)
    launcher.start(cfg)
    assert seen["cwd"] == str(cfg.data_dir)


def test_a_model_is_installed_by_a_process_started_in_the_data_folder(tmp_path, monkeypatch):
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir()
    seen = {}
    real_popen = model_install.subprocess.Popen

    def popen(argv, **kwargs):
        seen.update(kwargs)
        return real_popen([sys.executable, "-c", "pass"], **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "stdin", "text", "encoding", "errors", "env", "cwd")})

    monkeypatch.setattr(model_install.subprocess, "Popen", popen)
    monkeypatch.setattr(model_install, "detect_gpus", lambda *a, **k: [{"index": 0, "name": "g", "memory_total_mib": 24564}])
    monkeypatch.setattr(model_install, "uv_executable", lambda: "uv")
    monkeypatch.setattr(model_install, "has_download_support", lambda: True)
    monkeypatch.setattr(model_install, "free_disk_gib", lambda path: 500.0)
    manager = InstallManager(cfg, command=lambda model_id, key: [sys.executable, "-c", "pass"])
    manager.start("qwen-image-2.1", "qwen-research")
    assert seen["cwd"] == str(cfg.data_dir)


def test_a_model_that_is_ready_is_not_installed_again(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model_index.json").write_text("{}", encoding="utf-8")
    cfg = Config(data_dir=tmp_path / "data",
                 models=[ModelConfig(id="qwen-image-2.1", worker="qwen_image", variant="int8", python=sys.executable, weights_dir=str(weights))])
    started: list = []
    manager = InstallManager(cfg, command=lambda model_id, key: started.append(model_id) or [sys.executable, "-c", "pass"])
    monkeypatch.setattr(model_install, "detect_gpus", lambda *a, **k: [{"index": 0, "name": "g", "memory_total_mib": 24564}])
    with pytest.raises(ApiError) as caught:
        manager.start("qwen-image-2.1", "qwen-research")
    assert caught.value.status == 409 and caught.value.code == "already_installed" and not started
