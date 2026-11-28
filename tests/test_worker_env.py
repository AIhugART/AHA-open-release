"""A model's engine (its worker environment) is either complete or not there.

Root cause this guards: `worker_python` only looked for `python.exe`, which `uv venv` creates BEFORE the packages. A setup cut off by a
lost connection or a Cancel left a model that said "ready", could not run a single job, and was never offered a new setup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aha_open import catalog, install
from aha_open.config import Config, ModelConfig

SCRIPTS = Path("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """No developer `.venv` next to the worker sources: only the environment in the data folder can count."""
    workers = tmp_path / "src"
    for family in ("qwen_image", "minimax_h3"):
        (workers / family).mkdir(parents=True)
        (workers / family / "worker.py").write_text("")
        (workers / family / "pyproject.toml").write_text("")
    monkeypatch.setattr(catalog, "WORKERS_DIR", workers)
    monkeypatch.setattr(install, "WORKERS_DIR", workers)
    monkeypatch.setattr(install, "uv_executable", lambda: "uv")
    return Config(data_dir=tmp_path / "data", models=[])


def fake_uv(monkeypatch, fail_at: str | None):
    """`uv venv` leaves a python behind, as the real one does; `fail_at` names the step that ends with an error."""

    def run(cmd, **_):
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / SCRIPTS
            python.parent.mkdir(parents=True)  # refuses an existing folder, as `uv venv` does
            python.write_text("")
        return subprocess.CompletedProcess(cmd, 1 if cmd[1] == fail_at else 0)

    monkeypatch.setattr(install.subprocess, "run", run)


MC = ModelConfig(id="qwen-image-2.1", worker="qwen_image", variant="int8")


def test_a_setup_that_failed_half_way_does_not_count_and_can_be_run_again(cfg, monkeypatch):
    fake_uv(monkeypatch, fail_at="sync")
    assert install._make_worker_env(cfg, "qwen_image") != 0
    assert (cfg.data_dir / "workers" / "qwen_image" / "venv" / SCRIPTS).exists()  # the half environment is there ...
    assert catalog.worker_python(cfg, MC) is None  # ... and does not count

    fake_uv(monkeypatch, fail_at=None)  # the second try starts over (the fake refuses to create over an existing folder)
    assert install._make_worker_env(cfg, "qwen_image") == 0
    assert catalog.worker_python(cfg, MC) is not None


def test_an_environment_built_by_an_earlier_version_still_counts(cfg):
    python = cfg.data_dir / "workers" / "qwen_image" / "venv" / SCRIPTS
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert catalog.worker_python(cfg, MC) == python


def test_the_reason_names_what_to_do_when_the_weights_are_here_and_the_engine_is_not(cfg):
    spec = catalog.CATALOG["qwen-image-2.1"]
    weights = cfg.data_dir / "models" / "qwen-image-2.1" / "int8"
    for name in ("model_index.json", *spec.required_files):
        (weights / name).parent.mkdir(parents=True, exist_ok=True)
        (weights / name).write_text("{}")
    state, detail = catalog.model_state(cfg, MC, spec)
    assert state == "not_installed"
    assert "engine is not set up" in detail and "Installed Models" in detail
