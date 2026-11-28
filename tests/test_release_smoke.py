"""The release gate: real jobs from the installed program, one per model family and build type.

Root cause it guards: dev2 was released with workers that could not import their own modules. Every test was green and the installer
smoke passed, because nothing ran a job from what had been installed. The gate itself needs a GPU; these tests keep it demanded.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from aha_open.catalog import CATALOG

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "windows_installer"))
import build_installer as b  # noqa: E402
import release_smoke  # noqa: E402

TAG = b.tag_for(b.read_version())


def test_every_family_and_build_type_of_the_catalog_has_a_required_job():
    def kind(model: str, variant: str) -> tuple[str, str]:
        return CATALOG[model].family, CATALOG[model].variants[variant].get("runtime", "")

    offered = {kind(m.id, v) for m in CATALOG.values() if not m.internal for v in m.variants}
    assert offered == {kind(m, v) for m, v in b.REQUIRED_JOBS}


def make_release(dist: Path, jobs=b.REQUIRED_JOBS) -> None:
    (dist / b.installer_name(b.read_version())).write_bytes(b"installer")
    record = {"installer_sha256": hashlib.sha256(b"installer").hexdigest(), "passed_at": "2026-01-01T00:00:00Z",
              "jobs": [{"model": m, "variant": v} for m, v in jobs]}
    (dist / b.SMOKE_RECORD).write_text(json.dumps(record))


def test_a_release_with_every_job_passed_is_accepted(tmp_path):
    make_release(tmp_path)
    assert b.check_release_record(TAG, tmp_path).startswith("OK")


@pytest.mark.parametrize("broken, says", [
    (lambda d: (d / b.SMOKE_RECORD).unlink(), "no real job"),
    (lambda d: (d / b.installer_name(b.read_version())).write_bytes(b"rebuilt"), "another installer"),
    (lambda d: make_release(d, jobs=b.REQUIRED_JOBS[:-1]), "minimax-h3-fl2va"),
])
def test_a_release_without_proof_is_refused(tmp_path, broken, says):
    make_release(tmp_path)
    broken(tmp_path)
    with pytest.raises(b.BuildError, match=says):
        b.check_release_record(TAG, tmp_path)


def test_neither_smoke_installs_over_a_real_installation(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "_uninstall_key_present", lambda: True)
    ran = []
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **k: ran.append(a))
    with pytest.raises(b.BuildError, match="already installed"):
        b.smoke(tmp_path / "setup.exe")
    with pytest.raises(b.BuildError, match="already installed"):
        release_smoke.install_exe(tmp_path / "setup.exe", tmp_path)
    assert ran == []


def fake_setup(monkeypatch, python_home) -> list[dict]:
    """Stand in for the `.exe`: record the environment it was given and leave the venv `uv` would have made."""
    given: list[dict] = []

    def run(argv, env, **kwargs):
        given.append(env)
        venv = Path(next(a for a in argv if a.startswith("/DIR="))[5:]) / "venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(f"home = {python_home(env)}\nversion_info = 3.12.0\n", encoding="utf-8")
        return release_smoke.subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(b, "refuse_if_installed", lambda: None)
    monkeypatch.setattr(release_smoke.subprocess, "run", run)
    return given


def test_the_installer_is_run_as_on_a_machine_without_python(monkeypatch, tmp_path):
    """dev3 passed this gate and could not be installed where there was no Python: the gate only ever ran where there was one."""
    monkeypatch.setenv("UV_PYTHON", "3.13")  # a developer's setting must not leak into the simulated machine
    given = fake_setup(monkeypatch, lambda env: Path(env["UV_PYTHON_INSTALL_DIR"]) / "cpython-3.12-windows-x86_64-none")
    release_smoke.install_exe(tmp_path / "setup.exe", tmp_path)
    env = given[0]
    assert Path(env["UV_PYTHON_INSTALL_DIR"]).is_relative_to(tmp_path)
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"  # a system Python is not looked for
    assert env["UV_PYTHON_INSTALL_REGISTRY"] == env["UV_PYTHON_INSTALL_BIN"] == "0"  # nothing left on the real machine
    assert "UV_PYTHON" not in env


def test_an_install_that_found_a_python_elsewhere_does_not_pass_the_gate(monkeypatch, tmp_path):
    """The simulation is proven by what was built, not by the variables that asked for it."""
    fake_setup(monkeypatch, lambda env: tmp_path.parent / "Python312")
    with pytest.raises(b.BuildError, match="without Python"):
        release_smoke.install_exe(tmp_path / "setup.exe", tmp_path)


@pytest.mark.parametrize("windowless, console, says", [
    (True, True, None),
    (True, False, "no console"),  # dev3/dev4: a terminal window opened while generating
    (False, True, "proves nothing"),  # a console `pythonw` would hide the defect, as the one in a dev `.venv` did
])
def test_the_gate_starts_the_gateway_as_the_shortcut_does(monkeypatch, tmp_path, windowless, console, says):
    ran = []

    def run(argv, **kwargs):
        ran.append(argv)
        (tmp_path / "data" / "run").mkdir(parents=True)
        (tmp_path / "data" / "run" / "gateway.json").write_text('{"pid": 1234}', encoding="utf-8")
        return release_smoke.subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(release_smoke.subprocess, "run", run)
    monkeypatch.setattr(release_smoke, "is_windowless", lambda exe: windowless)
    monkeypatch.setattr(release_smoke, "has_console", lambda pid: console and pid == 1234)
    py = tmp_path / "Scripts" / "python.exe"
    if says:
        with pytest.raises(b.BuildError, match=says):
            release_smoke.start_like_the_shortcut(py, {}, tmp_path)
    else:
        release_smoke.start_like_the_shortcut(py, {}, tmp_path)
        assert ran[0][0] == str(py.with_name("pythonw.exe"))


def test_removing_the_temporary_folder_never_touches_the_original_weights(tmp_path):
    source, target = tmp_path / "real", tmp_path / "temp"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "big.safetensors").write_bytes(bytes(release_smoke.LINK_FROM_BYTES))
    (source / "sub" / "install.json").write_text("original")
    release_smoke.link_weights(source, target)
    assert os.path.samefile(source / "sub" / "big.safetensors", target / "sub" / "big.safetensors")  # linked, not copied
    (target / "sub" / "install.json").write_text("rewritten by the program under test")
    release_smoke.shutil.rmtree(target)
    assert (source / "sub" / "big.safetensors").stat().st_size == release_smoke.LINK_FROM_BYTES
    assert (source / "sub" / "install.json").read_text() == "original"
