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
