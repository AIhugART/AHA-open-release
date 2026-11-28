"""The installed package must carry everything a worker needs.

Root cause this guards: `workers/` sits next to the package, not inside it, so a wheel built with the default package
discovery had no worker code and `pip install aha-open` could never run a model. The wheel is checked for real by
`uv build` in the release checklist; these tests keep the configuration that makes it work.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from aha_open import catalog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "windows_installer"))
import build_installer  # noqa: E402
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]


def test_workers_are_mapped_into_the_package():
    assert "aha_open.workers" in PYPROJECT["packages"]
    assert PYPROJECT["package-dir"]["aha_open.workers"] == "workers"


def test_every_worker_script_and_its_lock_are_shipped():
    patterns = PYPROJECT["package-data"]["aha_open.workers"]
    for needed in ("*/*.py", "*/pyproject.toml", "*/uv.lock"):
        assert needed in patterns
    for folder in (ROOT / "workers").iterdir():
        if folder.is_dir() and (folder / "worker.py").exists() and folder.name != "fake":
            # a real worker has its own pinned environment: lock + pyproject must exist to be shipped
            assert (folder / "uv.lock").exists() and (folder / "pyproject.toml").exists(), folder.name


def test_uv_installs_the_project_itself_so_the_aha_open_command_exists():
    """Root cause (found on a clean clone of the public repo): no [build-system] made uv treat the project as "virtual",
    so `uv sync` (the README quick start) installed the dependencies but not `aha_open` or the `aha-open` command, and a
    bare `pytest` could not import `aha_open`. Everything passed on the dev machine only because `python -m pytest` adds
    the cwd to sys.path."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject.get("build-system", {}).get("build-backend"), "no [build-system]: uv sync would not install aha-open"
    assert pyproject["project"]["scripts"]["aha-open"] == "aha_open.cli:main"


def test_the_ui_is_shipped():
    assert "ui/*" in PYPROJECT["package-data"]["aha_open"]
    assert (ROOT / "aha_open" / "ui" / "index.html").exists()


def test_workers_dir_resolves_to_real_workers():
    for family in {spec.family for spec in catalog.CATALOG.values()}:
        assert (catalog.WORKERS_DIR / family / "worker.py").exists() or family == "minimax_h3", family


def test_the_import_walk_sees_the_modules_the_workers_are_made_of():
    """If the walk found nothing, the wheel check below would pass for the wrong reason."""
    needed = build_installer.worker_files()
    for name in ("qwen_image/qwen_models.py", "qwen_image/convrot.py", "qwen_image/convrot_lora.py", "qwen_image/lenient_safetensors.py",
                 "minimax_h3/h3_runner.py", "minimax_h3/h3_video.py", "minimax_h3/h3_references.py", "minimax_h3/h3_models.py"):
        assert name in needed


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv builds the wheel")
def test_the_built_wheel_carries_every_module_each_worker_imports(tmp_path):
    """The artefact itself, not the configuration that should produce it: the file names in pyproject looked right in dev2."""
    # build from a copy: setuptools leaves a `build/` folder behind, and a stale one would put old files into the wheel
    source = tmp_path / "src"
    for name in ("aha_open", "workers"):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        shutil.copy2(ROOT / name, source / name)
    done = subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path / "out")], cwd=source, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    names = set(zipfile.ZipFile(next((tmp_path / "out").glob("*.whl"))).namelist())
    missing = [f for f in build_installer.worker_files() if f"aha_open/workers/{f}" not in names]
    assert not missing, f"the installed program could not run these workers, missing: {missing}"
    assert not [n for n in names if Path(n).name.startswith("p0_")], "measurement scripts are for developers only"
