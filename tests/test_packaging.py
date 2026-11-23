"""The installed package must carry everything a worker needs (plan P8).

Root cause this guards: `workers/` sits next to the package, not inside it, so a wheel built with the default package
discovery had no worker code and `pip install aha-open` could never run a model. The wheel is checked for real by
`uv build` in the release checklist; these tests keep the configuration that makes it work.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from aha_open import catalog

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]


def test_workers_are_mapped_into_the_package():
    assert "aha_open.workers" in PYPROJECT["packages"]
    assert PYPROJECT["package-dir"]["aha_open.workers"] == "workers"


def test_every_worker_script_and_its_lock_are_shipped():
    patterns = PYPROJECT["package-data"]["aha_open.workers"]
    for needed in ("*/worker.py", "*/pyproject.toml", "*/uv.lock"):
        assert needed in patterns
    for folder in (ROOT / "workers").iterdir():
        if folder.is_dir() and (folder / "worker.py").exists() and folder.name != "fake":
            # a real worker has its own pinned environment: lock + pyproject must exist to be shipped
            assert (folder / "uv.lock").exists() and (folder / "pyproject.toml").exists(), folder.name


def test_the_ui_is_shipped():
    assert "ui/*" in PYPROJECT["package-data"]["aha_open"]
    assert (ROOT / "aha_open" / "ui" / "index.html").exists()


def test_workers_dir_resolves_to_real_workers():
    for family in {spec.family for spec in catalog.CATALOG.values()}:
        assert (catalog.WORKERS_DIR / family / "worker.py").exists() or family == "minimax_h3", family
