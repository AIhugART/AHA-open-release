"""A pytest plugin (loaded by pyproject's `-p no_real_downloads`), not a conftest.

Why not `tests/conftest.py`: the conformance tests do `from conftest import Api`, which only works while there is exactly one module
called `conftest`; a second one made `conformance/` fail to import when both folders ran together.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_downloads(monkeypatch):
    """No test may download from the Hugging Face Hub.

    Why: a sabotage check of the territory gate (plan 2026-11-24_minimax_h3_plan.md, M1) removed the check on purpose, the
    install went on, and the test downloaded 33 GB of real MiniMax weights into pytest's temp folder for 90 minutes instead of
    failing. Any test that reaches a download now fails at once, so a broken gate shows as red, never as a silent download.
    Tests that need a downloader install their own fake (they replace the module, which this does not prevent).
    """
    try:
        import huggingface_hub
    except ImportError:
        return

    def refuse(*args, **kwargs):
        raise AssertionError(f"a test tried to download from the Hub: {kwargs.get('repo_id') or (args[:1] or '?')}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse, raising=False)
