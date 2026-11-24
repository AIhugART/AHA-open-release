"""`check_harness` must notice that the tool's own files changed after the tool started (the "code on disk is not the
code running" class that cost the reference harness five investigations). The package is COPIED so no real file is edited."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_a_change_to_the_package_after_it_loaded_is_reported_stale_and_undoing_it_is_fresh(tmp_path):
    shutil.copytree(SRC / "aha_open_mcp", tmp_path / "aha_open_mcp", ignore=shutil.ignore_patterns("__pycache__"))
    script = textwrap.dedent(
        """
        from pathlib import Path
        from aha_open_mcp import evidence
        target = Path(evidence.__file__).with_name("proc.py")
        original = target.read_bytes()  # bytes: text mode would translate line endings on Windows
        print("before:", evidence.harness_state())
        target.write_bytes(original + b"\\n# edited after the tool started\\n")
        print("edited:", evidence.harness_state())
        target.write_bytes(original)
        print("restored:", evidence.harness_state())
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    lines = dict(line.split(": ", 1) for line in done.stdout.strip().splitlines())
    assert lines["before"] == "fresh"
    assert lines["edited"].startswith("STALE HARNESS")
    assert lines["restored"] == "fresh"
