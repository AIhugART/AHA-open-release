"""Sandbox lifecycle: nothing left behind, and nothing deleted that the tool did not create."""

from __future__ import annotations

import json
from pathlib import Path

from aha_open_mcp import proc, targets, tools


def test_no_sandbox_is_an_actionable_error(stopped):
    assert tools.check_health("sandbox") == "ERROR: target=sandbox: no sandbox is running - call run_sandbox_start first"


def test_an_unknown_target_is_refused(stopped):
    assert tools.check_health("prod").startswith("ERROR: target must be one of")


def test_start_and_stop_leave_nothing_behind(stopped):
    out = tools.run_sandbox_start()
    assert out.startswith("SIDE EFFECTS:")  # said before it acts
    info = json.loads((targets.STATE_DIR / targets.SANDBOX_FILE).read_text(encoding="utf-8"))
    pid, data = info["pid"], Path(info["data_dir"])
    assert proc.pid_alive(pid) and data.exists()
    stopped_out = tools.run_sandbox_stop()
    assert "all gone" in stopped_out and "removed" in stopped_out
    assert not proc.pid_alive(pid) and not data.exists()


def test_stop_never_deletes_a_directory_it_did_not_create(stopped, tmp_path):
    victim = tmp_path / "aha-open-sandbox-not-mine"  # right name, but no marker file
    victim.mkdir()
    (victim / "keep.txt").write_text("precious", encoding="utf-8")
    targets.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (targets.STATE_DIR / targets.SANDBOX_FILE).write_text(
        json.dumps({"data_dir": str(victim), "base_url": "http://127.0.0.1:1", "pid": None, "launcher_pid": None}), encoding="utf-8")
    out = tools.run_sandbox_stop()
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "precious"
    assert "kept at" in out
