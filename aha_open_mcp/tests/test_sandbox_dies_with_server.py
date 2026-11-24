"""A sandbox is throwaway: it must not outlive the MCP server that started it.

Root cause this guards (observed 2026-11-23): the MCP server exited when its stdin closed while a sandbox was running, and
the sandbox gateway kept running with its temporary folder, once per aborted session.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aha_open_mcp import proc, sandbox, targets


def start_server(state: str) -> subprocess.Popen:
    env = {**os.environ, "AHA_OPEN_MCP_STATE_DIR": state, "PYTHONUTF8": "1"}
    return subprocess.Popen(
        [sys.executable, "-m", "aha_open_mcp.server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=env, text=True, encoding="utf-8",
    )


def send(server: subprocess.Popen, message: dict) -> None:
    server.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
    server.stdin.flush()


def test_closing_the_server_stdin_stops_the_running_sandbox_and_removes_its_folder():
    state = tempfile.mkdtemp(prefix="aha-open-mcp-die-state-")
    server = start_server(state)
    try:
        send(server, {"id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        send(server, {"method": "notifications/initialized"})
        send(server, {"id": 2, "method": "tools/call", "params": {"name": "run_sandbox_start", "arguments": {}}})
        info_path = Path(state) / targets.SANDBOX_FILE
        deadline = time.monotonic() + 120
        while not info_path.exists():
            assert time.monotonic() < deadline and server.poll() is None
            time.sleep(0.1)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        assert proc.pid_alive(info["pid"]) and Path(info["data_dir"]).exists()
        server.stdin.close()  # the agent session ends
        server.wait(timeout=60)
        assert not proc.pid_alive(info["pid"])
        assert not Path(info["data_dir"]).exists()
        assert not info_path.exists()
    finally:
        if server.poll() is None:
            server.kill()
        # If the behaviour under test is broken the sandbox survives: do not leave it behind.
        try:
            left = json.loads((Path(state) / targets.SANDBOX_FILE).read_text(encoding="utf-8"))
            for pid in (left.get("pid"), left.get("launcher_pid")):
                if pid:
                    proc.kill_pid(int(pid))
            shutil.rmtree(left["data_dir"], ignore_errors=True)
        except (OSError, ValueError, KeyError):
            pass


def test_a_sandbox_left_by_a_killed_server_is_cleaned_when_the_next_one_starts(stopped):
    leftover = Path(tempfile.mkdtemp(prefix="aha-open-sandbox-"))
    (leftover / targets.SANDBOX_MARKER).write_text("x", encoding="utf-8")
    targets.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (targets.STATE_DIR / targets.SANDBOX_FILE).write_text(
        json.dumps({"data_dir": str(leftover), "base_url": "http://127.0.0.1:1", "pid": None, "launcher_pid": None}), encoding="utf-8")
    info = sandbox.start()
    assert not leftover.exists()
    assert Path(info["data_dir"]) != leftover
