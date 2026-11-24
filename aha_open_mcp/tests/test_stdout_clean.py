"""Stdout of the server is the protocol channel: every line must be a JSON-RPC message, whatever the tools do.

The SDK's own client tolerates junk lines (it surfaces them as parse errors and carries on), so a test that only uses that
client cannot catch a stray `print`; other MCP clients may drop the connection. This test reads the raw bytes instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time


def rpc(i: int, method: str, params: dict) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params})


def test_every_line_the_server_writes_to_stdout_is_a_json_rpc_message():
    env = {**os.environ, "AHA_OPEN_MCP_STATE_DIR": tempfile.mkdtemp(prefix="aha-open-mcp-raw-state-"), "PYTHONUTF8": "1"}
    messages = [
        rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "raw-test", "version": "0"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        rpc(2, "tools/call", {"name": "run_sandbox_start", "arguments": {}}),
        rpc(3, "tools/call", {"name": "check_health", "arguments": {"target": "sandbox"}}),
        rpc(4, "tools/call", {"name": "read_gateway_log", "arguments": {"target": "sandbox"}}),
        rpc(5, "tools/call", {"name": "run_sandbox_stop", "arguments": {}}),
    ]
    proc = subprocess.Popen(
        [sys.executable, "-m", "aha_open_mcp.server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True, encoding="utf-8",
    )
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(ln for ln in proc.stdout if ln.strip()), daemon=True)
    reader.start()

    def answered() -> set:
        ids = set()
        for ln in list(lines):
            try:
                ids.add(json.loads(ln).get("id"))
            except ValueError:
                pass
        return ids

    try:
        for i, message in enumerate(messages):
            proc.stdin.write(message + "\n")
            proc.stdin.flush()
            wanted = json.loads(message).get("id")
            deadline = time.monotonic() + 120
            while wanted is not None and wanted not in answered():  # keep stdin open: closing it cancels a call in flight
                assert time.monotonic() < deadline and proc.poll() is None, (wanted, lines)
                time.sleep(0.05)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader.join(timeout=5)
    assert lines
    for line in lines:
        message = json.loads(line)  # a stray print raises here
        assert message.get("jsonrpc") == "2.0", line
