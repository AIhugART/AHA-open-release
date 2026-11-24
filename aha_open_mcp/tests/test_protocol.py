"""The tool is exercised the way an agent uses it: a real `aha-open-mcp` process, the real MCP protocol over stdio.

Whether stdout stays clean is NOT proved here (the SDK client tolerates junk lines): see test_stdout_clean.py.
"""

from __future__ import annotations

import os
import sys
import tempfile

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {
    "check_harness", "check_health", "probe_v1", "read_gateway_log", "read_worker_log", "full_diagnostics",
    "run_sandbox_start", "run_sandbox_stop", "run_upload", "run_job", "wait_job", "run_cancel", "assert_result_image",
    "list_scenarios", "get_scenario", "run_scenario", "diff_runs", "diagnose_job", "run_ui_url",
}


def text_of(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def test_the_server_lists_its_tools_and_runs_a_whole_sandbox_session_over_stdio():
    state = tempfile.mkdtemp(prefix="aha-open-mcp-proto-state-")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "aha_open_mcp.server"], env={**os.environ, "AHA_OPEN_MCP_STATE_DIR": state},
    )
    seen: dict[str, str] = {}

    async def session() -> None:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            listed = await client.list_tools()
            seen["tools"] = ",".join(sorted(t.name for t in getattr(listed, "tools", listed)))
            try:
                seen["start"] = text_of(await client.call_tool("run_sandbox_start", {}))
                seen["health"] = text_of(await client.call_tool("check_health", {"target": "sandbox"}))
                seen["probe"] = text_of(await client.call_tool("probe_v1", {"path": "/v1/uploads"}))
            finally:
                seen["stop"] = text_of(await client.call_tool("run_sandbox_stop", {}))

    anyio.run(session)
    assert set(seen["tools"].split(",")) == EXPECTED
    assert seen["start"].startswith("SIDE EFFECTS:") and "sandbox up at http://127.0.0.1:" in seen["start"]
    assert '"status": "ok"' in seen["health"] and "harness: fresh | gateway: fresh" in seen["health"]
    assert seen["probe"].startswith("ERROR: probe_v1 only reads")
    assert "all gone" in seen["stop"]
