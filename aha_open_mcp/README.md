# aha-open-mcp

An MCP server that lets a coding agent drive, observe and judge a running AHA open server end to end. What it is and the
rules it follows: [../docs/08_auto_test_and_debug.md](../docs/08_auto_test_and_debug.md).

```bash
cd aha_open_mcp
uv sync --extra dev          # its own environment; the gateway never needs it
uv run pytest                # the tool's own tests (real gateway, fake worker)
claude mcp add aha-open-mcp -- uv run --project <path-to>/aha_open_mcp aha-open-mcp
```

Restart the agent session after registering or updating: the tool list is fixed when the session starts.
