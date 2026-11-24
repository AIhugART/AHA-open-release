"""The MCP server (stdio). Registers the tool bodies of tools.py; contains no behaviour of its own."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer  # mcp 2.x (FastMCP of 1.x was renamed); pinned >=2,<3 in pyproject.toml

from . import sandbox, tools

mcp = MCPServer("aha-open-mcp")
for _tool in tools.TOOLS:
    mcp.tool()(_tool)


def main() -> None:
    try:
        mcp.run()  # stdio: stdout is the protocol channel, nothing else may print to it
    finally:
        sandbox.stop()  # a sandbox is throwaway: it must not outlive the server that started it (orphans were observed)


if __name__ == "__main__":
    main()
