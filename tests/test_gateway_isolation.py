"""The gateway must not depend on the test harness or on the MCP SDK, the same way it must not import torch.

AST scan of the gateway package: an import inside a docstring or a comment is not an import, and a lazy import inside a
function still is one.
"""

from __future__ import annotations

import ast
from pathlib import Path

GATEWAY = Path(__file__).resolve().parents[1] / "aha_open"
FORBIDDEN = {"mcp", "aha_open_mcp"}


def imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_gateway_module_imports_the_mcp_sdk_or_the_harness():
    offenders = {
        str(p.relative_to(GATEWAY)): sorted(imported_roots(p.read_text(encoding="utf-8")) & FORBIDDEN)
        for p in GATEWAY.rglob("*.py")
        if imported_roots(p.read_text(encoding="utf-8")) & FORBIDDEN
    }
    assert not offenders, offenders


def test_the_scanner_sees_a_lazy_import_but_not_a_docstring_mention():
    assert imported_roots("def f():\n    import mcp.server\n") == {"mcp"}
    assert imported_roots('"""import mcp"""\n# from mcp import x\nx = 1\n') == set()


def test_the_gateway_does_not_declare_the_harness_as_a_dependency():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "mcp" not in dependencies
