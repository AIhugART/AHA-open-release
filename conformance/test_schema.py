"""P1 gate: the machine-readable OpenAPI file and the human spec must describe the same /v1 surface."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schema" / "aha_provider_v1.openapi.yaml"
DOC = ROOT / "docs" / "03_api_v1.md"


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{id}", path)


def _doc_routes() -> set[tuple[str, str]]:
    """(METHOD, path) pairs from the route table of docs/03_api_v1.md section 2."""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## 2. Route", 1)[1].split("\n## ", 1)[0]
    pairs: set[tuple[str, str]] = set()
    for row in section.splitlines():
        if not row.startswith("| `"):
            continue
        first = row.split("|")[1]
        for method, path in re.findall(r"`(GET|POST|PUT) (/v1[^`]*)`", first):
            pairs.add((method, _norm(path.split("?")[0])))
    return pairs


def _schema_routes(spec: dict) -> set[tuple[str, str]]:
    return {
        (method.upper(), _norm(path))
        for path, item in spec["paths"].items()
        for method in item
        if method in {"get", "post", "put", "delete"}
    }


def test_every_documented_route_is_in_the_schema(spec: dict) -> None:
    documented = _doc_routes()
    assert documented, "the doc route table was not parsed"
    missing = documented - _schema_routes(spec)
    assert not missing, f"documented but absent from OpenAPI: {sorted(missing)}"


def test_every_schema_route_is_documented(spec: dict) -> None:
    extra = _schema_routes(spec) - _doc_routes()
    assert not extra, f"in OpenAPI but undocumented: {sorted(extra)}"


def test_all_refs_resolve(spec: dict) -> None:
    refs: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref":
                    refs.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    assert refs
    for ref in refs:
        assert ref.startswith("#/"), ref
        node: object = spec
        for part in ref[2:].split("/"):
            assert isinstance(node, dict) and part in node, f"dangling $ref {ref}"
            node = node[part]


def test_error_codes_in_the_doc_table() -> None:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## 8. Error codes", 1)[1].split("\n## ", 1)[0]
    codes = set(re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.MULTILINE))
    assert {"invalid_request", "unauthorized", "forbidden", "not_found", "busy", "model_loading"} <= codes


def test_unauthenticated_routes_are_exactly_health_and_enroll(spec: dict) -> None:
    open_routes = {
        path
        for path, item in spec["paths"].items()
        for method, op in item.items()
        if method in {"get", "post", "put"} and op.get("security") == []
    }
    assert open_routes == {"/v1/health", "/v1/enroll", "/v1/enroll/{enroll_id}"}
