"""Gates on the scenario DATA: a broken or drifting scenario is red here, not silently absent and not discovered mid-run."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aha_open_mcp import scenarios

GOOD = """
id: {id}
title: "t"
covers: "conformance/test_v1_core.py::test_progress_is_reported"
target: sandbox
side_effects: [creates_job]
vars: {{prompt: "x"}}
steps:
  - tool: run_job
    args: {{body: {{task: image.generate, model: example-image, prompt: "${{prompt}}"}}}}
  - tool: wait_job
assert:
  - {{path: job.status, op: eq, value: succeeded}}
"""


def write(folder: Path, name: str, text: str) -> None:
    (folder / f"{name}.yaml").write_text(textwrap.dedent(text), encoding="utf-8")


def test_every_shipped_scenario_loads_and_none_is_invalid():
    ok, invalid = scenarios.load_all()
    assert not invalid, invalid
    assert len(ok) >= 6


def test_every_step_tool_is_exercised_by_some_shipped_scenario():
    ok, _ = scenarios.load_all()
    used = {step["tool"] for sc in ok.values() for step in sc.steps}
    assert used == set(scenarios.PRODUCES), set(scenarios.PRODUCES) ^ used


def test_a_good_scenario_in_a_scratch_folder_loads(tmp_path):
    write(tmp_path, "fine", GOOD.format(id="fine"))
    ok, invalid = scenarios.load_all(tmp_path)
    assert list(ok) == ["fine"] and not invalid


@pytest.mark.parametrize(
    ("label", "mutate", "reason"),
    [
        ("id differs from the file name", lambda s: s.replace("id: bad", "id: other"), "must equal the file name"),
        ("unknown side effect", lambda s: s.replace("[creates_job]", "[burns_money]"), "unknown side_effects"),
        ("dead covers pointer (no such test)", lambda s: s.replace("test_progress_is_reported", "test_no_such_test"), "no test named"),
        ("dead covers pointer (no such file)", lambda s: s.replace("conformance/test_v1_core.py", "conformance/nope.py"), "does not exist"),
        ("assert reads something no step produces", lambda s: s.replace("path: job.status", "path: image.sha256"), "not produced by any step"),
        ("step uses an unknown variable", lambda s: s.replace("${prompt}", "${nope}"), "neither a var nor produced"),
        ("unknown op", lambda s: s.replace("op: eq", "op: roughly"), "needs path and op"),
        ("unknown step tool", lambda s: s.replace("tool: wait_job", "tool: do_magic"), "tool must be one of"),
        ("unknown precondition", lambda s: s.replace("target: sandbox", "target: sandbox\npreconditions: [moon_phase]"), "unknown preconditions"),
        ("target not sandbox/live", lambda s: s.replace("target: sandbox", "target: prod"), "target must be one of"),
    ],
)
def test_a_broken_scenario_is_listed_invalid_with_its_reason_never_skipped(tmp_path, label, mutate, reason):
    write(tmp_path, "bad", mutate(GOOD.format(id="bad")))
    ok, invalid = scenarios.load_all(tmp_path)
    assert "bad" not in ok, label
    assert reason in invalid["bad"], (label, invalid)


def test_a_yaml_syntax_error_is_invalid_too(tmp_path):
    (tmp_path / "broken.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    ok, invalid = scenarios.load_all(tmp_path)
    assert not ok and "broken" in invalid


def test_a_doc_heading_pointer_resolves_and_a_wrong_anchor_does_not():
    scenarios.resolve_covers("docs/product/08_auto_test_and_debug.md#2-boundaries")
    with pytest.raises(scenarios.ScenarioError):
        scenarios.resolve_covers("docs/product/08_auto_test_and_debug.md#no-such-heading")


def test_assertion_operators_and_paths():
    ctx = {"job": {"status": "ok", "results": [{"w": 3}, {"w": 4}], "n": 5}}

    def ok(path, op, value=None):
        return scenarios.check(ctx, {"path": path, "op": op, "value": value})["ok"]

    assert ok("job.status", "eq", "ok") and not ok("job.status", "eq", "no")
    assert ok("job.results[1].w", "eq", 4) and ok("job.results", "len_eq", 2)
    assert ok("job.n", "gt", 4) and ok("job.n", "ge", 5) and ok("job.n", "lt", 6) and ok("job.n", "le", 5) and ok("job.n", "ne", 6)
    assert ok("job.status", "in", ["ok", "x"]) and ok("job.status", "exists") and ok("job.nope", "absent")
    assert not ok("job.nope", "eq", 1) and not ok("job.results[9].w", "exists")  # a missing path never "equals" anything
