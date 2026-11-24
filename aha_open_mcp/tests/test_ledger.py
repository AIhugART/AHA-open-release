"""A regression must show up as a DIFF between two recorded runs, and undoing it as FIXED (docs/08).

The "defect" is the product's own: the fake worker returns an image of the wrong size when the prompt says so, and the
gateway must reject it (no silent resize). Breaking and then restoring that behaviour is a real change in what the server does.
"""

from __future__ import annotations

import json

from aha_open_mcp import ledger, tools


def run(prompt_override: str | None = None) -> str:
    overrides = json.dumps({"prompt": prompt_override}) if prompt_override else ""
    return tools.run_scenario("generate_exact_size", overrides_json=overrides)


def test_a_broken_run_is_a_regression_and_the_restored_run_is_fixed(sandbox_up):
    run()
    assert run("x __wrongsize").count("VERDICT: FAIL") == 1
    regression = tools.diff_runs("generate_exact_size")
    assert "REGRESSION job.status eq: expected 'succeeded', was 'succeeded', now 'failed'" in regression
    assert "CHANGED    fact job_status: 'succeeded' -> 'failed'" in regression
    run()
    fixed = tools.diff_runs("generate_exact_size")
    assert "FIXED      job.status eq: now 'succeeded'" in fixed
    assert "REGRESSION" not in fixed


def test_two_identical_runs_have_the_same_result_bytes_and_no_difference(sandbox_up):
    run()
    run()
    assert tools.diff_runs("generate_exact_size").endswith("no difference")
    a, b = ledger.last_two("generate_exact_size")
    assert a["facts"]["result_sha256"] == b["facts"]["result_sha256"]


def test_a_diff_needs_two_runs(sandbox_up):
    assert "a diff needs two" in tools.diff_runs("cancel_running_job")


def test_a_scenario_id_cannot_walk_out_of_the_ledger_folder(sandbox_up):
    assert tools.diff_runs("..\\..\\x").startswith("ERROR: scenario id must match")
