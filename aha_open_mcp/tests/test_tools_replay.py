"""Replay of every M2 tool body against a real gateway (no mock, so the tool cannot drift from the contract)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from aha_open_mcp import evidence, proc, targets, tools


def payload(text: str) -> dict:
    """The JSON object that follows the two header lines."""
    return json.loads(text[text.index("{"):])


def test_every_output_starts_with_a_header_naming_machine_target_worker_evidence(sandbox_up):
    out = tools.check_health("sandbox")
    first, second = out.splitlines()[:2]
    assert first.startswith("machine: ") and "| target: sandbox |" in first and "| worker: fake |" in first
    assert "evidence: 🧪 (fake worker)" in first
    assert second == "harness: fresh | gateway: fresh"


def test_check_health_reports_the_server_state(sandbox_up):
    h = payload(tools.check_health("sandbox"))
    assert h["http_status"] == 200 and h["health"]["status"] == "ok"
    assert h["health"]["queue"] == {"queued": 0, "running": 0}


def test_probe_v1_reads_the_manifest_with_an_allowlist_of_fields(sandbox_up):
    m = payload(tools.probe_v1("/v1/manifest"))["manifest"]
    assert {x["id"] for x in m["models"]} == {"example-image", "example-video"}
    assert set(m["models"][0]) <= {"id", "engine", "tasks", "state", "state_detail"}  # no capabilities, no license text
    assert m["provider"]["name"] == "aha-open"


def test_probe_v1_refuses_anything_outside_the_read_only_allowlist(sandbox_up):
    for path in ("/v1/uploads", "/admin/loras", "/v1/jobs/x/cancel", "/v1/health?x=1", "/ui/index.html"):
        assert tools.probe_v1(path).startswith("ERROR: probe_v1 only reads"), path


def test_an_unknown_job_is_a_404_with_the_error_code_not_a_crash(sandbox_up):
    r = payload(tools.probe_v1("/v1/jobs/job_does_not_exist"))
    assert r["http_status"] == 404 and r["error"]["code"] == "not_found"


def test_read_gateway_log_names_the_file_its_age_and_the_worker_start(sandbox_up):
    tools.probe_v1("/v1/manifest")  # produce an access-log line
    out = tools.read_gateway_log("sandbox", 50)
    assert re.search(r"log: .*gateway\.log", out) and "last modified: " in out and "(age " in out
    assert "/v1/manifest" in out


def test_read_worker_log_rejects_a_name_that_could_walk_out_of_the_log_folder(sandbox_up):
    assert tools.read_worker_log("..\\..\\config").startswith("ERROR: worker must match")
    assert tools.read_worker_log("../x").startswith("ERROR: worker must match")


def test_a_missing_log_says_not_found_instead_of_an_empty_tail(sandbox_up):
    assert "NOT FOUND" in tools.read_worker_log("qwen_image")


def test_full_diagnostics_gathers_header_health_manifest_and_logs(sandbox_up):
    out = tools.full_diagnostics("sandbox")
    for part in ("== health", "== manifest", "== gateway.log (last 20)"):
        assert part in out


def test_the_log_read_follows_a_rotated_backup(tmp_path):
    from aha_open_mcp import logs

    (tmp_path / "gateway.log.1").write_text("old-1\nold-2\n", encoding="utf-8")
    (tmp_path / "gateway.log").write_text("new-1\n", encoding="utf-8")
    out = logs.read_tail(tmp_path / "gateway.log", 3)
    assert "gateway.log.1" in out and out.rstrip().endswith("old-1\nold-2\nnew-1")


def test_a_gateway_running_older_code_than_the_disk_is_flagged_stale(sandbox_up):
    t = targets.resolve("sandbox")
    rec_path = t.data_dir / "run" / "gateway.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    try:
        rec_path.write_text(json.dumps({**rec, "source_digest": "0" * 64}), encoding="utf-8")
        assert evidence.gateway_state(t).startswith("STALE GATEWAY")
        assert "STALE GATEWAY" in tools.check_health("sandbox")
    finally:
        rec_path.write_text(json.dumps(rec), encoding="utf-8")
    assert evidence.gateway_state(t) == "fresh"


def test_a_record_naming_a_dead_pid_says_not_running(sandbox_up):
    t = targets.resolve("sandbox")
    rec_path = t.data_dir / "run" / "gateway.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    try:
        rec_path.write_text(json.dumps({**rec, "pid": 2**22 - 1}), encoding="utf-8")
        assert evidence.gateway_state(t).startswith("NOT RUNNING")
    finally:
        rec_path.write_text(json.dumps(rec), encoding="utf-8")


def test_the_token_never_appears_in_any_output(sandbox_up):
    token = targets.resolve("sandbox").token()
    assert len(token) > 16
    outputs = [
        tools.check_harness(), tools.check_health(), tools.probe_v1("/v1/manifest"), tools.read_gateway_log(),
        tools.read_worker_log("fake"), tools.full_diagnostics(),
    ]
    for out in outputs:
        assert token not in out


