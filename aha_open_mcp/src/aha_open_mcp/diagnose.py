"""Auto-debug of ONE job: the cause and the action, the log lines of that job's time window with this tool's own poll
noise removed, and the last stage the worker reported. Read only: it never writes to the logs it reads."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from . import client, guide, jobs, logs
from .targets import Target

LINE_TIME = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} ")
POLL = re.compile(r'"GET /v1/(jobs/[^ "]+|health|manifest)[ ?"]')  # the access-log lines every poller (including this tool) makes
STAGE = re.compile(r"^\[worker \+")
MAX_WINDOW_LINES = 60
SLACK = timedelta(seconds=2)  # created_at is rounded down to the second


def _local(iso_utc: str) -> datetime:
    return datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def window(lines: list[str], start: datetime, job_id: str) -> tuple[list[str], int]:
    """Lines from `start` on that are about this job or about the server's own trouble; poll lines are counted, not shown."""
    shown: list[str] = []
    hidden = 0
    current = False  # lines without a timestamp (a traceback) belong to the entry before them
    for line in lines:
        m = LINE_TIME.match(line)
        if m:
            current = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") >= start
        if not current:
            continue
        if POLL.search(line):
            hidden += 1
        elif m is None or job_id in line or " WARNING " in line or " ERROR " in line or "aha_open." in line or "POST" in line:
            shown.append(line)
    return shown[-MAX_WINDOW_LINES:], hidden


def diagnose(target: Target, job_id: str) -> str:
    job = jobs.view(target, job_id)
    out = [f"job {job_id}: status {job['status']}"]
    if job["status"] == "succeeded":
        return "\n".join(out + ["Nothing to diagnose: the job succeeded. Use assert_result_image to judge the pixels."])
    if job["status"] in ("queued", "running", "loading_model"):
        health = client.probe(target, "/v1/health").get("health", {})
        return "\n".join(out + [f"progress: {job.get('progress')}  queue_position: {job.get('queue_position')}",
                                f"server queue: {health.get('queue')}  gpus: {[g.get('active_model') for g in health.get('gpus', [])]}",
                                "Still working. wait_job, then diagnose again if it fails."])
    error = job.get("error") or {}
    cause, action = guide.explain(error.get("code"), error.get("message"))
    out += [f"error: {error.get('code')}: {error.get('message')}", f"CAUSE:  {cause}", f"ACTION: {action}", ""]

    start = (_local(job["created_at"]) - SLACK) if job.get("created_at") else datetime.min
    gateway = logs.files_newest_first(target.log_dir / "gateway.log")
    if gateway:
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in reversed(gateway)).splitlines()
        shown, hidden = window(text, start, job_id)
        out.append(f"== gateway.log, from the job's creation ({hidden} poll lines hidden)")
        out += [f"  {ln}" for ln in shown] or ["  (no lines about this job in the window)"]
    else:
        out.append(f"== gateway.log: NOT FOUND in {target.log_dir} (the server was not started by `aha-open serve`, or logs elsewhere)")
    for kind in target.worker_kinds():
        worker_log = target.log_dir / f"worker-{kind}.log"
        stage_lines = [ln for f in reversed(logs.files_newest_first(worker_log)) for ln in f.read_text(encoding="utf-8", errors="replace").splitlines() if STAGE.match(ln)]
        out += ["", f"== worker-{kind}.log: last stage reported: " + (stage_lines[-1] if stage_lines else "none (the worker wrote no stage lines)")]
        out.append(logs.read_tail(worker_log, 25))
    return "\n".join(out)
