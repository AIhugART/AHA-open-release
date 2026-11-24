"""Record runs and diff them, so a regression shows up as a diff instead of a memory (docs/08).

A record holds verdicts and a few facts. It never holds a token, and a prompt only as a short hash."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import targets

SCENARIO_ID = re.compile(r"^[a-z0-9_]{1,80}$")


def ledger_dir(scenario_id: str) -> Path:
    if not SCENARIO_ID.match(scenario_id):
        raise targets.TargetError(f"scenario id must match {SCENARIO_ID.pattern}")
    return targets.STATE_DIR / "ledger" / scenario_id


def record(scenario_id: str, report: dict) -> Path:
    folder = ledger_dir(scenario_id)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = folder / f"{stamp}_{len(list(folder.glob('*.json'))):04d}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def last_two(scenario_id: str) -> list[dict]:
    files = sorted(ledger_dir(scenario_id).glob("*.json"))[-2:]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def diff(scenario_id: str) -> str:
    runs = last_two(scenario_id)
    if len(runs) < 2:
        return f"{scenario_id}: {len(runs)} recorded run(s); a diff needs two."
    before, after = runs
    lines = [f"{scenario_id}: {before['at']} ({before['verdict']})  ->  {after['at']} ({after['verdict']})"]
    old = {c["path"] + " " + c["op"]: c for c in before["checks"]}
    for c in after["checks"]:
        key = c["path"] + " " + c["op"]
        prev = old.get(key)
        if prev is None:
            lines.append(f"NEW        {key}: {'ok' if c['ok'] else 'FAILED'} (actual {c['actual']!r})")
        elif prev["ok"] and not c["ok"]:
            lines.append(f"REGRESSION {key}: expected {c['expected']!r}, was {prev['actual']!r}, now {c['actual']!r}")
        elif not prev["ok"] and c["ok"]:
            lines.append(f"FIXED      {key}: now {c['actual']!r}")
    for name in sorted(set(before["facts"]) | set(after["facts"])):
        if before["facts"].get(name) != after["facts"].get(name):
            lines.append(f"CHANGED    fact {name}: {before['facts'].get(name)!r} -> {after['facts'].get(name)!r}")
    if len(lines) == 1:
        lines.append("no difference")
    return "\n".join(lines)
