"""Scenarios are DATA (`scenarios/*.yaml`): adding a check means adding a file. A broken file is listed as INVALID with its
reason; it is never skipped silently (a loader that quietly `continue`s past a broken file is how a check stops existing
without anyone noticing).

A scenario runs a few steps against a target and then asserts over the CONTEXT the steps filled in: `job`, `submit`,
`probe`, `image`, `video`, `job_id`, `upload_id`."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import client, evidence, jobs
from .targets import TARGETS, Target, TargetError

REPO = Path(__file__).resolve().parents[3]
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
ID_RE = re.compile(r"^[a-z0-9_]{1,80}$")

# Allowlist, not blocklist: the SIDE EFFECTS banner is only trustworthy if it is complete, so an effect nobody named is refused.
SIDE_EFFECTS = {"none", "reads_only", "creates_job", "uploads_file", "cancels_job", "uses_gpu", "loads_model"}

# Named environment facts a scenario depends on (measured or read from code, with the reason shown by get_scenario).
PRECONDITIONS = {
    "fake_worker_markers": "the prompt contains a marker word that only the FAKE worker understands (workers/fake/worker.py); a real model would just draw it",
    "real_model_installed": "the model is installed and its weights are on disk (`aha-open models list` says ready); loading it takes GPU and RAM",
    "sized_video_sandbox": "the sandbox was started with sized_video=true, which adds the synthetic model example-video-sized (exact sizes, a frame grid, steps: the shape of MiniMax H3's declaration)",
}

# What each step writes into the context (used to check, before anything runs, that every assertion reads something that exists).
PRODUCES = {
    "run_upload": {"upload_id"},
    "run_job": {"submit", "job_id"},
    "wait_job": {"job"},
    "run_cancel": {"job"},
    "probe_v1": {"probe"},
    "check_result_image": {"image"},
    "check_result_video": {"video"},
}
OPS = ("eq", "ne", "gt", "ge", "lt", "le", "in", "exists", "absent", "len_eq")
VAR = re.compile(r"\$\{([a-z_][a-z0-9_]*)\}")


class ScenarioError(Exception):
    pass


@dataclass
class Scenario:
    id: str
    title: str
    covers: str
    target: str
    preconditions: list[str]
    side_effects: list[str]
    requires_worker: str | None
    vars: dict
    steps: list[dict]
    asserts: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ loading and validation
def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s", "-", text)


def resolve_covers(covers: str) -> None:
    """`covers` POINTS at something real (a test, or a doc heading) instead of copying its content; a dead pointer is an error."""
    if "::" in covers:
        rel, name = covers.split("::", 1)
        path = REPO / rel
        if not path.is_file():
            raise ScenarioError(f"covers: {rel} does not exist")
        if not re.search(rf"^\s*(async\s+)?def {re.escape(name)}\(", path.read_text(encoding="utf-8"), re.M):
            raise ScenarioError(f"covers: no test named {name} in {rel}")
    elif "#" in covers:
        rel, anchor = covers.split("#", 1)
        path = REPO / rel
        if not path.is_file():
            raise ScenarioError(f"covers: {rel} does not exist")
        anchors = {slug(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", path.read_text(encoding="utf-8"), re.M)}
        if anchor not in anchors:
            raise ScenarioError(f"covers: no heading with anchor #{anchor} in {rel}")
    else:
        raise ScenarioError("covers must be 'path::test_name' or 'path#heading-anchor'")


def _names_in(value) -> set[str]:
    if isinstance(value, str):
        return set(VAR.findall(value))
    if isinstance(value, dict):
        return set().union(*(_names_in(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_names_in(v) for v in value)) if value else set()
    return set()


def parse(stem: str, raw: object) -> Scenario:
    if not isinstance(raw, dict):
        raise ScenarioError("the file is not a mapping")
    missing = [k for k in ("id", "title", "covers", "target", "side_effects", "steps", "assert") if k not in raw]
    if missing:
        raise ScenarioError(f"missing keys: {missing}")
    if raw["id"] != stem or not ID_RE.match(str(raw["id"])):
        raise ScenarioError(f"id {raw['id']!r} must equal the file name {stem!r} and match {ID_RE.pattern}")
    if raw["target"] not in TARGETS:
        raise ScenarioError(f"target must be one of {TARGETS}")
    bad = [e for e in raw["side_effects"] if e not in SIDE_EFFECTS]
    if bad:
        raise ScenarioError(f"unknown side_effects {bad}; allowed: {sorted(SIDE_EFFECTS)}")
    pre = raw.get("preconditions") or []
    bad = [p for p in pre if p not in PRECONDITIONS]
    if bad:
        raise ScenarioError(f"unknown preconditions {bad}; allowed: {sorted(PRECONDITIONS)}")
    variables = raw.get("vars") or {}
    required = raw.get("requires_worker") or ("fake" if "fake_worker_markers" in pre else None)
    if required is not None and not isinstance(required, str):
        raise ScenarioError("requires_worker must be a worker name such as fake or qwen_image")
    known = set(variables)
    for i, step in enumerate(raw["steps"], 1):
        if not isinstance(step, dict) or step.get("tool") not in PRODUCES:
            raise ScenarioError(f"step {i}: tool must be one of {sorted(PRODUCES)}")
        unknown = _names_in(step.get("args") or {}) - known
        if unknown:
            raise ScenarioError(f"step {i}: ${{{', '.join(sorted(unknown))}}} is neither a var nor produced by an earlier step")
        known |= PRODUCES[step["tool"]]
    for a in raw["assert"]:
        if not isinstance(a, dict) or a.get("op") not in OPS or "path" not in a:
            raise ScenarioError(f"assert {a!r}: needs path and op in {OPS}")
        root = re.split(r"[.\[]", a["path"], maxsplit=1)[0]
        if root not in known:
            raise ScenarioError(f"assert path {a['path']!r}: {root!r} is not produced by any step (have {sorted(known)})")
        if a["op"] not in ("exists", "absent") and "value" not in a:
            raise ScenarioError(f"assert {a['path']!r}: op {a['op']} needs a value")
    resolve_covers(str(raw["covers"]))
    return Scenario(
        id=raw["id"], title=str(raw["title"]), covers=str(raw["covers"]), target=raw["target"], preconditions=list(pre),
        side_effects=list(raw["side_effects"]), requires_worker=required, vars=dict(variables), steps=list(raw["steps"]), asserts=list(raw["assert"]),
    )


def load_all(folder: Path | None = None) -> tuple[dict[str, Scenario], dict[str, str]]:
    ok: dict[str, Scenario] = {}
    invalid: dict[str, str] = {}
    for path in sorted((folder or SCENARIO_DIR).glob("*.yaml")):
        try:
            ok[path.stem] = parse(path.stem, yaml.safe_load(path.read_text(encoding="utf-8")))
        except (ScenarioError, yaml.YAMLError) as err:
            invalid[path.stem] = str(err)
    return ok, invalid


# ------------------------------------------------------------------------------------------------ evaluation
def substitute(value, scope: dict):
    if isinstance(value, str):
        whole = VAR.fullmatch(value)
        if whole:
            return scope[whole.group(1)]
        return VAR.sub(lambda m: str(scope[m.group(1)]), value)
    if isinstance(value, dict):
        return {k: substitute(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, scope) for v in value]
    return value


def lookup(ctx: dict, path: str) -> tuple[bool, object]:
    cur: object = ctx
    for token in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        try:
            if token.startswith("["):
                cur = cur[int(token[1:-1])]  # type: ignore[index]
            else:
                cur = cur[token]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return False, None
    return True, cur


def check(ctx: dict, assertion: dict) -> dict:
    found, actual = lookup(ctx, assertion["path"])
    op, expected = assertion["op"], assertion.get("value")
    if op == "exists":
        ok = found
    elif op == "absent":
        ok = not found
    elif not found:
        ok = False
    elif op == "eq":
        ok = actual == expected
    elif op == "ne":
        ok = actual != expected
    elif op == "in":
        ok = actual in expected
    elif op == "len_eq":
        ok = hasattr(actual, "__len__") and len(actual) == expected  # type: ignore[arg-type]
    else:
        try:
            ok = {"gt": actual > expected, "ge": actual >= expected, "lt": actual < expected, "le": actual <= expected}[op]  # type: ignore[operator]
        except TypeError:
            ok = False
    return {"path": assertion["path"], "op": op, "expected": expected, "actual": actual if found else "<missing>", "ok": bool(ok)}


# ------------------------------------------------------------------------------------------------ execution
def worker_gate(sc: Scenario, target: Target) -> None:
    """A scenario written for the fake worker's marker words means nothing on a real model (it would just draw the words), and
    a scenario for a real model cannot run on the fake one: refuse and say which."""
    if sc.requires_worker and sc.requires_worker not in target.worker_kinds():
        raise TargetError(f"scenario {sc.id} needs worker {sc.requires_worker!r} but target {target.name} has {target.worker_kinds()}")


def gpu_gate(target: Target, allow_gpu: bool) -> None:
    real = [k for k in target.worker_kinds() if k != "fake"]
    if real and not allow_gpu:
        raise TargetError(
            f"target {target.name} has real model worker(s) {real}: this loads a model (GPU and several GiB of RAM; the Qwen INT8 "
            "worker measured 17.6 GiB RAM on a 31.7 GiB machine) - pass allow_gpu=true if that is what you want"
        )


def run_step(tool: str, args: dict, ctx: dict, target: Target) -> str:
    """Run one step, update the context, return a one-line note. Raises TargetError when the step cannot be run at all."""

    def job_id() -> str:
        jid = args.get("job_id") or ctx.get("job_id")
        if not jid:
            raise TargetError("no job_id in the context: the submit was refused or no run_job step came before")
        return str(jid)

    if tool == "run_upload":
        w, h = args["synthetic_png"]
        ctx["upload_id"] = jobs.upload(target, jobs.synthetic_png(int(w), int(h), tuple(args.get("rgb", (0x20, 0x40, 0xA0)))), args.get("media_type", "image/png"))
        return f"upload_id={ctx['upload_id']}"
    if tool == "run_job":
        ctx["submit"] = jobs.submit(target, args["body"])
        if "job_id" in ctx["submit"]:
            ctx["job_id"] = ctx["submit"]["job_id"]
        else:
            ctx.pop("job_id", None)
        return f"HTTP {ctx['submit']['http_status']}" + (f" job_id={ctx['job_id']}" if "job_id" in ctx else f" refused: {ctx['submit']['error']['code']}")
    if tool == "wait_job":
        until = tuple(args["until"]) if args.get("until") else jobs.TERMINAL
        ctx["job"] = jobs.wait(target, job_id(), float(args.get("timeout_s", 60)), until, int(args.get("min_step", 0)))
        return f"status={ctx['job']['status']}" + (" (TIMED OUT)" if ctx["job"].get("timed_out") else "")
    if tool == "run_cancel":
        ctx["job"] = jobs.cancel(target, job_id())
        return f"status={ctx['job']['status']}"
    if tool == "probe_v1":
        ctx["probe"] = client.probe(target, args["path"], with_token=bool(args.get("with_token", True)))
        return f"HTTP {ctx['probe']['http_status']}"
    if tool == "check_result_image":
        ctx["image"] = jobs.check_result_image(target, job_id(), int(args.get("index", 0)))
        return "fetched" if ctx["image"].get("fetched") else f"not fetched: HTTP {ctx['image'].get('http_status')}"
    if tool == "check_result_video":
        ctx["video"] = jobs.check_result_video(target, job_id(), int(args.get("index", 0)))
        v = ctx["video"]
        if not v.get("fetched"):
            return f"not fetched: HTTP {v.get('http_status')}"
        return f"{v['width']}x{v['height']}, {v['frames']} frames, {v['fps']} fps" if v.get("has_video_track") else f"not a usable mp4: {v.get('parse_error')}"
    raise TargetError(f"unknown step tool {tool}")


def execute(sc: Scenario, target: Target, overrides: dict | None = None) -> dict:
    unknown = set(overrides or {}) - set(sc.vars)
    if unknown:
        raise TargetError(f"unknown override(s) {sorted(unknown)}; this scenario's vars are {sorted(sc.vars)}")
    variables = {**sc.vars, **(overrides or {})}
    ctx: dict = {}
    steps: list[dict] = []
    for i, step in enumerate(sc.steps, 1):
        try:
            args = substitute(step.get("args") or {}, {**variables, **ctx})
            note = run_step(step["tool"], args, ctx, target)
            steps.append({"n": i, "tool": step["tool"], "ok": True, "note": note})
        except TargetError as err:
            steps.append({"n": i, "tool": step["tool"], "ok": False, "note": str(err)})
            break  # later steps would run on a context that does not exist
    checks = [check(ctx, a) for a in sc.asserts]
    facts = {}
    for name, path in (("result_sha256", "image.sha256"), ("video_sha256", "video.sha256"), ("job_status", "job.status"), ("error_code", "job.error.code")):
        found, value = lookup(ctx, path)
        if found:
            facts[name] = value
    prompt = (next((s.get("args", {}).get("body", {}).get("prompt") for s in sc.steps if s["tool"] == "run_job"), "") or "")
    prompt = substitute(prompt, variables)
    passed = all(s["ok"] for s in steps) and all(c["ok"] for c in checks)
    return {
        "scenario_id": sc.id, "target": target.name, "evidence": target.evidence(), "worker": target.worker_kinds(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "harness": evidence.harness_state(),
        "gateway": evidence.gateway_state(target), "steps": steps, "checks": checks, "facts": facts,
        "prompt_sha256_12": hashlib.sha256(str(prompt).encode()).hexdigest()[:12] if prompt else None,
        "verdict": "PASS" if passed else "FAIL",
    }


def format_report(sc: Scenario, report: dict) -> str:
    lines = [f"scenario: {sc.id} - {sc.title}", f"covers: {sc.covers}"]
    lines += [f"  step {s['n']} {s['tool']}: {'ok' if s['ok'] else 'FAILED'} - {s['note']}" for s in report["steps"]]
    lines += [f"  {'PASS' if c['ok'] else 'FAIL'}  {c['path']} {c['op']} {c['expected']!r}  (actual {c['actual']!r})" for c in report["checks"]]
    n_ok = sum(c["ok"] for c in report["checks"])
    lines.append(f"VERDICT: {report['verdict']} ({n_ok}/{len(report['checks'])} checks) evidence {report['evidence']}")
    return "\n".join(lines)
