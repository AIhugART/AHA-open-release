"""The tool bodies, as plain functions returning text. `server.py` only registers them with MCP, so every behaviour here is
testable without a protocol and without mocks.

The prefix is the promise (docs/08): `check_*`, `read_*`, `probe_*` never change state; `run_*` says what it will do first."""

from __future__ import annotations

import functools
import json
import re

from . import client, diagnose, evidence, jobs, ledger, logs, ram, sandbox, scenarios
from .targets import TARGETS, Target, TargetError, resolve

WORKER_NAME = re.compile(r"^[a-z0-9_]{1,40}$")


def _with_header(target: Target, body: str) -> str:
    return f"{evidence.header(target)}\n\n{body}"


def _resolved(name: str) -> Target:
    if name not in TARGETS:
        raise TargetError(f"target must be one of {TARGETS}, got {name!r}")
    return resolve(name)


def _guard(fn):
    """A failure the agent can act on is said plainly; it is never swallowed into an empty result."""

    @functools.wraps(fn)
    def run(*args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except TargetError as err:
            return f"ERROR: {err}"

    return run


@_guard
def check_harness() -> str:
    """Is this tool, or the server it points at, running older code than the files on disk? Run it first after any edit."""
    lines = [f"harness: {evidence.harness_state()}", f"loaded digest:  {evidence.LOADED_DIGEST[:16]}", f"current digest: {evidence.package_digest()[:16]}"]
    for name in TARGETS:
        try:
            t = resolve(name)
            lines.append(f"{name}: gateway {evidence.gateway_state(t)}")
        except TargetError as err:
            lines.append(f"{name}: {err}")
    return "\n".join(lines)


@_guard
def check_health(target: str = "sandbox") -> str:
    """GET /v1/health (allowlisted fields)."""
    t = _resolved(target)
    return _with_header(t, json.dumps(client.probe(t, "/v1/health"), indent=2))


@_guard
def probe_v1(path: str, target: str = "sandbox") -> str:
    """Read-only GET of /v1/health, /v1/manifest or /v1/jobs/{id}; only allowlisted fields are returned."""
    t = _resolved(target)
    return _with_header(t, json.dumps(client.probe(t, path), indent=2))


@_guard
def read_gateway_log(target: str = "sandbox", max_lines: int = 100, contains: str = "") -> str:
    """Tail of <data_dir>/logs/gateway.log with path, mtime and age; follows rotated backups."""
    t = _resolved(target)
    return _with_header(t, logs.read_tail(t.log_dir / "gateway.log", max_lines, contains or None))


@_guard
def read_worker_log(worker: str = "fake", target: str = "sandbox", max_lines: int = 100, contains: str = "") -> str:
    """Tail of <data_dir>/logs/worker-<worker>.log (the worker's stderr: stage lines, native crash stacks)."""
    if not WORKER_NAME.match(worker):
        raise TargetError(f"worker must match {WORKER_NAME.pattern}, got {worker!r}")
    t = _resolved(target)
    return _with_header(t, logs.read_tail(t.log_dir / f"worker-{worker}.log", max_lines, contains or None))


@_guard
def full_diagnostics(target: str = "sandbox") -> str:
    """Everything passive in one call: header, health, manifest model states, the end of both logs."""
    t = _resolved(target)
    parts = [
        "== health\n" + json.dumps(client.probe(t, "/v1/health"), indent=2),
        "== manifest\n" + json.dumps(client.probe(t, "/v1/manifest"), indent=2),
        "== gateway.log (last 20)\n" + logs.read_tail(t.log_dir / "gateway.log", 20),
    ]
    for kind in t.worker_kinds():
        parts.append(f"== worker-{kind}.log (last 20)\n" + logs.read_tail(t.log_dir / f"worker-{kind}.log", 20))
    return _with_header(t, "\n\n".join(parts))


@_guard
def run_sandbox_start(fake_step_s: float = 0.02, crash_on_load: bool = False, real_model: bool = False, weights_dir: str = "",
                      allow_gpu: bool = False, allow_low_ram: bool = False, model: str = "qwen-image-2.1", sized_video: bool = False,
                      variant: str = "", memory_limit_gib: float = 0.0) -> str:
    """Start a throwaway gateway on a random loopback port with a temporary data directory. Default: the FAKE worker
    (`sized_video=true` adds a synthetic video model that declares exact sizes, a frame grid and steps, like MiniMax H3).
    `real_model=true` runs a real model from the weights and worker environment already in this checkout: `model` is
    `qwen-image-2.1` (default; INT8), `minimax-h3-fl2va` (video with sound) or `minimax-h3-ref2va` (the same with reference images);
    for MiniMax H3 `variant` picks the build (`mixed` by default, `int8`, `nvfp4`). `memory_limit_gib` is the operator's `[gpu]`
    limit, e.g. 15.5 to imitate a 16 GB card. It needs allow_gpu=true and
    enough free RAM (the model loads on the first job, not at start)."""
    if real_model:
        if not allow_gpu:
            raise TargetError("real_model=true loads a real model on the GPU (Qwen INT8: about 9.5 GiB VRAM at 1024x1024 and 17.6 GiB RAM; "
                              "MiniMax H3: about 17-20 GiB VRAM, measured) - pass allow_gpu=true")
        free = ram.free_ram_gib()
        if not allow_low_ram and (free is None or free < ram.REAL_MODEL_NEEDS_GIB):
            raise TargetError(
                f"free RAM is {'unknown' if free is None else f'{free:.1f} GiB'} but the real model needs about {ram.REAL_MODEL_NEEDS_GIB:.0f} GiB: "
                "close programs and try again, or pass allow_low_ram=true to accept the risk"
            )
        banner = (
            "SIDE EFFECTS: creates a temporary data directory and starts an `aha-open serve` process on a random loopback port with the REAL "
            + ("MiniMax H3 worker. A clip takes minutes (about 5 at 608x352, 22 at 1344x768 on an RTX 3090 Ti, measured) and holds the GPU while it runs. "
               if model in ("minimax-h3-fl2va", "minimax-h3-ref2va") else
               "Qwen INT8 worker. The first job loads the model (about 9.5 GiB VRAM and 17.6 GiB RAM, ~90 s cold, measured). ")
            + "Undo with run_sandbox_stop."
        )
    else:
        banner = (
            "SIDE EFFECTS: creates a temporary data directory, starts an `aha-open serve` process on a random loopback port with the "
            "FAKE worker. No GPU, no model, no network beyond loopback. Undo with run_sandbox_stop."
        )
    info = sandbox.start(fake_step_s=fake_step_s, crash_on_load=crash_on_load, real_model=real_model, weights_dir=weights_dir,
                         model=model, sized_video=sized_video, variant=variant, memory_limit_gib=memory_limit_gib)
    t = resolve("sandbox")
    return f"{banner}\n\n{evidence.header(t)}\n\nsandbox up at {info['base_url']} (gateway pid {info['pid']}, data dir {info['data_dir']})"


@_guard
def run_ui_url() -> str:
    """The URL of the sandbox's web page with its token in the fragment (never sent over the network). SANDBOX ONLY: this prints a
    token, which is acceptable only because it dies with the sandbox; a `live` token is never printed (plan D8b)."""
    t = resolve("sandbox")
    return (
        "SIDE EFFECTS: prints the sandbox's token (it stops working when run_sandbox_stop deletes the sandbox). "
        "Do not take a browser snapshot of the page after the token is in it: read the page with an evaluate call instead.\n\n"
        f"{_with_header(t, t.base_url + '/ui/#token=' + t.token())}"
    )


@_guard
def run_sandbox_stop(keep_data: bool = False) -> str:
    """Stop the sandbox this tool started and delete its temporary data directory."""
    return "SIDE EFFECTS: terminates the sandbox gateway and deletes its marked temporary directory.\n\n" + sandbox.stop(keep_data)


def _acting(target: str, allow_gpu: bool, what: str, allow_low_ram: bool = False) -> tuple[Target, str]:
    """Resolve the target for an action, apply the GPU and RAM gates, and build the SIDE EFFECTS banner that is printed first."""
    t = _resolved(target)
    scenarios.gpu_gate(t, allow_gpu)
    ram.gate(t, allow_low_ram)
    banner = f"SIDE EFFECTS: {what} on target {t.name} (worker: {','.join(t.worker_kinds())}). Nothing is retried automatically."
    return t, banner


def _json_arg(text: str, what: str) -> dict:
    try:
        value = json.loads(text) if text else {}
    except ValueError as err:
        raise TargetError(f"{what} is not valid JSON: {err}") from err
    if not isinstance(value, dict):
        raise TargetError(f"{what} must be a JSON object")
    return value


@_guard
def run_upload(width: int = 64, height: int = 64, target: str = "sandbox") -> str:
    """Upload a synthetic solid-colour PNG (an input for image.edit jobs); returns the upload_id."""
    t, banner = _acting(target, True, "uploads one synthetic PNG (a temporary file on the server)")
    return f"{banner}\n\n{_with_header(t, 'upload_id: ' + jobs.upload(t, jobs.synthetic_png(int(width), int(height))))}"


@_guard
def run_job(body_json: str, target: str = "sandbox", allow_gpu: bool = False, allow_low_ram: bool = False) -> str:
    """POST /v1/jobs with the given JSON body. Returns the job id, or the server's refusal. The prompt is never echoed."""
    t, banner = _acting(target, allow_gpu, "creates one job (a real model load and generation when the worker is not the fake one)", allow_low_ram)
    return f"{banner}\n\n{_with_header(t, json.dumps(jobs.submit(t, _json_arg(body_json, 'body_json')), indent=2))}"


@_guard
def wait_job(job_id: str, target: str = "sandbox", timeout_s: float = 60.0, until: str = "", min_step: int = 0) -> str:
    """Poll a job until it ends (or reaches one of the comma-separated states in `until`, and `min_step` sampling steps). Starts nothing."""
    t = _resolved(target)
    states = tuple(s for s in until.split(",") if s) or jobs.TERMINAL
    return _with_header(t, json.dumps(jobs.wait(t, job_id, timeout_s, states, min_step), indent=2))


@_guard
def run_cancel(job_id: str, target: str = "sandbox") -> str:
    """Cancel a job; a cancelled job never has a result."""
    t, banner = _acting(target, True, f"cancels job {job_id}")
    return f"{banner}\n\n{_with_header(t, json.dumps(jobs.cancel(t, job_id), indent=2))}"


@_guard
def assert_result_image(job_id: str, target: str = "sandbox", index: int = 0) -> str:
    """Download a job's result and judge the pixels: PNG signature, header size = reported size, sha256 and byte count match."""
    t = _resolved(target)
    r = jobs.check_result_image(t, job_id, index)
    ok = r.get("fetched") and all(r[k] for k in ("signature_ok", "ihdr_matches", "sha256_matches", "bytes_match"))
    return _with_header(t, f"{'PASS' if ok else 'FAIL'}\n{json.dumps(r, indent=2)}")


@_guard
def assert_result_video(job_id: str, target: str = "sandbox", index: int = 0) -> str:
    """Download a video result and judge the FILE: its own boxes say the picture size, frame count, frame rate, sound track
    (channels, sample rate) and the disclosure comment; they must match the job's metadata, sha256 and byte count too."""
    t = _resolved(target)
    r = jobs.check_result_video(t, job_id, index)
    keys = ("parsed", "size_matches", "fps_matches", "duration_matches", "audio_matches", "sha256_matches", "bytes_match")
    ok = bool(r.get("fetched")) and all(r.get(k) for k in keys)
    return _with_header(t, f"{'PASS' if ok else 'FAIL'}\n{json.dumps(r, indent=2)}")


@_guard
def list_scenarios() -> str:
    """Every scenario, and every file that failed to load (INVALID, with the reason - never skipped silently)."""
    ok, invalid = scenarios.load_all()
    lines = [f"{sc.id}  [{sc.target}; {', '.join(sc.side_effects)}]  {sc.title}" for sc in ok.values()]
    lines += [f"INVALID  {name}: {reason}" for name, reason in invalid.items()]
    return "\n".join(lines) or "no scenarios"


@_guard
def get_scenario(scenario_id: str) -> str:
    """One scenario as data: preconditions (with the reason each exists) are printed BEFORE the steps."""
    ok, invalid = scenarios.load_all()
    if scenario_id in invalid:
        raise TargetError(f"scenario {scenario_id} is INVALID: {invalid[scenario_id]}")
    if scenario_id not in ok:
        raise TargetError(f"no scenario {scenario_id!r}; have {sorted(ok)}")
    sc = ok[scenario_id]
    pre = [f"precondition {p}: {scenarios.PRECONDITIONS[p]}" for p in sc.preconditions]
    body = json.dumps({"vars": sc.vars, "steps": sc.steps, "assert": sc.asserts}, indent=2)
    return "\n".join([f"{sc.id} - {sc.title}", f"covers: {sc.covers}", f"target: {sc.target}", f"side effects: {sc.side_effects}", *pre, body])


@_guard
def run_scenario(scenario_id: str, target: str = "", overrides_json: str = "", record: bool = True, allow_gpu: bool = False,
                 allow_low_ram: bool = False) -> str:
    """Run a scenario's steps against a target and assert; the verdict says PASS or FAIL per check and carries the evidence level."""
    ok, invalid = scenarios.load_all()
    if scenario_id in invalid:
        raise TargetError(f"scenario {scenario_id} is INVALID: {invalid[scenario_id]}")
    if scenario_id not in ok:
        raise TargetError(f"no scenario {scenario_id!r}; have {sorted(ok)}")
    sc = ok[scenario_id]
    t, banner = _acting(target or sc.target, allow_gpu, f"runs scenario {sc.id} (effects: {', '.join(sc.side_effects)})", allow_low_ram)
    scenarios.worker_gate(sc, t)
    report = scenarios.execute(sc, t, _json_arg(overrides_json, "overrides_json"))
    saved = f"\nrecorded: {ledger.record(sc.id, report).name}" if record else ""
    return f"{banner}\n\n{_with_header(t, scenarios.format_report(sc, report))}{saved}"


@_guard
def diagnose_job(job_id: str, target: str = "sandbox") -> str:
    """Why did this job fail? The cause and the action for its error code, the log lines of the job's time window (poll noise
    removed, its count shown), and the last stage the worker reported. Read only."""
    t = _resolved(target)
    return _with_header(t, diagnose.diagnose(t, job_id))


@_guard
def diff_runs(scenario_id: str) -> str:
    """Compare the last two recorded runs of a scenario: REGRESSION, FIXED, NEW, CHANGED fact."""
    return ledger.diff(scenario_id)


TOOLS = [
    check_harness, check_health, probe_v1, read_gateway_log, read_worker_log, full_diagnostics, run_sandbox_start, run_sandbox_stop,
    run_upload, run_job, wait_job, run_cancel, assert_result_image, assert_result_video, list_scenarios, get_scenario, run_scenario, diff_runs,
    diagnose_job, run_ui_url,
]
