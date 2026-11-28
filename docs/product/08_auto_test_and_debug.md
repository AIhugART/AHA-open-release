# Auto-test and auto-debug: `aha-open-mcp`

> **Status: partly implemented.** ✅ Run for real, against a real `aha-open serve` process and over the real MCP protocol:
> observation tools, the throwaway sandbox, jobs, scenarios, assertions, recorded runs, diagnosis of a failed job
> (section 8). ✅ The `live` target and the real-model sandbox are built and their gates tested, and the real Qwen INT8 model ran
> through a scenario twice with byte-identical results (see [the playbook](../../aha_open_mcp/playbooks/ui_e2e.md)).
> Code: [`aha_open_mcp/`](../../aha_open_mcp/). The reasons behind each choice are in the project's internal plan.

## 1. What it is

`aha-open-mcp` is an [MCP](https://modelcontextprotocol.io) server that lets a coding agent **drive, observe and judge** a
running AHA open server end to end:

- **drive** — submit uploads and jobs, cancel them, over the public `/v1` API like any client;
- **observe** — read the gateway and worker logs, the server's process state and the result files, always with their path
  and age;
- **judge** — run declared scenarios and say PASS/FAIL with a reason, record runs and show a regression as a diff;
- **diagnose** — for a failed job, name the cause and the action to take, with the log lines from that job's time window.

It complements, and does not replace, the conformance suite in `conformance/`:

| | `conformance/` (pytest) | `aha-open-mcp` |
|---|---|---|
| Question | Does the gateway follow the `/v1` specification? | Does *this* server, with a *real* model, produce the right result — and if not, why? |
| Worker | fake (no torch) | fake in a sandbox, or the real model |
| Runs | in CI, on every change | when an agent or a developer asks |
| Evidence level | 🧪 | 🧪 with the fake worker, ✅ with a real model |

### The user's machine is the one place `aha-open-mcp` cannot reach

When the failure is on someone else's machine, the evidence is the **support report** (`aha-open report`, or the web page's *Report* entry; see
[04_architecture.md](04_architecture.md) section 6.2). Read it in this order: the *Code* line (is the process the code on disk?), the age of each log (a log older
than the failure is a hypothesis), then the log ends. The report is built from named fields and redacted; it holds no token and no prompt.

## 2. Boundaries

- It is a **client**. It talks to the server only over `/v1` and reads only the files an operator can read in the data
  directory. It adds no route to the server.
- It never prints a token. It reads tokens from files and copies an **allowlist** of fields into its output.
- It never retries a failed job by itself.
- It sends nothing off the machine.

## 3. Two targets

| Target | What it is | When |
|---|---|---|
| `sandbox` (default) | A throwaway gateway the tool starts itself: temporary data directory, random loopback port, the fake worker unless a real one is asked for | Development, breaking code on purpose to prove a check catches it |
| `live` | The operator's running server | Only when asked explicitly (`target="live"`); a job on a real model also needs `allow_gpu=true` |

## 4. Tool contract — the prefix is the promise

| Prefix | Promise |
|---|---|
| `check_*`, `read_*`, `probe_*`, `list_*`, `get_*`, `diagnose_*` | Read only. Never changes state |
| `assert_*`, `record_*`, `diff_*` | Turn "it ran" into "it passed"; `record_*` writes only to the tool's own `runs/` folder |
| `run_*`, `wait_*` | Act. Every `run_*` prints a `SIDE EFFECTS:` block **before** it acts (GPU time, model load, RAM) |

Every output starts with a header: machine, target, worker kind, evidence level, and whether the tool itself
(`harness`) or the server (`gateway`) is running code older than the files on disk (`STALE`).

## 5. Scenarios are data

A scenario is a YAML file in `aha_open_mcp/scenarios/`; adding a check means adding a file, not editing Python.

```yaml
id: generate_exact_size          # must equal the file name
title: "An image job ends with exactly the size asked for, and the downloaded pixels match the job's metadata"
covers: "conformance/test_v1_core.py::test_image_job_end_to_end_with_exact_size_and_range"  # a real test or doc heading
target: sandbox                  # sandbox | live
preconditions: []                # named environment facts, each with its reason (get_scenario prints them first)
side_effects: [creates_job]      # allowlist; an unknown value makes the scenario INVALID
vars: {prompt: "a red square", seed: 1}   # overridable per run: run_scenario(overrides_json=...)
steps:                           # run_upload | run_job | wait_job | run_cancel | probe_v1 | check_result_image
  - tool: run_job
    args: {body: {task: image.generate, model: example-image, prompt: "${prompt}", params: {seed: "${seed}", width: 320, height: 180}}}
  - tool: wait_job
  - tool: check_result_image
assert:                          # over what the steps produced: submit, job_id, upload_id, job, probe, image
  - {path: job.status, op: eq, value: succeeded}
  - {path: image.ihdr_matches, op: eq, value: true}
```

A broken scenario file is listed as `INVALID` with the reason; it is never skipped silently.

## 6. Server-side support it relies on

Two additions to the gateway (planned; they do not change the `/v1` API):

| File | Content | Why |
|---|---|---|
| `<data_dir>/logs/gateway.log` | The gateway log, rotated by size, in addition to stdout | Today the gateway logs only to the terminal that started it |
| `<data_dir>/run/gateway.json` | `pid`, `started_at`, `version`, a digest of the gateway source | Lets a passive observer tell "the product is wrong" from "the process runs old code" |

Neither file contains a token or a prompt; a negative test searches for sample values and expects zero matches.

## 7. Registering it

```bash
claude mcp add aha-open-mcp -- uv run --project <path-to>/aha_open_mcp aha-open-mcp
```

The tool list is fixed when the agent session starts: restart the session after registering or updating the tool.
Its environment is its own (`aha_open_mcp/pyproject.toml`, `uv.lock`, MCP SDK `>=2,<3`): the gateway never imports it
(a test scans the gateway's imports).

## 8. What exists today

| Tool | Does |
|---|---|
| `check_harness` | Is the tool itself (`STALE HARNESS`) or the server (`STALE GATEWAY`) running older code than the files on disk |
| `check_health`, `probe_v1` | Read-only GET of `/v1/health`, `/v1/manifest`, `/v1/jobs/{id}`; only allowlisted fields are copied; any other path is refused |
| `read_gateway_log`, `read_worker_log` | Tail with path, last-modified time and age; follows rotated files; `NOT FOUND` is said, never an empty tail |
| `full_diagnostics` | Header, health, manifest, and the end of the logs in one call |
| `run_upload`, `run_job`, `wait_job`, `run_cancel` | Act through `/v1` like any client (`run_upload` takes `rgb` for a plain-colour reference image; one `wait_job` waits at most 600 s, so a longer job chains two steps; `min_step` waits until that sampling step is reported, so a cancel lands in the middle of sampling). Each prints `SIDE EFFECTS:` first; a target with a real model is refused unless `allow_gpu=true`; nothing is retried; a prompt is never echoed |
| `assert_result_image` | Downloads the result and judges the pixels: PNG signature, header size equals the reported size, sha256 and byte count match |
| `assert_result_video` | Downloads the result and judges the FILE by reading its own mp4 boxes (no codec library): picture size, frame count, frame rate, duration, audio track (channels, sample rate), the label comment; PASS needs every comparison with what the job reported (size, fps ±0.01, duration ±0.05 s, audio present or absent, sha256, byte count). A cut or non-mp4 file is `parsed: false` with the reason, never an exception |
| `list_scenarios`, `get_scenario`, `run_scenario`, `diff_runs` | Scenarios are data (section 5); a broken file is listed `INVALID` with its reason. `run_scenario` runs, asserts and records (verdicts, a few facts, a 12-hex hash of the prompt - never the prompt); `diff_runs` shows `REGRESSION`, `FIXED`, `NEW` and `CHANGED fact` between the last two runs |
| `diagnose_job` | For a failed job: the cause and the action for its error code (a table guarded in both directions against the gateway's source and this specification), the gateway log lines of the job's time window with the pollers' own lines removed and counted, and the last stage line the worker wrote. `generation_failed` is split by the gateway's own message (out of RAM, native crash, wrong output size, timeout...) |
| `run_ui_url` | Sandbox only: the web page's address with the sandbox token in the fragment (it dies with the sandbox). A `live` token is never printed |
| `run_sandbox_start`, `run_sandbox_stop` | A real `aha-open serve` (`sized_video=true` adds the fake model `example-video-sized` that enforces exact video sizes, for the scenarios `cancel_running_video` and `video_size_mismatch_rejected`; `model=minimax-h3-fl2va` or `model=minimax-h3-ref2va` with `real_model` points at the real worker, weights from `AHA_OPEN_H3_WEIGHTS` or `<data>/models/minimax-h3-fl2va/mixed`, and `variant=int8|mixed|nvfp4` picks the build (a build that is unknown or not on disk is refused with the command that fetches it); `memory_limit_gib` sets the operator's `[gpu]` limit, e.g. 15.5 to imitate a 16 GB card; `crash_on_load` makes the fake worker die natively while loading, the shape of the 2026-11-23 Qwen crash) on a random loopback port with the fake worker; stopped when the MCP server exits; only a folder the tool created and marked is ever deleted |

`real_model=true` runs the real Qwen INT8 worker from the weights and worker environment already in the checkout; (Qwen: only the `int8` variant; the sandbox chooses a build for MiniMax H3 only. The ConvRot variants were run for real through the `live` target on 2026-11-25 — `POST /admin/models/{id}/use` to switch, then `run_job` / `wait_job` / `assert_result_image` — with the operator's configuration saved first and restored byte for byte afterwards; see [05_models.md](05_models.md) §3.4.) it needs
`allow_gpu=true` and at least 20 GiB of free RAM (the measured peak is 17.6 GiB; a load with only a few GiB free once died with a native
crash), otherwise it refuses and says the number. `allow_low_ram=true` is the person's explicit acceptance of that risk. A model that is
already loaded is not re-checked (its RAM is why free RAM is low). A scenario written for the fake worker's marker words is refused on a
real model, and the reverse.

Every output starts with the header of section 4. Two measured facts shaped the design: on Windows the virtual
environment's `python.exe` is a launcher, so the gateway's real pid comes from its own `run/gateway.json`; and the MCP SDK's client
tolerates stray lines on the server's stdout, so a test reads the raw bytes instead.
