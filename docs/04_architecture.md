# AHA open architecture

> Status: the gateway is implemented and passes the conformance suite (`conformance/`) with a fake worker and with the real
> Qwen worker (✅ see [05_models.md](05_models.md) §3). The Qwen worker and the `/ui/` page are described there too.
> Gateway ↔ worker uses `subprocess.Popen` + a reader thread (not asyncio subprocess).

## 1. Two kinds of process

```text
                 ┌──────────────────────────── aha-open gateway ────────────────────────────┐
client ──/v1──►  │ HTTP (FastAPI/uvicorn)  ·  auth/token  ·  uploads  ·  job store (SQLite)  │
                 │ per-GPU queue (fair between clients)  ·  model residency management       │
                 └──────────────┬───────────────────────────────────────┬────────────────────┘
                                │ stdio JSON-lines + job folder           │
                 ┌──────────────▼─────────────┐           ┌──────────────▼─────────────┐
                 │ worker qwen-image           │           │ worker minimax-h3 (deferred)│
                 │ own venv (Py 3.13, torch,   │           │ own venv (torch, diffusers  │
                 │ diffusers @80c7ed2)         │           │ @f53d552)                   │
                 └─────────────────────────────┘           └─────────────────────────────┘
```

| Process | Has torch? | Role |
|---|---|---|
| **gateway** | **no** | The whole `/v1` API, authentication, storage, the queue, and deciding which model is on the GPU. Starts fast and always answers `/v1/health`. |
| **worker** (one per model family) | yes | Loads exactly one pipeline, runs one job at a time, reports progress, stops at a step boundary when cancelled. |

Why they are separate: the two models pin two different versions of diffusers/transformers (measured) — one shared
environment would break one of them. A worker is also the unit that can die (OOM, a CUDA error) without taking the gateway down.

## 2. Gateway ↔ worker communication

The gateway starts the worker as a child process and talks to it over **stdin/stdout, one JSON object per line** (no network
port is opened ⇒ no extra network attack surface). Pixels/video do **not** travel through the pipe: the gateway puts inputs in the job folder and the worker
writes its outputs there.

```text
gateway → worker: {"op": "load", "model": "qwen-image-2.1", "variant": "int8"}
worker  → gateway: {"event": "loaded", "vram_peak_mib": 12380}
gateway → worker: {"op": "run", "job_id": "job_…", "dir": "<data>/jobs/job_…", "task": "image.generate", "params": {…}}
worker  → gateway: {"event": "progress", "job_id": "job_…", "step": 12, "total_steps": 40}
gateway → worker: {"op": "cancel", "job_id": "job_…"}
worker  → gateway: {"event": "done", "job_id": "job_…", "outputs": [{"file": "out_0.png", "width": 2048, "height": 2048}]}
                   {"event": "failed", "job_id": "job_…", "code": "gpu_out_of_memory", "message": "…"}
gateway → worker: {"op": "unload"}
```

Cancelling: the worker checks a cancel flag in diffusers' per-step callback (`callback_on_step_end`) and stops at the next step.
A worker that does not answer within a grace period is killed by the gateway and the job is marked `failed`; a result that appears
after the job was `cancelled` is dropped.

## 3. Model residency and the queue

- One queue per GPU. At any moment each GPU has exactly **one** model in VRAM.
- Choosing the next job: round-robin between clients (fair), in submission order within a client. A job for the model that is
  already loaded is preferred unless that would make a client wait longer than `max_wait_for_swap_s` (this cuts the number of
  model swaps, which cost tens of seconds).
- Swapping a model: `unload` the old worker → `load` the new one; the job is in the `loading_model` state meanwhile.
- Limits: `max_queue_per_client`, `max_queue_total`; beyond them ⇒ `429 busy`.
- A failed job is never retried automatically; the client decides.

## 4. Storage

```text
<data_dir>/
  config.toml           configuration (network profile, port, enabled models, GPU)
  clients.sqlite        clients, tokens (hash only), pairing state
  jobs.sqlite           jobs, state, result metadata
  uploads/<upload_id>/  parts + assembled file, deleted after the job used it or when it expires
  jobs/<job_id>/        input (a link to the upload), output, the job's log; deleted after result_ttl_s
  models/               weights downloaded from Hugging Face (by the operator, who accepted the license)
  loras/                <id>.safetensors [+ <id>.json] put there by the operator (`aha-open loras add`); safetensors only, id a-z0-9_-
  tls/                  self-signed certificate for the lan profile
```

A token is stored only as a hash; its value appears exactly once, when it is created/approved.

## 5. Configuration (example)

```toml
profile = "local"            # local | lan | internet
listen = "127.0.0.1:8890"    # lan: name the interface explicitly, e.g. "192.168.1.20:8890"

[auth]
require_token = true
enroll = true                # internet: false by default

[[models]]
id = "qwen-image-2.1"
worker = "qwen_image"
variant = "int8"             # int8 | bf16
gpu = 0

[[models]]
id = "minimax-h3-fl2va"
worker = "minimax_h3"
gpu = 0
enabled = false              # enable after the hardware has been measured
```

## 6. CLI

| Command | What it does |
|---|---|
| `aha-open init --profile local` | Create `data_dir` and a default configuration |
| `aha-open models install qwen-image-2.1 --variant int8 --accept-license qwen-research` | Create the worker venv from its lock file, download the weights; refuses without `--accept-license` |
| `aha-open models list` | Models, state, license |
| `aha-open serve` | Run the gateway |
| `aha-open open` | Open the bundled web page in a browser |
| `aha-open loras add FILE` · `list` · `remove ID` | Manage the LoRA files the server offers |
| `aha-open clients create --name "Laptop An"` | Create a token by hand (printed once) |
| `aha-open clients approve <code>` / `revoke <id>` | Approve a pairing / revoke a client |
| `aha-open tls fingerprint` | Print the certificate fingerprint (lan profile) |
| `aha-open check` | Check a running server end to end from the outside |
| `aha-open tunnel` | Help with the Cloudflare Tunnel setup (internet profile) |
| `aha-open doctor` | Check the GPU, CUDA, disk space, port and configuration |

## 7. Code layout

```text
aha_open/            gateway (never imports torch — a test enforces it)
  app.py  scheduler.py  store.py  auth.py  catalog.py  install.py  loras.py  cli.py ...
  workers/           worker sources (mapped from the top-level workers/ folder when the wheel is built)
  ui/                the bundled web page
workers/
  qwen_image/        its own pyproject + uv.lock, an adapter calling diffusers
  minimax_h3/        (deferred)
  fake/              a fake worker for tests (no torch)
conformance/         the /v1 conformance suite (pytest, real HTTP calls)
schema/              OpenAPI
docs/
```

## 8. Invariants

1. The gateway does not import torch (a test runs in a child process and checks `sys.modules`).
2. There is no route besides `/v1/*`, the static files `/ui/*` (the web page — it only calls `/v1`) and `/admin/*` if an admin page exists (loopback only).
3. No pixel or prompt leaves the machine except in the response to the client that sent it.
4. The output size is exactly the accepted `params`; there is no silent resize.
5. A job that has been `cancelled` never has a result.
