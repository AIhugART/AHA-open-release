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
gateway → worker: {"op": "load", "model": "qwen-image-2.1", "variant": "int8-convrot", "weights_dir": "<data>/models/qwen-image-2.1/convrot", "gpu": 0, "files": ["transformers/qwen_image_2.1_bf16_int8_convrot.safetensors"]}
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
  uploads/<upload_id>/  parts + assembled file, deleted when it expires (upload_ttl_s after creation)
  jobs/<job_id>/        the job's own hard links to its inputs (a copy where there are none; removed when the job ends), output;
                        deleted after result_ttl_s
  models/               weights downloaded from Hugging Face (by the operator, who accepted the license)
  loras/                <id>.safetensors [+ <id>.json] put there by the operator (`aha-open loras add`); safetensors only, id a-z0-9_-
  tmp/                  a LoRA the web page is receiving; renamed into loras/ or deleted at once, never kept
  tls/                  self-signed certificate for the lan profile
  logs/                 gateway.log (rotated by size; the same lines as the terminal) and worker-<name>.log; never a token or a prompt
  run/gateway.json      written by `aha-open serve`: pid, started_at, version, source_digest (removed on a clean exit; a crash leaves it, naming a dead pid)
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
variant = "int8"             # int8 is the only variant Qwen offers
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
| `aha-open models install qwen-image-2.1 --accept-license qwen-research` | Create the worker venv from its lock file, download the weights; refuses without `--accept-license` |
| `aha-open models install qwen-image-2.1 --variant int8-convrot ...` | Pick a variant (default: the best one that fits this machine). A second variant is added next to the first and never switches it |
| `aha-open models list` | Models, state, license, and for each model every variant: installed / in use / not installed, disk, GPU fit |
| `aha-open models use qwen-image-2.1 mixed-p015` | Select an installed variant (restart the server to load it) |
| `aha-open models remove qwen-image-2.1 mixed-p015 [--include-shared] [--yes]` | Delete one variant's files; shows what it would delete unless `--yes`; shared files stay while another variant needs them |
| `aha-open serve` | Run the gateway |
| `aha-open start` | Local profile: start the gateway in the background if it is not running (it asks the port first, so a second click never starts a second one), wait for `/v1/health`, open the page. A failure says why from the gateway's own output (`logs/start.log`); `--gui` also shows it in a box, for the Start-menu shortcut that has no console |
| `aha-open stop` | Stop the gateway `start` started, and what it started (a download in progress, a worker). It uses the pid the gateway wrote itself (`run/gateway.json`), waits until the process is really gone, and leaves a foreign program on the port alone |
| `aha-open open` | Open the bundled web page in a browser |
| `aha-open loras add FILE` · `list` · `remove ID` | Manage the LoRA files the server offers |
| `/ui/` ▸ *Settings ▸ Manage Models ▸ Manage LoRA files* | The same add/remove from the web page, on the machine that runs AHA open (local profile only); **Refresh** re-reads the list |
| `/ui/` ▸ *Settings ▸ Download a Model* | The same `models install` from the web page (local profile only), see section 6.1; the first download can pick a variant (default: the recommended one, one tick and one click) |
| `/ui/` ▸ *Settings ▸ Manage Models* | Every variant of an installed model: size, measured GPU memory and time, fit on this machine, the publisher's quality note; **Use this**, **Download**, **Delete** (asks first and says what it frees; the shared files are offered separately when the last variant goes) and **Cancel download** — the `/admin/models` routes of section 6.1 |
| `aha-open clients create --name "Laptop An"` | Create a token by hand (printed once) |
| `aha-open clients approve <code>` / `revoke <id>` | Approve a pairing / revoke a client |
| `aha-open tls fingerprint` | Print the certificate fingerprint (lan profile) |
| `aha-open check` | Check a running server end to end from the outside |
| `aha-open tunnel` | Help with the Cloudflare Tunnel setup (internet profile) |
| `aha-open doctor` | Check the GPU, CUDA, disk space, port and configuration |
| `aha-open report [--out FILE] [--note TEXT]` | Print (or save) the support report, section 6.2. Works with the server stopped |

### 6.2 The support report (`GET /admin/report`, `aha-open report`)

One text the person copies or saves and sends to whoever supports their installation (the web page's *Report* entry, see [10_web_ui.md](10_web_ui.md)).
Nothing is sent by AHA open: it builds a string (`aha_open/report.py`), and the person reads it before sending it anywhere.

| Part | Source | Notes |
|---|---|---|
| Machine | OS, Python, **time zone**, GPU name and memory (`nvidia-smi`), RAM, free disk where the models are stored | named fields only. The times inside the logs are local time and the header's are UTC; the time zone line says which is which |
| Gateway | version on disk, profile, listen address, whether a token is required, and `run/gateway.json` (pid, version, start time) | says **DIFFERENT** when the digest recorded by the running gateway is not the digest of the files on disk: a process started before an edit still runs the old code |
| Models | for each configured model its `state` and the reason | the data folder is written `<data>` |
| Logs | the end of `logs/gateway.log` (across its rotated backups), every `logs/worker-*.log` and `logs/start.log` | each with its size, **last written time and age**; the budget (256 KiB by default) is shared by water-filling, so a small log is whole and no log is starved by a large one; every cut starts at a whole line. A line that repeats (a client polling a job every 60 ms filled the last 128 KiB of a real log) is shown once with its count, so the line that explains a failure is not pushed out |

What it never reads: `config.toml`, the token files, the database, the uploaded or generated files. What it removes from what it does read: the data folder
(`<data>`), the home folder (`~`) and the name in any other `Users\<name>` / `/home/<name>` path, and anything shaped like a secret (`aho_…`, `Bearer …`,
`hf_…`, `token=…`, `password: …`). The gateway does not log a token or a prompt (a test searches for sample values in the real log), so the removal is a second
layer, not the first. Building the report writes nothing. The route is guarded like the other `/admin/*` routes (section 2 rule 2; `404` outside the local profile).

### 6.1 Installing a model from the web page (`/admin/models/*`)

For the person at the machine, on the same terms as the LoRA manager (local profile, loopback peer and name, token; anything else answers `404`).
It is **not** part of `/v1`: installing a model is the operator's decision, not a client's, so the neutral API does not carry it.

| Request | Answer |
|---|---|
| `GET /admin/models` | The models that can be installed here: `state`, the licence to show (`license.key`, `name`, `commercial_use`, `notes`, `url`, `excluded_territories`), `requires_territory_confirmation`, `disk_gib` (GB **still to download**), `variants` (every variant the model offers: `name`, `label`, `disk_gib`, `download_gib` = what is still to fetch, `own_gib` = what deleting it frees (its own transformer file for a ConvRot build; the shared text encoder and VAE stay), `vram_gib`, `ram_gib`, `seconds`, `loras`, `quality_note`, `fit`, `installed`, `selected`, `recommended`), a `blocker` (`{code, message}` or `null`) and the current `install` status. `engine_missing: true` means the weights are here and the environment that runs them is not: the page offers *Set up engine*, which is the `install` call below with an installed variant (nothing already on disk is downloaded; the files are checked first, phase `verifying`). |
| `POST /admin/models/{id}/install` with `{"accept_license": "<license.key>"}` (and `"confirm_territory": true` when the listing says `requires_territory_confirmation`) | `202` + the status. The key must be the exact one listed, and `confirm_territory` must be the boolean `true` (a string or a number is not enough); `"variant": "<name>"` picks one (default: the best that fits this machine; not a name the model offers is `400`). Anything else is `400` and nothing starts |
| `GET /admin/models/{id}/install` | `{state: idle \| running \| succeeded \| failed \| cancelled, phase: preparing \| downloading \| verifying \| creating_environment \| finishing, downloaded_bytes, expected_bytes, error: {code, message} \| null, started_at, finished_at}` |
| `POST /admin/models/{id}/use` with `{"variant": "<name>"}` | Select an installed variant; the worker that holds the old one is released and the next job loads the new one. `409 variant_not_installed`, `409 model_busy` (a job of this model is waiting, loading or running), `409 install_running`; an unknown name is `400`. The response is the model's entry |
| `DELETE /admin/models/{id}/variants/{variant}[?include_shared=true]` | Delete one variant's files (the entry + a `message` saying what was freed and what was kept, and `shared_offer: true` when the last variant went and its shared files are still there). Only inside the models folder; files another installed variant needs stay, and the shared ones go only with the last variant **and** `include_shared=true`. `409 model_busy`/`install_running` as above, `404` when it is not installed, `400` for an unknown name |
| `DELETE /admin/models/{id}/install` | `204` and the whole process tree is killed; a download resumes next time. `404` when nothing is running |

Models offered: Qwen-Image-2.1, MiniMax H3 (text/frames to video) and MiniMax H3 (references to video). The page draws one card each. A licence that
excludes territories (MiniMax H3: EU, GB, KR, US) gets **a second checkbox of its own** — "I confirm that I am not in …" — and the button needs both
ticked; the server enforces the same rule, so a script that skips the page is refused too. The command-line `--confirm-territory` is the
equivalent. `disk_gib` counts only the files that are not on disk yet: the reference model shares the text model's folder (the encoder and the VAEs are
the same files), so with the text model installed it asks for about 15 GB, not 61. The variant and the "this GPU is too small" sentence come
from the model's own rule (`recommend_variant_for`, the one the command line uses): a 12 GiB card can install Qwen but not MiniMax H3 (needs 18 GiB).

How it works: the gateway starts a **child process that runs the same `aha-open models install`** the command line runs (one way to install,
not two; for a model with territories the child gets `--confirm-territory`), reads its output for the phase and for the bytes (the child, started with `AHA_OPEN_PROGRESS=lines`, prints `AHAOPEN_PROGRESS <done> <total>` from the
downloader's own byte counter; the weights folder cannot be measured instead, because `hf-xet` keeps what it downloads in its own cache and
assembles each file only at the end — observed on a real 18 GB download), and never waits for it inside a request. When the child
exits cleanly the gateway re-reads `config.toml` and replaces `cfg.models` in place; a model's state is computed from the disk on every request,
so the manifest changes with no restart. Stopping the gateway kills a running install.

Refused **before anything starts**, each with its own code and sentence (`409`): `unsupported_hardware` (no NVIDIA GPU, or less than the measured
minimum), `uv_missing`, `download_support_missing` (no `huggingface-hub`), `not_enough_disk`, `install_running` (one install at a time),
`already_installed`. A download that was cut off (a cancel, a restart, a power cut) is **not** `ready`: `model_index.json` is among the first
files to arrive, so the state also looks for files still marked `.incomplete` by `huggingface_hub` — unless `install.json`, written only after a
download returned successfully, says it finished (an earlier cut-off attempt leaves a stray `.incomplete` that nothing removes) — and pressing *Download* again continues
(`install_model` downloads again whenever the download is unfinished — it used to skip the download as soon as `model_index.json` existed).
After it started, a failure is `certificate_untrusted` (the machine reached the server but does not trust its certificate: a company proxy, an antivirus that scans HTTPS, a captive portal — say so, with `SSL_CERT_FILE`, instead of "no internet"), `network_unreachable` (the output shows the network could not be reached) or `install_failed`
(the last `error:` line, with the data folder replaced by `<data folder>`). `uv` is looked up in `AHA_OPEN_UV` (the Windows installer sets it), then `PATH`.

### 6.2 The MiniMax H3 worker (built and run for real: `minimax-h3-fl2va` and `minimax-h3-ref2va`, [05_models.md](05_models.md) §2.3, §2.4)

The only MiniMax H3 files AHA open uses are community re-quantizations in ComfyUI's single-file format (`convrot` INT4/INT8,
[05_models.md](05_models.md) §2.0). diffusers has the H3 model, VAEs, scheduler and pipeline blocks (Apache-2.0) but cannot read
these files, and ComfyUI, which can, is GPL-3.0 and a whole application. So the worker carries a small loader of its own
(`workers/minimax_h3/`), written from the files' headers, diffusers and the Apache-2.0 kernel library `comfy-kitchen`:

| File | What it does |
|---|---|
| `lenient_safetensors.py` | reads tensors by the header's offsets; the community files carry a 61–83 byte trailer that the `safetensors` package refuses |
| `convrot.py` | linear layers over INT8 / packed-INT4 weights rotated per 256 inputs, and over NVFP4 weights (FP4 packed high nibble first, FP8 scale per 16 inputs in cuBLAS' tiled layout kept as FP8, one float32 scale per tensor); the weight is decoded to bf16 just before each product (faster than the INT8 product on GPUs without cuBLASLt, closer to the unquantized model, and the NVFP4 build then runs on any NVIDIA GPU: NVIDIA's FP4 kernels need a Blackwell card) |
| `h3_models.py` | builds the transformer (key mapping, the "pruned" baked time table, chunked feed-forward), the 50-layer text encoder whose layers are read from disk when called (it also loads its vision tower, only for a job that has references; its attention repeats the key/value heads first, because PyTorch's grouped-query path has no memory-efficient kernel on Windows and would build a 10 GiB score matrix), and both VAEs; each builder refuses a file with a tensor left over or missing |
| `h3_references.py` | turns the request's ordered `inputs` (`reference`, `reference_video`, `reference_audio`) into the diffusers reference objects, in the order they were sent; refuses a URL, an unknown role and an audio reference with nothing beside it |
| `h3_runner.py` | runs the diffusers pipeline in three stages (encode, denoise, decode) so only one large part is on the GPU at a time (a job with references adds a stage that turns them into latents through both VAEs before the transformer arrives); cancellation is checked before every sampling step; caps PyTorch at the GPU memory that is physically free, or at the operator's `[gpu] memory_limit_gib` when smaller |

One worker serves two models: `minimax-h3-fl2va` (text to video) and `minimax-h3-ref2va` (one reference image, its own transformer
`transformer_ref`); they share one install folder because the encoder, the VAEs and the small configs are the same files. Each model
offers builds (`int8`, `mixed`, `nvfp4`) that differ only in the transformer file; the gateway names that file in the worker's `load`
message (the catalog is the single list). Deleting a build never removes the files another build, of either model, still needs.

Invariants specific to this worker: its environment never contains `comfyui` (a test reads the lock); every downloaded file
is checked against a pinned sha256 before `install.json` vouches for it; going over the GPU's physical memory is an error,
not a silent slowdown ([06_deployment.md](06_deployment.md) §6.1).

## 7. Code layout

```text
aha_open/            gateway (never imports torch — a test enforces it)
  app.py  scheduler.py  store.py  auth.py  catalog.py  install.py  model_install.py  launcher.py  loras.py  cli.py ...
  workers/           worker sources (mapped from the top-level workers/ folder when the wheel is built)
  ui/                the bundled web page
workers/
  qwen_image/        its own pyproject + uv.lock, an adapter calling diffusers
  minimax_h3/        its own pyproject + uv.lock (no ComfyUI); a loader for the community ConvRot files + diffusers' H3 pipeline
  fake/              a fake worker for tests (no torch)
windows_installer/   Inno Setup script + build_installer.py (docs/product/06_deployment.md section 0.1)
conformance/         the /v1 conformance suite (pytest, real HTTP calls)
schema/              OpenAPI
docs/
```

## 8. Invariants

1. The gateway does not import torch (a test runs in a child process and checks `sys.modules`).
2. There is no route besides `/v1/*`, the static files `/ui/*` (the web page — it only calls `/v1` and `/admin/*`), `/admin/loras*` (the LoRA manager), `/admin/models*` (installing a model, section 6.1) and `/admin/report` (the support report, section 6.2) — `/admin/*` is available only in the `local` profile, only for a loopback peer asking for a loopback name, and it still needs a token; every other case answers `404`.
3. No pixel or prompt leaves the machine except in the response to the client that sent it.
4. The output size is exactly the accepted `params`; there is no silent resize.
5. A job that has been `cancelled` never has a result.
