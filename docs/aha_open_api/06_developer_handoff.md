# Developer hand-off — building an app on AHA open

One page for the developer who takes this over. It says what you are getting, what has really been run, what to ask the operator for, what to build first, and where the traps are.
Everything else is linked, not repeated.

## What you are getting

| Item | Where |
|---|---|
| The API definition (the single original; pin the version you build against) | [`../03_api_v1.md`](../product/03_api_v1.md), machine-readable [`../../schema/aha_provider_v1.openapi.yaml`](../../schema/aha_provider_v1.openapi.yaml) — API version **`1.0` (draft)** |
| How-to guides | this folder: [01 quickstart](01_quickstart.md) · [02 concepts](02_concepts.md) · [03 build a client](03_build_a_client.md) · [04 recipes](04_recipes.md) · [05 errors](05_errors_and_troubleshooting.md) |
| Model reference and server behaviour | [07 models](07_models.md) (Qwen-Image-2.1, MiniMax H3) · [08 limits and behaviour](08_limits_and_behaviour.md) |
| Models you can build for | `qwen-image-2.1` (images, research use only) · `minimax-h3-fl2va` (text to video) · `minimax-h3-ref2va` (one reference image to video) |
| A working client you can copy | [examples/python/aha_open_client.py](examples/python/aha_open_client.py) (stdlib Python), [examples/node/generate.mjs](examples/node/generate.mjs) (Node 18+), plus PowerShell and curl |
| The server | AHA open `0.1.0.dev5` (pre-release). Windows installer and source: [../../README.md](../../README.md) |

## What has really been run (evidence)

| Claim | Level |
|---|---|
| Text to image, edit, transparent background and cancel against the **real** Qwen-Image-2.1 (INT8) on an RTX 3090 Ti, from Python, PowerShell, curl and Node | ✅ 2026-11-24 |
| The same prompt, seed and size give the same bytes (sha256 `c142df9e…ba45`) from Python, PowerShell and Node | ✅ 2026-11-24 |
| Every example script and every route named in these guides is run by the conformance suite against a live gateway with a synthetic model | 🧪 (`uv run pytest conformance`) |
| Reference images, masks, LoRA recipe | 🧪 synthetic model only. Qwen-Image-2.1 has **no mask** (`mask: false`); no real LoRA was installed when this was written |
| **Video** with MiniMax H3 through a real gateway: text to video at `preview` (job, downloaded file checked, cancel) and one reference image (also from the web page) | ✅ 2026-11-25, RTX 3090 Ti |
| MiniMax H3 builds `int8`, `mixed`, `nvfp4` at both sizes (frames and sound checked) | ✅ 2026-11-26, one RTX 3090 Ti ([07_models.md](07_models.md) section 3.4) |
| The `lan`/`internet` rules for MiniMax H3 (off until the operator accepts the hosted-service duties) | 🧪 |
| MiniMax H3 with a first or last picture, other lengths, other aspect ratios, several references | ⬜ not supported / not measured (the manifest says so) |
| The `lan` profile from a second machine, the `internet` profile (Cloudflare), pairing from a real second client | ⬜ not tried |
| Speed on other graphics cards | ⬜ to be measured (3090 Ti: first job ~120 s including model load, then ~36–42 s per 1024² image) |

## Ask the operator for these four things

1. **The base URL** (`http://127.0.0.1:8890` on the same machine; an `https://` address otherwise) and which profile it is (`local`, `lan`, `internet`).
2. **A token for your app** (`aha-open clients create --name "<your app>"` on the server; shown once). On the same machine and same user the local token file is enough ([01_quickstart.md](01_quickstart.md)).
3. **For `lan`:** the server's certificate file. **For `internet`:** a Cloudflare Access service token (`CF-Access-Client-Id` / `-Secret`).
4. **Confirmation that the model is installed** (`aha-open check <url> --token <token>` lists each model's state). The model files are not shipped; the operator downloads them and accepts the model's license.

## Build in this order

1. Read [02_concepts.md](02_concepts.md) (10 minutes) and run [01_quickstart.md](01_quickstart.md) once, so you have seen a real picture come back.
   Then read the section of [07_models.md](07_models.md) for each model your app will offer, and [08_limits_and_behaviour.md](08_limits_and_behaviour.md).
2. Copy `aha_open_client.py` (or port it: it is about 200 lines) and build against **`GET /v1/manifest`**: model list, sizes, steps, license, what is `ready`.
3. Text to image with progress, cancel, and saved-before-expiry results.
4. Edit and reference images (uploads).
5. Error handling with the sentences in [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md).
6. Only then: transparent background, LoRA, video, pairing, `lan`/`internet`.

Acceptance: the checklist at the end of [03_build_a_client.md](03_build_a_client.md).

## The traps that cost the most time

- **A web page calling AHA open directly gets `403`** (browser `Origin`). Put a small backend between them, or use a native app ([03_build_a_client.md](03_build_a_client.md) "Web apps").
- **Upload parts are numbered from 0 and must be exactly `part_size` bytes** (the last is the remainder); a wrong size is `400`.
- **Sizes are exact.** No stretching, no cropping: send a size from `capabilities.sizes` or get `400`.
- **`busy` means wait `retry_after_s`**, then send the *same* request again. Always send a `client_request_id` so a retry can never run twice.
- **A failed job is never re-run for you** and costs GPU time; let the user decide.
- **Results are deleted after `limits.result_ttl_s`.** Save what you need and the `seed`.
- **Do not log the token, the prompt or the pictures.**
- **Qwen-Image-2.1 is research and evaluation only** (`commercial_use: not_allowed`). Show the license from the manifest; if `required_ui_label` is present, show it verbatim.
- **MiniMax H3 puts duties on your app**: the label, excluded territories, disclosure of machine-made video, and more ([07_models.md](07_models.md) section 5).
- **Always send `short_edge` for video.** Left out, `minimax-h3-fl2va` makes the 23-minute size.
- **Uploads expire, used or not, and at most 32 are stored per client** ([08_limits_and_behaviour.md](08_limits_and_behaviour.md)).
- **A repeated `client_request_id` returns the old job**, also a failed one. *Try again* needs a new id.
- **Only `/v1` is the API.** `/ui/` and `/admin/*` are not contracts; do not call them.

## Rules you can rely on (kept by the server and tested)

No telemetry; prompts and pictures go only back to the client that sent them. A job is never waited on in an open request (job + poll). A cancelled job never has a result.
The server listens on `127.0.0.1` unless the operator chooses otherwise. Changing the API adds fields (minor) or breaks (major `/v2`); **ignore fields you do not know** so a minor change does not break you ([`../03_api_v1.md`](../product/03_api_v1.md) §10).

## If something is wrong in the guides or the API

Check [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md) first. A security problem: [`../../SECURITY.md`](../../SECURITY.md). Anything else: open an issue in the project's repository
([AHA-open-release](https://github.com/AIhugART/AHA-open-release)) with the HTTP status, the `error.code` and `message`, and the job id, never the token.
