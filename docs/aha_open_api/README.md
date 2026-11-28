# Using the AHA open API

> **Who this is for.** People who want to generate or edit images from a script, and developers who build an app on top of AHA open.
> It is a *how-to* guide. The *definition* of the API (every field, every error code) is [`../03_api_v1.md`](../product/03_api_v1.md); the
> machine-readable form is [`../../schema/aha_provider_v1.openapi.yaml`](../../schema/aha_provider_v1.openapi.yaml). This guide links
> to them instead of repeating them, so there is one place to look when something is unclear.

AHA open runs open AI models on your own GPU and serves them over HTTP at `/v1`. You send a **job** ("draw this", "edit that"), it
answers at once with a job id, and you **poll** until the picture is ready. Nothing waits on an open connection while the GPU works.

## Pick your road

| You are… | Start here | You will have |
|---|---|---|
| **A person who wants a picture from a script** | [01_quickstart.md](01_quickstart.md) | A PNG on disk in about five minutes, from Python, PowerShell or curl |
| **A developer building an app** | [02_concepts.md](02_concepts.md) → [03_build_a_client.md](03_build_a_client.md) | The flow every client follows, and the traps (browser `403`, upload part numbers, `busy`) |
| **Looking for a specific task** | [04_recipes.md](04_recipes.md) | Text to image, edit, reference images, transparent background, LoRA, video, cancel |
| **Designing for Qwen-Image-2.1 or MiniMax H3** | [07_models.md](07_models.md) | Sizes, parameters, inputs, variants, timings and license duties of each model |
| **Handing the project to a developer** | [06_developer_handoff.md](06_developer_handoff.md) | The one page to give them first |
| **Something went wrong** | [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md) | Each error code, what it means for the person at the keyboard, and what to do |

## The files

| File | What it is |
|---|---|
| [01_quickstart.md](01_quickstart.md) | Run AHA open, find your token, make the first picture |
| [02_concepts.md](02_concepts.md) | URLs, tokens, the manifest, jobs, uploads, results |
| [03_build_a_client.md](03_build_a_client.md) | The recommended client flow, step by step, for app developers |
| [04_recipes.md](04_recipes.md) | Copy-and-run answers to "how do I…" |
| [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md) | Errors in plain words |
| [06_developer_handoff.md](06_developer_handoff.md) | One page to hand to the developer taking over: what is verified, what to ask the operator for, build order, traps |
| [07_models.md](07_models.md) | Model reference: `qwen-image-2.1`, `minimax-h3-fl2va`, `minimax-h3-ref2va` |
| [08_limits_and_behaviour.md](08_limits_and_behaviour.md) | Limits, expiry, retries, defaults, queueing, and what your app should store |

## The examples

Every example is a real file that the project's tests **run against a live gateway**, so it cannot quietly drift away from the API.
All of them read the same two settings: `AHA_OPEN_URL` (default `http://127.0.0.1:8890`) and `AHA_OPEN_TOKEN` (default: the local
token file, see the quickstart).

| File | Language | Does |
|---|---|---|
| [examples/python/aha_open_client.py](examples/python/aha_open_client.py) | Python 3.9+, standard library only | A small client you can copy into your project: upload, job, poll, result, cancel |
| [examples/python/generate.py](examples/python/generate.py) | Python | Text to image from the command line |
| [examples/python/edit.py](examples/python/edit.py) | Python | Edit an image, with reference images and (when the model supports it) a mask |
| [examples/python/recipes.py](examples/python/recipes.py) | Python | Transparent background, LoRA, video (pick a size tier, MiniMax H3 licence duties), cancel (see [04_recipes.md](04_recipes.md)) |
| [examples/powershell/generate.ps1](examples/powershell/generate.ps1) | PowerShell 5.1 / 7 | Text to image, no modules |
| [examples/node/generate.mjs](examples/node/generate.mjs) | Node 18+, built-in `fetch` | Text to image, one file, no packages (server-side only: a browser page gets `403`) |
| [examples/curl/generate.sh](examples/curl/generate.sh) | `sh` + curl | Text to image with raw HTTP calls |

## Three facts worth knowing before you write any code

1. **The manifest is the source of truth.** Ask `GET /v1/manifest` which models exist, which are `ready`, which sizes they accept and
   what their license allows. Do not hard-code model names or sizes.
2. **Sizes are exact.** AHA open never stretches or crops. Ask for a size the model declares, or the request is refused with `400`
   before anything is queued.
3. **Each model has its own license**, and the manifest states it; show it to your users. Qwen-Image-2.1 is for research and
   evaluation only (`commercial_use: not_allowed`). MiniMax H3 is `allowed_with_conditions`: show `MiniMax H3` verbatim, respect the
   excluded territories and read the other duties ([07_models.md](07_models.md) section 5).
