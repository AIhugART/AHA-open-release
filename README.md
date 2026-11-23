# AHA open

**AHA open** is an open-source service, written in Python, that runs open AI models on your own GPU and offers them through a
standard HTTP API (`/v1`). A client sends a request to generate or edit an image and gets the result back; AHA Studio
(a creative-suite client) is the first client, and the bundled web page (`/ui/`) is another.

> **Status (2026-11-23): no release yet (`0.1.0.dev`).** The `/v1` gateway, the conformance suite, the Qwen-Image-2.1 worker
> (INT8 build, size tiers, LoRA) and the web page exist. Run for real on one machine (RTX 3090 Ti 24 GiB, Windows): image
> generation, editing, transparent images, cancelling and LoRA through the gateway, installed from a built wheel in a clean
> environment. The web page (`/ui/`) was also driven in a real browser (connect, generate, result image, download). **Not tried yet:** the `lan` profile from a second machine, the `internet` profile
> (needs a real Cloudflare account), and video (MiniMax H3 is deferred). Per-item evidence levels are in
> [docs/05_models.md](docs/05_models.md) §3.

## Quick start (from source, needs [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU)

```bash
uv sync --extra dev --extra download
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research   # the INT8 build, ~18 GB
aha-open serve                                                          # listens on 127.0.0.1:8890
aha-open open                                                           # opens the web page, no token to paste
```

Qwen-Image-2.1 is offered as the **INT8 build only** (~18 GB of disk, ~9.5 GiB of VRAM at 1024², suits 12–24 GB GPUs);
the full BF16 model is not supported.

**LoRA:** `aha-open loras add my-style.safetensors --name "My style" --trigger mystyle` — the file is checked (safetensors
only) and then appears in the manifest and in `/ui/`; a client picks it with `params.loras`.

On the machine that runs AHA open, the web page can also add and remove LoRA files (**Manage LoRA files**), and **Refresh** re-reads the list after you copy files in by hand; no restart is needed.

A wheel can be built with `uv build`; it carries the worker sources and the web page, and `aha-open models install` creates
the worker's own environment from its lock file.

## v1 goals

| Model | What it does | Notes |
|---|---|---|
| Qwen-Image-2.1 | Text-to-image, image editing (≤ 10 reference images), transparent RGBA images, LoRA | Non-commercial license (Qwen Research License) |
| MiniMax H3 | Video with stereo audio | **Deferred**; only a community quantized build is planned (not the ~133 GB full model); its checkpoints are ComfyUI-format (ConvRot W4A4), which the diffusers worker cannot load, so support is paused until a runtime is chosen. See [docs/05_models.md](docs/05_models.md) §2.0 |

## Three ways to run it

| Profile | Where | Security |
|---|---|---|
| `local` | On the same machine as the client | Listens on `127.0.0.1` only |
| `lan` | One GPU machine on the local network | HTTPS required + one token per machine |
| `internet` | Through a Cloudflare Tunnel | Cloudflare Access + one token per machine |

## Documentation

| File | Content |
|---|---|
| [docs/00_index.md](docs/00_index.md) | Documentation map |
| [docs/01_overview.md](docs/01_overview.md) | What AHA open is, boundaries, principles |
| [docs/03_api_v1.md](docs/03_api_v1.md) | The `/v1` API specification (OpenAPI: [schema/](schema/)) |
| [docs/04_architecture.md](docs/04_architecture.md) | Gateway, workers, the queue, VRAM management |
| [docs/05_models.md](docs/05_models.md) | Qwen-Image-2.1 and MiniMax H3: capabilities, hardware, licenses, measurements |
| [docs/06_deployment.md](docs/06_deployment.md) | Installing the `local`, `lan` and `internet` profiles (Cloudflare Tunnel) |
| [docs/07_security.md](docs/07_security.md) | Threat model and security rules |
| [SECURITY.md](SECURITY.md) | How to report a vulnerability |
| [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) | Licenses of the packages AHA open installs |

Vietnamese: [README.vi.md](README.vi.md) and [docs/vi/](docs/vi/) (translations; English is canonical and the Vietnamese may lag).

## License

AHA open's code is under **Apache-2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)); it is free to use. **Model weights are not
in this repository**: each model has its own license, which the operator downloads, accepts and is responsible for complying with. See
[docs/05_models.md](docs/05_models.md).
