# AHA open

**AHA open** is an open-source service, written in Python, that runs open AI models on your own GPU and offers them through a
standard HTTP API (`/v1`). A client sends a request to generate or edit an image and gets the result back; AHA Studio
(a creative-suite client) is the first client, and the bundled web page (`/ui/`) is another.

> **Use `0.1.0.dev5` or later.** The `dev2` installer left out worker code (Qwen-Image-2.1 ConvRot and MiniMax H3 cannot run from it).
> The `dev3` installer fails on a computer without Python 3.12 (seen in a simulation of one) and opens a terminal window while
> generating. Installing dev5 over an older version should keep your models and settings (an upgrade over dev3 has not been tried).
>
> **Status (2026-11-28): pre-release `0.1.0.dev5`, with a Windows installer.** The `/v1` gateway, the conformance suite, the Qwen-Image-2.1 worker
> (four quantized variants to choose from, size tiers, LoRA) and the web page exist. Run for real on one machine (RTX 3090 Ti 24 GiB, Windows): image
> generation, editing, transparent images, cancelling and LoRA through the gateway, installed from a built wheel in a clean
> environment. The web page (`/ui/`) was also driven in a real browser (connect, generate, result image, download). Video with
> MiniMax H3 (text to video, and one reference image) has also been run for real through the gateway on that machine.
> **Not tried yet:** the `lan` profile from a second machine and the `internet` profile
> (needs a real Cloudflare account). Per-item evidence levels are in
> [docs/product/05_models.md](docs/product/05_models.md) §3.

## Quick start on Windows (installer)

1. From the [Releases page](https://github.com/AIhugART/AHA-open-release/releases) download `AHA-Open-Setup-<version>.exe` and `SHA256SUMS`
   (it is a pre-release for now) and compare the file's SHA-256 with the one in `SHA256SUMS`. The installer is **not code-signed**, so
   Windows SmartScreen may warn the first time.
2. Run it (about 16 MiB, no administrator rights; it needs the internet the first time, to fetch Python). Then Start menu ▸ **AHA open**: the
   gateway starts in the background and the web page opens.
3. On the page, tick the licence box and press **Download model** (Qwen-Image-2.1, the variant that fits your GPU, about 17 GB; research and evaluation use only —
   read the licence it shows). You need an NVIDIA GPU with about 10 GiB of VRAM or more and about 25 GB of free disk. The **Models** panel then lets you
   download another variant, switch between them, or delete one.

Tried for real on one machine (RTX 3090 Ti, Windows 11): installed, model downloaded through the page, the reference picture generated
with the same bytes as the earlier measurement, uninstalled. **Not tried yet:** the interactive wizard, the SmartScreen prompt, a
physically clean machine (the no-Python case was only simulated). macOS is not supported in v1. Details and how to build the installer: [docs/product/06_deployment.md](docs/product/06_deployment.md) §0.1.

## Quick start (from source, needs [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU)

```bash
uv sync --extra dev --extra download
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research   # the best variant for your GPU (~17 GB); see `models list`
aha-open serve                                                          # listens on 127.0.0.1:8890
aha-open open                                                           # opens the web page, no token to paste
```

Qwen-Image-2.1 is offered in **four quantized variants** to fit the machine — `int8`, `int8-convrot`, `mixed-p010`, `mixed-p015`
(15–18 GB of disk, 8–10 GiB of VRAM at 1024², suit 12–24 GB GPUs; `aha-open models list` shows which fit and which are installed, `models use` / `models remove` switch and delete);
the full BF16 model is not supported.

**LoRA:** `aha-open loras add my-style.safetensors --name "My style" --trigger mystyle` — the file is checked (safetensors
only) and then appears in the manifest and in `/ui/`; a client picks it with `params.loras`.

On the machine that runs AHA open, the web page can also add and remove LoRA files (**Settings ▸ Manage Models**), and **Refresh** re-reads the list after you copy files in by hand; no restart is needed. If something does not work, **Report** (or `aha-open report`) makes one text with the machine, the models and the end of the logs, without your token or prompts, for you to copy and send; nothing is sent by AHA open. The page is described in [docs/product/10_web_ui.md](docs/product/10_web_ui.md).

A wheel can be built with `uv build`; it carries the worker sources and the web page, and `aha-open models install` creates
the worker's own environment from its lock file.

**Tests:** `uv run pytest` (no GPU needed). Seven tests are skipped until the Qwen worker's own environment exists inside the
checkout: `cd workers/qwen_image && uv sync --frozen` (this is a separate environment from the one `aha-open models install`
creates under the data directory, ~5 GB).

## v1 goals

| Model | What it does | Notes |
|---|---|---|
| Qwen-Image-2.1 | Text-to-image, image editing (≤ 10 reference images), transparent RGBA images, LoRA | Non-commercial license (Qwen Research License) |
| MiniMax H3 | Video with stereo audio | **Deferred**; only a community quantized build is planned (not the ~133 GB full model); its checkpoints are ComfyUI-format (ConvRot W4A4), which the diffusers worker cannot load, so support is paused until a runtime is chosen. See [docs/product/05_models.md](docs/product/05_models.md) §2.0 |

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
| [docs/product/01_overview.md](docs/product/01_overview.md) | What AHA open is, boundaries, principles |
| [docs/product/03_api_v1.md](docs/product/03_api_v1.md) | The `/v1` API specification (OpenAPI: [schema/](schema/)) |
| [docs/product/04_architecture.md](docs/product/04_architecture.md) | Gateway, workers, the queue, VRAM management |
| [docs/product/05_models.md](docs/product/05_models.md) | Qwen-Image-2.1 and MiniMax H3: capabilities, hardware, licenses, measurements |
| [docs/product/06_deployment.md](docs/product/06_deployment.md) | Installing the `local`, `lan` and `internet` profiles (Cloudflare Tunnel) |
| [docs/product/07_security.md](docs/product/07_security.md) | Threat model and security rules |
| [SECURITY.md](SECURITY.md) | How to report a vulnerability |
| [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) | Licenses of the packages AHA open installs |

Vietnamese: [README.vi.md](README.vi.md) and [docs/vi/](docs/vi/) (translations; English is canonical and the Vietnamese may lag).

## License

AHA open's code is under **Apache-2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)); it is free to use. **Model weights are not
in this repository**: each model has its own license, which the operator downloads, accepts and is responsible for complying with. See
[docs/product/05_models.md](docs/product/05_models.md).
