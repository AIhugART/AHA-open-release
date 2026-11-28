# Quickstart — your first picture in five minutes

This page gets one picture out of AHA open from a script, on the **`local` profile** (the script and AHA open run on the same
computer, the default). For other setups (another computer, the internet) read [02_concepts.md](02_concepts.md) first.

**You need:** AHA open installed with a model downloaded ([../../README.md](../../README.md)), and one of Python 3.9+, Windows PowerShell,
or curl. An NVIDIA GPU with about 10 GiB of VRAM is needed to run Qwen-Image-2.1 (9 GiB for its ConvRot variants). Video with
MiniMax H3 needs a 16 GB card or larger ([07_models.md](07_models.md)).

## 1. Make sure AHA open is running

- **Windows installer:** Start menu ▸ **AHA open**. The gateway starts in the background and the web page opens.
- **From source:** `aha-open serve` (it listens on `http://127.0.0.1:8890`).

Check it answers (`/v1/health` needs no token):

```powershell
curl.exe http://127.0.0.1:8890/v1/health
```

```bash
curl http://127.0.0.1:8890/v1/health
```

You should see `"status":"ok"` and, under `gpus`, your graphics card. `"status":"starting"` means wait a few seconds. No answer at
all means AHA open is not running (see [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md)).

(In Windows PowerShell write `curl.exe`: plain `curl` there is a different command.)

## 2. Your token

Every route except `GET /v1/health` needs a token: an `Authorization: Bearer <token>` header. You usually **do not have to do anything**:
on the `local` profile AHA open writes a token into a file that only your Windows/Linux user can read, and all the examples below look
there by themselves.

| System | The token file |
|---|---|
| Windows | `%LOCALAPPDATA%\AHA-open\local_client.token` |
| Linux / macOS | `~/.local/share/aha-open/local_client.token` (or `$XDG_DATA_HOME/aha-open/…`) |
| Anywhere | the folder named by the environment variable `AHA_OPEN_DATA_DIR`, if you set one |

Run the script as a **different user**, or from a different computer? Then ask the person who runs AHA open for a token, or, if that is
you: `aha-open clients create --name "my script"` prints one **once**. Give it to the script through an environment variable, never
by typing it into a command line that is saved in your shell history or into source code:

```powershell
$env:AHA_OPEN_TOKEN = Read-Host "AHA open token"      # PowerShell: asks, does not echo it into history
```

```bash
read -rs AHA_OPEN_TOKEN && export AHA_OPEN_TOKEN       # bash: asks, does not echo
```

## 3. Make a picture

Pick one. Each does the same thing and asks the server which model is ready and which size it accepts.

**Python** (standard library only):

```bash
cd docs/aha_open_api/examples/python
python generate.py "a lighthouse on a cliff at sunset, oil painting" --seed 42 --out lighthouse.png
```

**PowerShell** (Windows PowerShell 5.1 or PowerShell 7):

```powershell
cd docs\aha_open_api\examples\powershell
.\generate.ps1 "a lighthouse on a cliff at sunset, oil painting" -Seed 42 -Out lighthouse.png
```

**curl** (Linux, macOS, Git Bash; it needs no other tool, but shows the model and size in plain variables):

```bash
cd docs/aha_open_api/examples/curl
AHA_OPEN_MODEL=qwen-image-2.1 sh generate.sh "a lighthouse on a cliff at sunset, oil painting" lighthouse.png
```

The scripts report on the terminal's error stream (so a pipe still carries only what you ask for). On the project's test machine
the Python and the PowerShell script both printed:

```text
model qwen-image-2.1 - license: Qwen RESEARCH LICENSE AGREEMENT (commercial use: not_allowed)
saved lighthouse.png: 1024x1024, 2312944 bytes, seed 42
```

(Python also shows a live `running: step N/40` counter.) Same prompt, same `seed`, same size and steps give the **same picture, byte for byte**:
both files had sha256 `c142df9e…ba45`, the value recorded for this prompt in [../05_models.md](../product/05_models.md) §1 (✅ 2026-11-24, Python 3.12 / Windows PowerShell 5.1 against the real model).
The curl script does not set a seed, so its picture differs every time.

### How long does it take?

Measured on one RTX 3090 Ti 24 GiB, INT8 build, 1024×1024, 40 steps ([../05_models.md](../product/05_models.md) §1 and §3.1): the **first**
picture after AHA open starts also loads the model (about 120 s once), then each picture takes about 36–42 s. Other graphics cards: **to be measured**.
While the model loads the job shows `loading_model`; that is not an error.

## 4. What just happened

Five HTTP calls, the same five for every client in every language:

1. `GET /v1/manifest` — which models exist, which are `ready`, which sizes they accept, what their license says.
2. `POST /v1/jobs` — "draw this". The answer comes back at once (`202`) with a job id; the GPU has not started yet.
3. `GET /v1/jobs/{id}` — asked again and again until `status` is `succeeded` (or `failed`, `cancelled`).
4. `GET /v1/jobs/{id}/result` — the PNG bytes.
5. Nothing else: no session, no cookie, no callback.

Why a job and not "send the prompt, get the picture back"? A picture takes seconds to minutes. A connection held open that long is
broken by proxies, laptops that sleep and flaky Wi-Fi; a job id survives all of those.

## 5. Something went wrong?

| You see | It means | Do this |
|---|---|---|
| `cannot reach http://127.0.0.1:8890` | AHA open is not running (or listens elsewhere) | Start it (step 1); if you changed the port, set `AHA_OPEN_URL` |
| `unauthorized` / "did not accept the token" | No token, a wrong one, or one that was revoked | Step 2 |
| `no model is ready … is not_installed` | The model is not downloaded yet | Open the web page (`aha-open open`) and press **Download model** |
| `no model is ready … is loading` | The model is loading into the GPU | Wait about two minutes and run again |

The complete list, in plain words: [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md).

## 6. Next

- Edit a picture, use reference images, a transparent background or a LoRA: [04_recipes.md](04_recipes.md).
- Build your own app: [02_concepts.md](02_concepts.md), then [03_build_a_client.md](03_build_a_client.md).
