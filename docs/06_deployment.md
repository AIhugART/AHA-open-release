# Deploying AHA open — three profiles

> The `aha-open` commands below exist (🧪 tested; the `lan` profile has run for real over loopback but not yet from a second machine;
> `internet` has not been tried with a real Cloudflare account). The examples use made-up addresses/domain names.

## 0. Common requirements

- An NVIDIA GPU + CUDA driver (macOS/Apple Silicon is not part of v1; to be measured).
- Disk: 15–19 GB for one Qwen variant (the shared text encoder and VAE are downloaded once; each further ConvRot variant adds 4.6–7.3 GB). MiniMax H3 (installable from the command line or the web page) needs
  **~43-50 GB** for one build: the shared text encoder 25.3 GiB + VAEs 5.4 GiB, plus the transformer of the chosen build (NVFP4
  11.7 GiB, mixed 14.8 GiB, INT8 19.5 GiB), from three pinned repositories, plus its own PyTorch environment — see
  [05_models.md](05_models.md) §2.0 and §2.5.
- GPU for MiniMax H3: **at least 16 GB** (8 and 12 GB cards are not supported); the reference model needs 18 GB or more. Which build and
  size fit: [05_models.md](05_models.md) §2.5.
- Python + [`uv`](https://docs.astral.sh/uv/) (AHA open creates a separate venv for each worker from the worker's lock file).

```bash
uv sync --extra download    # from a source checkout (not on PyPI yet); or install a wheel built with `uv build`
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research            # the best variant that fits your GPU (~17 GB); `--variant` picks another, see `models list`
# MiniMax H3 (~46 GB): also confirm you are not in an excluded territory
# aha-open models install minimax-h3-fl2va --accept-license minimax-h3 --confirm-territory
# MiniMax H3 with reference images (a second model): it goes into the same folder and downloads only its own 14 GiB transformer when the line above was done
# aha-open models install minimax-h3-ref2va --accept-license minimax-h3 --confirm-territory
aha-open doctor
```

## 0.1 Windows installer (Windows 10/11, 64-bit)

`AHA-Open-Setup-<version>.exe` is a small installer (about 16 MiB: `uv` and the AHA open wheel; **no** PyTorch, **no** model weights). It
installs for the current user only (no administrator rights) into `%LOCALAPPDATA%\Programs\AHA open`, creates a Python environment for the
gateway with `uv` (the first time this downloads Python, so it needs the internet), pins every dependency to the versions in `uv.lock`, and
writes the default `local` configuration. Your settings, models and images live apart, in `%LOCALAPPDATA%\AHA-open`.

| Step | What you do |
|---|---|
| Install | Run the `.exe`. It is **not code-signed**, so Windows SmartScreen may warn the first time; compare the file's SHA-256 with `SHA256SUMS` on the release page |
| First start | Start menu ▸ *AHA open* (or the *Start AHA open now* box at the end of the setup). It starts the gateway in the background — no console window — and opens the web page |
| The model | The model is **not** downloaded by the installer. The page shows *Download a Model* with the licence; tick the box and press the button ([05_models.md](05_models.md), [04_architecture.md](04_architecture.md) section 6.1). About 18 GB, plus the model's own Python environment. MiniMax H3 is on the same panel (about 46 GB; its licence excludes the EU, UK, South Korea and the US, so a second checkbox asks the person to confirm they are not in one of them; the reference model then needs only ~15 GB more) |
| Stop | Start menu ▸ *Stop AHA open* (it also stops a download in progress) |
| Uninstall | Settings ▸ Apps. It stops the gateway first and **keeps your models and settings** unless you answer *Yes* to the question about deleting them (a silent uninstall keeps them) |

A failed install says why — no internet / no free disk / Windows refused to write — and leaves nothing behind (it exits with a non-zero
code and no entry in *Add or remove programs*). The model download is refused before it starts if there is no NVIDIA GPU (or less than
~10 GiB of VRAM), or less than ~20 GB of free disk. macOS is not supported in v1.

Evidence: ✅ the built installer was installed into a temporary folder with a temporary data folder; the installed packages matched the pins
and contained no PyTorch; it was uninstalled (program files gone, data kept); a second install over the first upgraded it with a single
*Add or remove programs* entry; two real network failures (offline, a dead proxy) exited non-zero, registered nothing and explained
themselves. 🧪 the guards in `tests/test_windows_installer.py`. ⬜ Not yet seen: the interactive wizard (only the silent path was run),
the SmartScreen prompt, a machine with no Python at all (Python 3.12 was already on the test machine). There is no icon yet.
✅ End to end on a real machine: installed, started with the shortcut's command, the model downloaded through the page (cut off once
and continued), the reference picture generated (sha256 identical to the P0 measurement, [05_models.md](05_models.md)), then
uninstalled with the program removed and the data kept.

**Correction (2026-11-24).** The first smoke runs started the gateway with `python -m aha_open.cli` from the source checkout; `-m` puts the
*current folder* first on `sys.path`, so they exercised the checkout, not the installed package (users were not affected: the shortcut starts
in the install folder). The smoke now runs everything from a neutral folder, checks that `aha_open` is imported from the installed
environment and that the running gateway's `source_digest` (`run/gateway.json`) equals the digest of the installed files; `aha-open start`
and the web page's model install also start their child processes in the data folder for the same reason. Only the evidence from
the fixed smoke counts.

### Building it (maintainers)

```powershell
winget install JRSoftware.InnoSetup        # Inno Setup 6; ISCC.exe is NOT put on PATH, the build script finds it
python windows_installer/build_installer.py --smoke     # tests -> wheel -> installer -> install it into a temp folder, run it, uninstall
python windows_installer/build_installer.py --check-tag v0.1.0-dev1    # a tag must match the one version in pyproject.toml
```

The build refuses: uncommitted changes under the packaged paths (a build packs the working tree, not a commit; `--allow-dirty` to try one),
red tests (`AHA_OPEN_BUILD_SKIP_QUALITY_GATE=1` skips them, loudly), a wheel that needs PyTorch or carries model weights, missing dependency
pins, and a size outside 10–80 MiB. `--smoke-offline` installs with the internet made unreachable and checks it fails cleanly.
The build writes `dist/AHA-Open-Setup-<version>.exe`, the wheel it contains and `dist/SHA256SUMS` (exactly those two files).

### Publishing a release (maintainers)

Every step that is public (the snapshot push, the GitHub Release) is a decision of the project owner each time.

1. **One version, never reused.** `pyproject.toml` and `aha_open/__init__.py` carry the same number (and `uv lock` is run). A number that is
   already public is never used again for different bytes: a release whose wheel differs from the wheel inside its installer is a
   different build and gets a new number.
2. `python windows_installer/build_installer.py --smoke` (clean tree, tests, then the real installer installed, run and uninstalled).
3. `python windows_installer/build_installer.py --check-tag v<version>`; `gh auth status`.
4. *(public)* `python tools/release/snapshot.py OUT_DIR` writes the public tree from `HEAD` (committed bytes only; it fails if a top-level
   folder is not classified in `tools/release/`); push it to the public release repository.
5. *(public)* **Upload first, then say so anywhere else:** `gh release create v<version> --repo <public repo> --prerelease --notes-file NOTES
   dist/AHA-Open-Setup-<version>.exe dist/aha_open-<version>-py3-none-any.whl dist/SHA256SUMS`.
6. Verify from the outside, anonymously: download the three assets from the release page and run `sha256sum -c SHA256SUMS`
   (PowerShell: `Get-FileHash`). "The upload finished" is not the check.
7. Only then update whatever points at the release (a download page) with the hashes **read from the release**, not typed from memory.

| If you already did… | Take it back by… | Careful |
|---|---|---|
| Uploaded the wrong asset | `gh release upload --clobber` with the right file | Do not delete an asset someone may already link to |
| Published the wrong version | `gh release edit --draft` to hide it, then publish a **new** number | Never reuse a public number |
| Pointed a page at the wrong file | Fix the page and publish it again | Fastest of all: no binary changes |

### Problems met while building (symptom → cause → what to do)

| Symptom | Cause | What to do |
|---|---|---|
| "ISCC not found" although Inno Setup is installed | `winget` installs it per user and does not add it to `PATH` | The build script looks in `%LOCALAPPDATA%\Programs\Inno Setup 6`; check with `winget list --id JRSoftware.InnoSetup` |
| A silent uninstall started from Git Bash hangs on a window | MSYS rewrites `/VERYSILENT` into `C:/Program Files/Git/VERYSILENT` | Use PowerShell (`Start-Process … -ArgumentList "/VERYSILENT"`) or `MSYS_NO_PATHCONV=1` |
| A failed install exited 0 and stayed in *Add or remove programs* | The environment was built after the files were registered; Inno only logs an exception raised there | It is built in `PrepareToInstall`; `--smoke-offline` guards it |
| The smoke passed but tested the source checkout | `python -m aha_open.cli` puts the current folder first on `sys.path` | The smoke runs from a neutral folder and compares `source_digest`; `start` and the model install start their children in the data folder |
| "weights found in the installer" for a few bytes | `.pth` is also a Python path file (`_virtualenv.pth`) | A file counts as weights by suffix **and** size (≥ 10 MiB) |
| The offline failure said only "could not be prepared" | The marker list did not contain uv's real wording | Markers come from real logs kept in `tests/fixtures/installer/`; add the new log when a new wording appears |
| The page's progress bar stayed at 0% for minutes | `hf-xet` keeps the bytes in its own cache and assembles each file at the end | The bar uses the downloader's own byte count (`AHAOPEN_PROGRESS`) |
| The smoke failed at `/admin/models` on a good build | Its list of expected models still held only Qwen after MiniMax H3 was added to the page | The list is `WEB_MODELS`; a test holds it equal to what the page can install. A smoke that fails leaves its gateway running: stop and uninstall the temporary install by hand |
| A cut-off download was reported *ready* and *Download* did not continue | `model_index.json` is among the first files to arrive | Unfinished = a `.incomplete` file, or hf began and no `install.json` ended it |

## 1. The `local` profile — the client on the same machine

```bash
aha-open serve            # listens on http://127.0.0.1:8890
aha-open clients create --name "Client on this machine"   # prints a token once
```

The `local` profile writes `local_client.token` into the data folder (readable only by the current user), so a client running
as the same user can connect without pasting a token; `aha-open open` opens the bundled web page already signed in. `clients create`
is only needed when a client runs under another operating-system account.

Note: the client and AHA open share the GPU with graphics applications (Photoshop/Premiere). When VRAM is tight, set
`variant = "int8"` and avoid running a video model on the machine that is editing video.

### Limiting the GPU memory the server uses

```toml
[gpu]
memory_limit_gib = 15.5   # absent or 0 = the whole card
```

For a card shared with other programs. The server judges which model variants and sizes fit against this number instead of the
card's size, and the workers may not allocate more PyTorch memory than that (the MiniMax worker keeps 0.5 GiB of it free). It is
also how a smaller card is imitated on a larger one; the imitation is not exact (bandwidth and the driver's own memory differ).
`aha-open doctor` and the support report show the limit and the GPU's compute capability (its architecture generation).

## 2. The `lan` profile — one GPU machine serving the whole local network

```toml
profile = "lan"
listen = "192.168.1.20:8890"   # name the interface; it never listens on 0.0.0.0 by itself
[auth]
require_token = true
enroll = true
```

```bash
aha-open tls init           # create a self-signed certificate (or point to your company's internal CA certificate)
aha-open tls fingerprint    # SHA-256 for staff to compare
aha-open serve
```

A staff machine: add the server `https://192.168.1.20:8890` in the client ▸ pair ▸ compare the fingerprint ▸ the administrator runs
`aha-open clients approve <code>`. One token per machine, revoked individually with `aha-open clients revoke <id>`.

Firewall: open the port only to the internal network range.

## 3. The `internet` profile — through a Cloudflare Tunnel

Goal: use the GPU machine at home/the office from anywhere, **without opening a router port** and without exposing its IP.

```text
client (far away) ──HTTPS──► Cloudflare edge (Access checks the service token) ──tunnel──► cloudflared ──► 127.0.0.1:8890 (AHA open)
```

Steps:

1. Have a domain on Cloudflare (e.g. `aha-open.example.com`).
2. Configure AHA open:

   ```toml
   profile = "internet"
   listen = "127.0.0.1:8890"      # loopback only; cloudflared is the only thing that talks to it
   hosts = ["aha-open.example.com"]   # the public name the tunnel serves; any other Host is rejected
   access_confirmed = true            # set only after step 4 (Cloudflare Access) is done
   max_queue_per_client = 2
   [auth]
   require_token = true
   enroll = false                # off by default on internet
   ```

3. Install `cloudflared` and create a tunnel pointing `aha-open.example.com` → `http://127.0.0.1:8890`.
4. Cloudflare Zero Trust ▸ Access ▸ create an application for `aha-open.example.com` with a **Service Auth** policy, and create one
   **service token** per user/machine.
5. `aha-open clients create --name "Laptop An"` → an AHA open token.
6. In the client: URL `https://aha-open.example.com`, paste the AHA open token, `CF-Access-Client-Id` and `CF-Access-Client-Secret`.

`aha-open tunnel --hostname aha-open.example.com` prints a ready `config.yml` for cloudflared, the lines to add to
`config.toml` and the Access steps (steps 3–4 need to sign in to the operator's Cloudflare account, so AHA open does **not**
do them for you). When done, check from a different network:

```bash
aha-open check https://aha-open.example.com --cf-id <id> --cf-secret <secret> --token <AHA open token>
```

`aha-open check` says which layer is broken: the server is off, the name does not resolve, the certificate is not trusted,
Cloudflare Access blocks (a wrong service token), or the AHA open token is wrong/revoked.

**Know this before choosing this profile:**

| Point | Consequence |
|---|---|
| TLS ends at Cloudflare's edge | Cloudflare can see images/prompts in clear. For sensitive data use `lan` or a VPN |
| Request-body limit (believed to be 100 MB on the Free plan — **to be measured**) | That is why inputs go through `POST /v1/uploads` in parts |
| The proxy's response timeout (**to be measured**) | The API is an asynchronous job; no request waits for a whole generation |
| Model licenses | Qwen (`not_allowed`) is for research/evaluation only. MiniMax H3 is **off by default** on this profile: serving others through an API is a "Hosted Service" under the MiniMax license, and the operator must have their own user terms, safety measures and a violation-reporting channel before setting `hosted_service_obligations = "accepted"`. MiniMax cannot be used in the EU/UK/Korea/US |
| GPU/power cost | Limit the queue per client so one person cannot take it all |

## 4. Running as a system service

- Windows: a service (planned: `aha-open service install`) or Task Scheduler at sign-in.
- Linux: a `systemd` unit (a template will be in the repository).
- Updating: `aha-open update` (planned) — stop accepting new jobs, wait for running jobs to finish, then restart.

## 5. Checking after installation

```bash
aha-open doctor                      # GPU, configuration, models
aha-open open                        # opens the web page (/ui/); the local profile needs no token pasting
curl http://127.0.0.1:8890/v1/health
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8890/v1/manifest
```

From another machine (lan): `curl --cacert <certificate> https://192.168.1.20:8890/v1/health`.
Over the internet: `curl -H "CF-Access-Client-Id: …" -H "CF-Access-Client-Secret: …" https://aha-open.example.com/v1/health`.

## 6. When the worker process dies

A job can fail with a message such as `while loading the model: the worker crashed in native code (access violation) (exit
code 3221225477 = 0xC0000005). The request is not the cause; try again, and if it repeats report it with that log.
worker-qwen_image.log in the server's logs folder shows where it stopped.` The error `code` stays `generation_failed`.

- The message names the cause the operating system reported (a crash, or running out of memory) and gives different advice
  for each: memory problems say to free memory; a crash says the request is not the cause. The client is told only the log's
  **file name**, never a server path.
- The log is `<data dir>/logs/worker-<family>.log`. While a model loads the worker writes lines like
  `[worker +  41.2s rss=14.3GiB ram_available=9.8GiB] load text_encoder: weights read` (seconds since the worker started,
  the worker's memory, and the RAM still free), so a crash shows **how far loading got and how much RAM was left**. If the
  process dies natively, `faulthandler` also writes the Python stack of every thread (`Windows fatal exception: access
  violation` / `Fatal Python error: Segmentation fault`), which says which line was running.
- Memory is thin on a 32 GiB machine: loading the INT8 Qwen build makes the worker hold about **17.6 GiB** of RAM (measured on
  an RTX 3090 Ti machine, 31.7 GiB RAM), which left 1.8–4.5 GiB free while other programs used the rest. With less free RAM the
  load still finishes, only slower (132 s instead of 87 s in a test with 5.8 GiB free at the start), because Windows moves the
  worker's memory to the page file; keep free space on the drive that holds the page file.
- A native crash while loading was seen once (exit code 3221225477, cold first load) and could not be reproduced afterwards, so
  its cause is not known; the lines above are what will show it if it happens again.
- Nothing is retried automatically: a retry costs GPU minutes, so the client decides. The next job starts a fresh worker.

### 6.1 A full GPU on Windows: slow instead of an error (MiniMax H3)

On Windows the NVIDIA driver can move GPU memory to system RAM when the card is full, instead of reporting "out of memory".
Nothing fails; the work just becomes several times slower. Measured on an RTX 3090 Ti with MiniMax H3 at 1344×768: one
transformer block took **3.95 s with the card full** and **1.19 s with room left** (same code), which turned a 20-minute clip
into 40 minutes. The MiniMax worker therefore caps PyTorch at the memory that is physically free when a job starts (minus
0.5 GiB) — going over is then an out-of-memory error with a clear message — and computes its feed-forward layers in slices
so that a 1344×768 clip stays under that cap. Close other programs that use the GPU (games, other AI tools) before a long clip.
The NVIDIA Control Panel setting *CUDA – Sysmem Fallback Policy: Prefer No Sysmem Fallback* makes the driver itself refuse to
spill; it is optional.
