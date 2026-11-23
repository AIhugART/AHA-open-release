# Deploying AHA open — three profiles

> The `aha-open` commands below exist (🧪 tested; the `lan` profile has run for real over loopback but not yet from a second machine;
> `internet` has not been tried with a real Cloudflare account). The examples use made-up addresses/domain names.

## 0. Common requirements

- An NVIDIA GPU + CUDA driver (macOS/Apple Silicon is not part of v1; to be measured).
- Disk: ~19 GB for Qwen INT8. MiniMax H3 (community quantized build only, not installable yet) needs about 15–20 GB for the model plus ~5 GB of VAEs and a text encoder — see [05_models.md](05_models.md) §2.0.
- Python + [`uv`](https://docs.astral.sh/uv/) (AHA open creates a separate venv for each worker from the worker's lock file).

```bash
uv sync --extra download    # from a source checkout (not on PyPI yet); or install a wheel built with `uv build`
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research            # the INT8 build, ~18 GB
# MiniMax (when its worker ships): also asks you to confirm you are not in an excluded territory
# aha-open models install minimax-h3-fl2va --accept-license minimax-h3-community --confirm-territory
aha-open doctor
```

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
