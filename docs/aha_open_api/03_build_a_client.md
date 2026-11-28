# Build a client — the flow every app follows

For developers writing an app (desktop, mobile, server, script) that uses AHA open. Read [02_concepts.md](02_concepts.md) first. A complete,
tested client in about 200 lines of standard-library Python is [examples/python/aha_open_client.py](examples/python/aha_open_client.py);
a one-file Node version is [examples/node/generate.mjs](examples/node/generate.mjs). Definitions are in [`../03_api_v1.md`](../product/03_api_v1.md).

## The flow

```text
 configure ─► manifest ─► (upload inputs) ─► create job ─► poll ─► download result
   token        pick a model     parts from 0      202 + id     backoff      verify, keep
   base URL     check sizes      sha256            idempotent   show progress
                show license                       client_request_id          cancel any time
```

## 0. Configure

- **Base URL and token come from settings, not from code.** The user (or the operator) tells your app where the server is and gives it a token.
- **Store the token like a password**: the operating system's credential store, or a file only the user can read. Never in source code,
  a URL, a log line, a crash report or an analytics event. (AHA open stores only a hash of it; a lost token cannot be recovered, only replaced.)
- **Plain `http://` only for `127.0.0.1` / `localhost`.** Refuse it for anything else.
- Send `Authorization: Bearer <token>` on every request except `GET /v1/health`.
- Do not send an `Origin` header (a native HTTP client does not). See "Web apps" below.

## 1. Health (optional, cheap)

`GET /v1/health` needs no token and answers in milliseconds: `ok`, `degraded` (a worker is broken) or `starting`. Use it to tell
"the server is off" (no answer) from "the server is on but rejects my token" (answer here, `401` on the manifest). Those two need different
sentences for the user.

## 2. Manifest: choose a model, then build your form from it

`GET /v1/manifest`. Never hard-code what it says.

1. Keep models whose `state` is `ready` and whose `tasks` include what you want. If none, tell the user which model is in which state
   (`not_installed` → the operator must install it; `loading` → wait; `unsupported_hardware`; `error`).
2. **Sizes:** offer only `capabilities.sizes`. Send `width` and `height` together exactly as listed, or `aspect_ratio` (the server takes the first
   matching entry, the smaller "1k" tier). A size not in the list is refused with `400` — the server never stretches. Entries with `tier: "native"`
   are the model's own larger size and need far more VRAM and time (for Qwen-Image-2.1: measured 269–326 s per image and 26.9 GiB, [`../05_models.md`](../product/05_models.md) §3.1).
   **Video sizes:** a video model lists its sizes the same way, with a `short_edge` in each entry. Send `aspect_ratio` and `short_edge` of the entry the
   user chose (MiniMax H3: `16:9` with `short_edge` 768 → 1344×768 `native`, or 352 → 608×352 `preview`) and keep `duration_s` inside
   `durations_s` (MiniMax H3: exactly 5). Pair or length not on offer ⇒ `400` listing what is. Show the user a time estimate with the
   size: about 5 minutes for `preview`, 22 minutes for `native` on an RTX 3090 Ti (measured, [`../05_models.md`](../product/05_models.md) §2.3);
   `variants.<name>.vram_gib` says how much GPU memory each tier needs. **Always send `short_edge`:** left out, `minimax-h3-fl2va`
   makes its largest size. `minimax-h3-ref2va` needs one reference image and offers `preview` only ([07_models.md](07_models.md)).
3. **Steps:** `capabilities.steps` gives `min`, `max`, `default`. **References:** at most `max_reference_images`. **Mask / transparency / LoRA:** only when
   `mask`, `rgba`, `loras` say so. Hide the control when the model cannot do it; do not send it and wait for a `400`.
4. **License:** show `license.name` and `commercial_use` wherever the user picks a model. If `required_ui_label` exists, show exactly that text
   when the model can be picked (for example `MiniMax H3`). Qwen-Image-2.1 is `not_allowed` for commercial use: say so before the user builds a business on it.
   MiniMax H3 also lists `license.excluded_territories` (`EU`, `GB`, `KR`, `US`): show it, because the licence does not allow use there.
   The video file carries the comment `AI-generated (MiniMax H3)`; keep it, and tell viewers a clip is machine-made when you publish it.
5. Re-read the manifest when the user opens the model picker, after a `409 model_not_installed`, and now and then while the app is open:
   states and the LoRA list change while the server runs, and adding a LoRA file needs no restart.

## 3. Uploads (only for edits and video frames)

```text
POST /v1/uploads                  {"bytes": 3500, "sha256": "<hex of the whole file>", "media_type": "image/png"}
  → 201 {"upload_id": "up_…", "part_size": 52428800, "bytes": 3500}
PUT  /v1/uploads/{id}/parts/0     <the first part_size bytes>        → 204
PUT  /v1/uploads/{id}/parts/1     <the next …, the last part is the remainder>
POST /v1/uploads/{id}/complete    → 200 {"status": "complete"}
```

- Parts are numbered **from 0**, and each is **exactly** `part_size` bytes except the last (the remainder). A file smaller than `part_size` is one part, number 0.
  Short parts, long parts and a number out of range are `400`.
- A part can be sent again (the new bytes replace the old). After `complete` it cannot.
- A wrong sha256 at `complete` → `400 upload_incomplete` **and the upload is thrown away**: start over with a new `POST /v1/uploads`.
- An upload can only be used by the token that made it. It expires after a while (set by the operator, one hour by default), used or not,
  so create the job soon after. A job already accepted keeps its inputs ([08_limits_and_behaviour.md](08_limits_and_behaviour.md) section 2).
- `media_type` must be `image/*`, `video/*` or `audio/*`; each role wants the right family (a `source` must be an image).
- A client may have only a limited number of stored uploads; beyond that you get `429 busy`.

## 4. Create the job

```json
POST /v1/jobs
{"task": "image.generate", "model": "qwen-image-2.1", "prompt": "a lighthouse at sunset",
 "params": {"width": 1024, "height": 1024, "seed": 42},
 "client_request_id": "6f1c2f0e9a3b4d27"}
```

- **Always send `client_request_id`** (a random string per *logical* request, kept when you retry). If the request times out and you do not know
  whether it arrived, send it again: within 10 minutes the server returns the original job instead of making a second one (which would cost GPU minutes).
  It does so even if the body differs or that job failed, so a person's *Try again* needs a **new** id.
- `202` → keep `id`. Save it **before** you start polling (to disk if the app might close): a job survives your app, a lost id does not.
- `400 invalid_request` → a parameter is outside `capabilities`, or something is missing. It is **before** queueing; fix it, do not retry unchanged.
- `409 model_not_installed` / `unsupported_hardware` → tell the user; retrying will not help.
- `429 busy` → **wait `retry_after_s`** (also the `Retry-After` header), then send the same request again. `busy` is not only the per-client
  queue limit in the manifest: the server's whole queue, a per-minute job rate and the number of stored uploads can also answer it, and those
  numbers are the operator's, not published. Never retry in a tight loop.
- Never retry a **failed job** by yourself: it cost GPU time, and the cause (out of memory, a crashed worker) usually repeats. Show the message and let the user decide.

## 5. Poll

`GET /v1/jobs/{id}` until `status` is `succeeded`, `failed` or `cancelled`.

| `status` | Show the user |
|---|---|
| `queued` | "Waiting" + `queue_position` (it is `null` when not queued) |
| `loading_model` | "Loading the model" (the first job after a start or a model change takes about two minutes on the test machine; this is normal) |
| `running` | A progress bar from `progress.fraction` (`step` of `total_steps`) |
| `succeeded` | The result |
| `failed` | `error.message` (a person-readable sentence) — see [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md) |
| `cancelled` | "Cancelled" |

- Poll with a growing pause: 0.25 s, then ×1.5 up to about 2 s. AHA open does not rate-limit reading a job (only creating jobs and uploads is limited), but a
  job that takes a minute does not need ten requests a second, and a shared server is kinder to everyone when you back off.
- If your app (or the server) restarts, poll the saved job id again. A job that was running when the **server** restarted is `failed` with the message
  "the server restarted while this job was waiting or running"; create it again (same inputs) only if the user wants.
- Give up *waiting* before giving up on the job: after your own timeout, tell the user it is still running and let them keep waiting or cancel.
  For video make that timeout long (40 minutes or more for MiniMax H3 at `native`) and show `progress.fraction`: it moves once per sampling
  step (19 of them), about every 9 s at `preview` and every 60 s at `native`, and stays at 0 for the first 1–2 minutes while the model loads.
- Cancelling a video: the server tells the worker, which stops at its next sampling step; if it has not stopped within `cancel_grace_s`
  (15 s by default) the server stops the worker process itself. Either way the job ends `cancelled` within about 15 s and never has a
  result. (From the server's code; a cancel of a `native` MiniMax H3 clip has not been timed yet. The next job simply starts a fresh worker.)

## 6. Download the result

`GET /v1/jobs/{id}/result?index=0` streams the bytes. `Range` is supported, so a big video can resume.

- Check what you got against `results[i]`: `bytes`, `sha256`, and for images `width`/`height`. They are exact; the server never resizes.
- Save it before `limits.result_ttl_s` runs out (at least that long after completion; then it is deleted and the route answers `404`).
- Asking for the result of a job that is not `succeeded` is `409 job_not_ready`. A cancelled job never has a result.
- `results[].seed` is the recipe: store it with the picture to reproduce it. `results[].loras` lists the LoRAs used, with their hashes.

## 7. Cancel

`POST /v1/jobs/{id}/cancel` stops a queued or running job. The server waits a few seconds for the job to actually stop before it answers, so the body
is normally `cancelled`; if it still says `running`, keep polling until it is final. Cancelling twice, or cancelling a finished job, is harmless.

## Web apps (browsers): the one that surprises everyone

A page in a browser always sends an `Origin` header, and **AHA open answers `403` to any request with an `Origin` header** that the operator has not
allowed (a hostile web page must not be able to use your GPU; the only exception is the page AHA open serves itself, `/ui/`). It also offers no CORS. Your choices:

1. **A small backend of your own** talks to AHA open and your page talks to that backend. This is the normal answer: the token stays on the server,
   out of the page.
2. The operator can list your page's address in `allowed_origins` in AHA open's `config.toml`. Only do this for a page you control, on a network you trust,
   and remember the token would then live in the page.
3. A desktop app (Electron, Tauri) that makes its requests from its main process, not from a web page, sends no `Origin` and needs neither.

Also: a request whose `Host` is not one of the names the server answers to is `403 forbidden` ("this Host is not served here"). Use the address the operator gave you.

## Other networks

- **`lan`:** HTTPS with the server's own certificate. Your HTTP library must trust that certificate (give it the `.crt` file, as `Client(cafile=…)` does), or show
  the user the certificate's SHA-256 fingerprint to compare with the one `aha-open tls fingerprint` prints on the server. Never switch certificate checking off.
- **`internet`:** also send `CF-Access-Client-Id` and `CF-Access-Client-Secret` (a Cloudflare Access service token). A `403` or a redirect *before* AHA open
  answers means Cloudflare stopped you, and `aha-open check <url> --cf-id … --cf-secret … --token …` tells you which layer.
- **Pairing (optional, only when `GET /v1/manifest` says `optional.enroll` is true):** instead of typing a token, a client can ask to be approved.

  ```text
  POST /v1/enroll   {"client_name": "Laptop An", "client_kind": "my-app"}
    → 201 {"id": "enr_…", "enroll_code": "K7Q4-2M", "expires_at": "…"}      show the code to the user
  (the operator runs:  aha-open clients approve K7Q4-2M)
  GET /v1/enroll/enr_…  →  {"status": "pending"} … then {"status": "approved", "token": "…"}   (the token is returned ONCE)
                           or {"status": "denied"} / {"status": "expired"}
  ```

  Store the token the moment you receive it; a second poll does not return it again. Pairing is off by default on `internet`.

## What to tell the user when it goes wrong

Map the `code`, never the status alone: [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md) has a sentence for each, and the causes that look alike
("the server is off", "the token is wrong", "the model is loading") get different sentences on purpose.

## Checklist

- [ ] Base URL and token are settings; the token is never logged.
- [ ] Model, sizes, steps, references, mask, transparency and LoRA all come from the manifest; unavailable controls are hidden.
- [ ] License name and `commercial_use` are shown; `required_ui_label` is shown verbatim when present.
- [ ] Every job has a `client_request_id`; the job id is saved before polling.
- [ ] `busy` waits `retry_after_s`; a failed job is never re-run automatically.
- [ ] Polling backs off; the app survives its own restart by polling the saved id.
- [ ] Results are saved before they expire; the size and sha256 are checked.
- [ ] A cancel button exists and works on queued and running jobs.
- [ ] No `Origin` header (backend or native HTTP), HTTPS off-machine, certificate checking on.
