# Concepts — how the AHA open API fits together

Read this once before you build anything. Field-by-field definitions are in [`../03_api_v1.md`](../03_api_v1.md); this page explains the
*shape* of the API so those definitions make sense.

## 1. Where to send requests

The base URL depends on how the person who runs AHA open set it up (the *profile*, [`../03_api_v1.md`](../03_api_v1.md) §9 and
[`../06_deployment.md`](../06_deployment.md)):

| Profile | Base URL | Token | Extra |
|---|---|---|---|
| `local` (default) | `http://127.0.0.1:8890` | the local token file, or one made with `aha-open clients create` | none |
| `lan` | `https://<server-ip>:8890` | one token per computer | HTTPS with the server's own certificate: trust its certificate or compare its fingerprint |
| `internet` | `https://<your-name>.<domain>` (through a Cloudflare Tunnel) | one token per client | two more headers, `CF-Access-Client-Id` and `CF-Access-Client-Secret` |

A client accepts plain `http://` only for `127.0.0.1` / `localhost`. Anything that leaves the machine must be HTTPS.

## 2. Who you are: the token

Send `Authorization: Bearer <token>` on every request. Only three routes work without it: `GET /v1/health`, `POST /v1/enroll` and
`GET /v1/enroll/{id}` (pairing, [03_build_a_client.md](03_build_a_client.md)). Everything else, `GET /v1/manifest` included, answers
`401` without a valid token.

- A token identifies **one client**. Your jobs and uploads belong to your token: another token asking for your job gets `404`, as if
  it never existed.
- The server stores only a hash of the token; if you lose it nobody can show it to you again. Make a new one and revoke the old.
- Treat it like a password: not in source code, not in logs, not in a URL.

## 3. The manifest: what this server can do right now

`GET /v1/manifest` returns the server's models and limits. **Always read it, never hard-code it.** Two servers, or one server on two
days, offer different things. It lists only the models the operator configured. For each model:

| Field | Use it to… |
|---|---|
| `id` | name the model in a job |
| `state` | `ready` can run now · `loading` wait · `not_installed` the operator must install it · `unsupported_hardware` this GPU cannot run it · `error` |
| `state_detail` | a sentence for a person explaining a state other than `ready` (may be `null`) |
| `tasks` | `image.generate`, `image.edit`, `video.generate`, … |
| `variant`, `variants` | which build of the model runs on this server and what each build costs; display only, a job cannot choose one |
| `capabilities.sizes` | the exact sizes accepted (each has `width`, `height`, `aspect_ratio` and maybe a `tier`). For video each also has `short_edge`: you choose one by `aspect_ratio` + `short_edge`, never a free width and height |
| `capabilities.steps` | `min`, `max`, `default` number of steps |
| video: `durations_s`, `fps`, `frame_grid`, `audio`, `first_frame`, `last_frame`, `max_reference`, `reference_required` | how long a clip can be (a duration is rounded **up** to the model's frame grid: MiniMax H3's 5 s is 124 frames = 5.17 s), its frame rate and whether it has sound, and whether you may give a first or last picture |
| `variants.<name>.vram_gib` | measured GPU memory per size tier, so you can warn before a job that the card is too small |
| `capabilities.mask`, `rgba`, `max_reference_images`, `max_outputs`, `loras` | what the model can do beyond plain generation |
| `license` | what the model's license allows. Show `name` and `commercial_use` to your users; if `required_ui_label` is present you **must** show that exact text whenever the user picks the model |

`limits` tells you `part_size` (for uploads), `max_upload_bytes`, `max_queue_per_client` and `result_ttl_s` (how long results are kept).

## 4. Jobs

A job is "run this task with these inputs". You create it, then ask about it until it ends.

```text
queued → loading_model → running → succeeded
                                 ↘ failed
(any state before the end) → cancelled
```

- `POST /v1/jobs` answers `202` at once. **It never waits for the picture or the video.** A video is slow: MiniMax H3 on an RTX 3090 Ti
  takes about 5 minutes at its `preview` size and 22 minutes at `native`, so give your own wait a limit of at least 40 minutes and
  keep the job id (see [03_build_a_client.md](03_build_a_client.md) §5).
- `GET /v1/jobs/{id}` gives `status`, `progress` (`step`, `total_steps`, `fraction`) and, in `queued`, `queue_position`.
- When `succeeded`, `results[]` lists each output with `width`, `height`, `bytes`, `sha256` and the `seed` that made it.
  `GET /v1/jobs/{id}/result?index=0` downloads it (`Range` works).
- `POST /v1/jobs/{id}/cancel` stops it. A cancelled job **never** has a result.
- Results are kept for at least `limits.result_ttl_s` seconds, then deleted. Download what you need.
- `client_request_id` makes creating a job safe to repeat: send the same value again within 10 minutes (after a timeout, a retry) and
  you get the original job back, not a second one.
- The `seed` is the recipe for the picture. Leave it out and the server picks one and reports it in `results[].seed`; send the same
  `seed` with the same other settings later to reproduce it.

**Poll politely.** Start at a quarter of a second and slow down to about two seconds. A job that takes a minute does not need ten
requests per second.

## 5. Inputs: uploads

To edit a picture you first **upload** it, then name the upload in the job (`inputs: [{"role": "source", "upload_id": "up_…"}]`).
Uploads are chunked so that a big file does not hit the body-size limit of a proxy:

1. `POST /v1/uploads` with the file's `bytes`, `sha256` and `media_type` → `upload_id` and `part_size`.
2. `PUT /v1/uploads/{id}/parts/{n}` for each part. **Numbering starts at 0; every part is exactly `part_size` bytes except the last,
   which is the remainder.** A file smaller than `part_size` is one part, number 0.
3. `POST /v1/uploads/{id}/complete` — the server checks the sha256.

Roles: `source` (the picture to edit), `reference` (a picture to take style or content from), `mask` (only if the model's
`capabilities.mask` is true — Qwen-Image-2.1's is **false**), `first_frame` / `last_frame` / `reference_video` / `reference_audio` (video; only when the model's
`first_frame`, `last_frame`, `max_reference` say so). MiniMax H3: `minimax-h3-fl2va` takes text only, `minimax-h3-ref2va` takes one
`reference` image, and neither takes a first or last frame (`400`).
Details: [`../03_api_v1.md`](../03_api_v1.md) §2 and §5, and per model [07_models.md](07_models.md).

## 6. Errors

Every error has an HTTP status and one JSON shape:

```json
{"error": {"code": "busy", "message": "this client already has the maximum number of waiting jobs", "retry_after_s": 10}}
```

- Branch on `code`, show `message` to a person. When `retry_after_s` is a number, wait that long (the `Retry-After` header carries it too).
- A job that was accepted can still **fail** later. That is an error *inside* the job (`status: "failed"`, `error: {code, message}`),
  not an HTTP error: `gpu_out_of_memory`, `generation_failed`, `user_cancelled`.
- Bad parameters are refused with `400` **before** anything is queued. The server never silently changes your size or settings.
- Table of every code and what to tell the user: [05_errors_and_troubleshooting.md](05_errors_and_troubleshooting.md).

## 7. What is *not* part of the API

Only paths under `/v1` are the API. The web page at `/ui/` and the operator's `/admin/*` pages exist on the machine that runs AHA open
but are not a contract: do not call them from your app, and expect `404` from `/admin/*` when you are not the person at that machine.

## 8. A few rules AHA open keeps, so you can rely on them

- It never stretches or resizes a result: the picture has exactly the width and height you asked for (a worker that returns another size makes the job `failed`).
- It never sends your prompts or pictures anywhere else. No telemetry.
- It never re-runs a failed job by itself; the client decides.
- A cancelled job never has a result.
