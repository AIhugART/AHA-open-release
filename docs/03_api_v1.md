# `/v1` API specification (draft)

> This standard is **neutral**: any service can implement it; AHA open is one implementation. The machine-readable source of
> truth is [`schema/aha_provider_v1.openapi.yaml`](../schema/aha_provider_v1.openapi.yaml).
>
> **The single original of the `/v1` standard lives in this repository**. Clients — AHA Studio included — only
> link here and pin the version of the specification they support; they keep no copy of their own.
>
> English is the canonical language of this document. A Vietnamese translation is in [vi/03_api_v1.md](vi/03_api_v1.md)
> and may lag behind.

## 1. General conventions

- JSON, UTF-8; times are ISO 8601 UTC (`2026-11-22T10:00:00Z`).
- Authentication: `Authorization: Bearer <client_token>` — required on the `lan`/`internet` profiles; on `local` it is
  configurable (on by default).
- Only `GET /v1/health`, `POST /v1/enroll` and `GET /v1/enroll/{id}` can be called without a token. Every other route —
  `GET /v1/manifest` included — answers `401 unauthorized` without a valid token.
- The `internet` profile adds two Cloudflare Access headers: `CF-Access-Client-Id` and `CF-Access-Client-Secret`
  (Cloudflare checks them at the edge; AHA open does not need to read them).
- Every error: an HTTP status plus `{"error": {"code": "<code>", "message": "<sentence for a person>", "retry_after_s": <number|null>}}`.
- When `retry_after_s` is a number, the response also carries the HTTP header `Retry-After` with the same value in whole
  seconds, rounded up.
- A request carrying a browser `Origin` header is rejected with `403`, unless the operator has explicitly allowed it.
- Only routes under `/v1` belong to this standard. A server may offer other paths (an implementation's own web page, an
  operator-only `/admin/*`); a client must never depend on them, and a server may answer `404` to them.
- Clients must ignore fields they do not know (so that adding a field does not break compatibility).

## 2. Routes

| Route | Purpose |
|---|---|
| `GET /v1/health` | Alive/dead, GPU state, loaded model, queue length |
| `GET /v1/manifest` | Models, tasks, capabilities, licenses, limits, optional features |
| `POST /v1/uploads` | Create an upload (declare `bytes`, `sha256`, `media_type`) → `upload_id`, `part_size` |
| `PUT /v1/uploads/{id}/parts/{n}` | Send part `n` (≤ `part_size`) |
| `POST /v1/uploads/{id}/complete` | Close the upload; the server checks the sha256 |
| `POST /v1/jobs` | Create a job → `202` + `job_id` immediately |
| `GET /v1/jobs/{id}` | Status, progress, result metadata |
| `POST /v1/jobs/{id}/cancel` | Cancel |
| `GET /v1/jobs/{id}/result?index=N` | Download a result as a stream, `Range` supported |
| `POST /v1/enroll` · `GET /v1/enroll/{id}` | Pairing (optional, declared in the manifest) |

Chunked upload exists because a large video/image input can exceed the request-body limit of a proxy (for example Cloudflare).

Upload rules (a client that breaks one gets `400 invalid_request` or `400 upload_incomplete`):

- Parts are numbered **from 0**. An upload of `bytes` bytes has `max(1, ceil(bytes / part_size))` parts, numbered `0` to
  `parts - 1`; any other number is refused.
- Every part must be **exactly** `part_size` bytes, except the last, which is exactly the remainder
  (`bytes - part_size × (parts - 1)`). A shorter or longer part is refused.
- A part may be sent again before `complete`; the new bytes replace the old ones. After `complete`, sending a part is refused.
- `complete` with a missing part, or with a file whose size or sha256 differs from the declared values, answers
  `400 upload_incomplete`; in the sha256 case the server discards the upload, so the client starts again with a new `POST /v1/uploads`.
  `complete` on an upload that is already complete answers `200` again.
- An upload can be used only by the token that created it (any other token gets `404`). An upload is deleted a server-set time
  after it was created, whether or not a job used it (operator setting; not in the manifest).
- A job that was accepted keeps its inputs: deleting the upload never takes them away from a queued or running job.

## 3. Health

```json
GET /v1/health
{"status": "ok", "api_version": "1.0",
 "gpus": [{"index": 0, "name": "…", "memory_total_mib": 24564, "active_model": "qwen-image-2.1"}],
 "queue": {"queued": 2, "running": 1}}
```

`status`: `ok` · `degraded` (a worker is broken) · `starting`.

## 4. Manifest

```json
{
  "api_version": "1.0",
  "provider": {"name": "example-provider", "version": "0.1.0"},
  "profile": "lan",
  "limits": {"part_size": 52428800, "max_upload_bytes": 1073741824, "max_queue_per_client": 4, "result_ttl_s": 3600},
  "optional": {"enroll": true},
  "models": [
    {
      "id": "qwen-image-2.1",
      "display_name": "Qwen-Image-2.1",
      "engine": "diffusers",
      "tasks": ["image.generate", "image.edit"],
      "state": "ready",
      "capabilities": {
        "sizes": [{"aspect_ratio": "1:1", "width": 2048, "height": 2048}],
        "max_reference_images": 10,
        "mask": true,
        "rgba": true,
        "max_outputs": 1
      },
      "license": {"name": "Qwen RESEARCH LICENSE", "commercial_use": "not_allowed", "url": "https://…"}
    }
  ]
}
```

- `state`: `ready` · `loading` · `not_installed` · `unsupported_hardware` · `error`.
- `license.commercial_use`: `allowed` · `allowed_with_conditions` · `not_allowed` · `unknown`.
- Optional (additive) license fields: `excluded_territories` (region codes, e.g. `["EU","GB","KR","US"]`),
  `required_ui_label` (a string the client **must** show verbatim when it lets the user pick the model), `notes` (one sentence for the user).
  A client that does not understand a field still shows `name` + `commercial_use`.

MiniMax example:

```json
"license": {"name": "MiniMax H3 Community License", "commercial_use": "allowed_with_conditions",
            "excluded_territories": ["EU", "GB", "KR", "US"], "required_ui_label": "MiniMax H3", "url": "https://…"}
```
- An entry of `capabilities.sizes[]` may carry a `tier` (`"1k"` ≈ 1 MP, fast; `"native"` ≈ 4 MP, the model's own size, which
  may need far more VRAM). When only `aspect_ratio` is given the server picks the **first matching entry** (the `1k` tier); for the
  native size send `width` + `height`. Only sizes for which an image has **actually been generated** are declared.
- A model that supports LoRA declares `capabilities.loras` = `{"max": 3, "scale": {"min", "max", "default"}, "format": "safetensors",
  "available": [{"id", "display_name", "default_scale", "bytes", "trigger?", "license?", "notes?"}]}`. `available` lists files the
  **operator placed on the server** (clients do not upload LoRAs through `/v1`), re-read on every manifest call — adding
  a file needs no restart. A path is never exposed.
- A model that offers variants (the field is general; Qwen-Image-2.1 offers four: `int8`, `int8-convrot`, `mixed-p010`, `mixed-p015`; MiniMax H3 offers `int8`, `mixed`, `nvfp4`) also
  declares `variant` (the variant running on this server) and
  `variants` = `{ "<name>": {"label", "disk_gib", "vram_gib": {"1k", "native?"}, "ram_gib", "seconds": {"1k"}, "loras", "quality_note",
  "installed", "fit", "selected", "recommended"} }` (all additive).
  `recommended` is a hint derived from the server's hardware (display only; the server never switches variant by itself).
  `vram_gib`, `ram_gib` and `seconds` hold measured numbers (`null` = not measured yet; a number that was not measured never refuses
  anything). `installed` says the variant's weights are complete on the server's disk. `fit` is `fits` | `tight` | `too_small` | `unknown`,
  from the measured 1024x1024 peak against this server's GPU (a video model: the peak of its smallest measured size). "This server's
  GPU" is the card, or the operator's `[gpu] memory_limit_gib` when that is smaller. A video model's variant also carries
  `fit_by_tier` = `{ "<tier>": "fits"|"tight"|"too_small"|"unknown" }`, the same judgement for each size tier, so a client can leave
  out the sizes this machine cannot make (an image model has no such field: its peaks are taken with offload on and are not a minimum).
  `too_small_up_to_gib` = `{ "<tier>": G }` (optional): that size was tried on a G GiB card and ran out of memory, so its
  `vram_gib` is `null`; a card of that size or smaller is `too_small`, a larger one `unknown`. `loras` is `true` when a real LoRA run succeeded on that variant, `null`
  when none was tried. `quality_note` is one sentence about the variant's quality, quoting its publisher's own review.
- Video capabilities: `durations_s` (a list or `{min, max}`), `fps`, `short_edges`, `aspect_ratios`,
  `audio` (`{channels, sample_rate}` or `null`), `first_frame`, `last_frame`, `max_reference` (`{image, video, audio, total}`).
  A video model may also declare (additive): `sizes` = `[{"aspect_ratio", "short_edge", "width", "height", "tier"}]`, the exact
  output sizes (only pairs that have been generated; a pair not listed is refused), `frame_grid` = `{"step", "base"}` (the model
  makes `base + n * step` frames: a duration is rounded up to the next such count, e.g. MiniMax H3's 5 s at 24 fps is 124 frames =
  5.17 s) and `steps` (`{min, max, default}` as for images).
  **MiniMax H3** (`minimax-h3-fl2va`) declares only what has been run on the reference machine: `durations_s` `{min: 5, max: 5}`,
  `aspect_ratios` `["16:9"]`, sizes 1344×768 (`native`, `short_edge` 768) and 608×352 (`preview`, `short_edge` 352), `steps` 20,
  `first_frame: false`, `last_frame: false` (a request with a frame is `400`), audio 2 channels at 32 kHz. Its `variants.<name>.vram_gib`
  gives the measured peaks per size (`mixed` 16.7 / 19.6, `nvfp4` 13.1 / 16.4, `int8` 21.4 / ran out of memory, so
  `too_small_up_to_gib.native` is 24); it is offered from 16 GB cards (`unsupported_hardware` below that). A clip takes minutes
  (about 5 minutes at `preview`, 23 minutes at `native` on an RTX 3090 Ti): always create the job and poll it. `license.required_ui_label` is `"MiniMax H3"` and must be shown.
  **MiniMax H3 references** (`minimax-h3-ref2va`, a separate model) has the same clip length, steps, audio and licence, offers **only the `preview` size (608×352, `short_edge` 352)** (native 1344×768 has not been run with a reference), and declares
  `reference_required: true` (a request with no reference image or video is `400`; use `minimax-h3-fl2va` for text only) and
  `max_reference` `{image: 1, video: 0, audio: 0, total: 1}`: only one reference image has been run for real, so only that is accepted
  (a second image, a video or an audio reference is `400`). It offers the `mixed` and `nvfp4` builds and needs a card of 18 GB or
  more (no build ran with a reference on a 16 GB card). It is installed into the folder of `minimax-h3-fl2va` (they share the encoder
  and the VAEs), so it shows `not_installed` until its own transformer is there.
- Tasks defined in v1: `image.generate`, `image.edit`, `video.generate`, `audio.generate`, `speech.generate`.

## 5. Creating a job

```json
POST /v1/jobs
{
  "task": "image.edit",
  "model": "qwen-image-2.1",
  "prompt": "make the jacket red",
  "inputs": [
    {"role": "source", "upload_id": "up_…"},
    {"role": "reference", "upload_id": "up_…"},
    {"role": "mask", "upload_id": "up_…"}
  ],
  "params": {"width": 2048, "height": 2048, "seed": 42, "num_outputs": 1},
  "client_request_id": "c1f0…"
}
```

- `role`: `source` · `reference` · `mask` · `first_frame` · `last_frame` · `reference_video` · `reference_audio`.
- Images: `params` holds `width`+`height` (together) or `aspect_ratio`, `seed`, `steps`, `num_outputs`, and `transparent`
  (bool; only for models declaring `rgba: true`; the server adds a transparent-background description to the prompt). When `seed` is
  absent the server picks a random seed and records it in `results[].seed`; when `steps` is absent `capabilities.steps.default` is used.
- A model declaring `mask: false` rejects the `mask` role.
- Video references: send `reference` (image), `reference_video` (video) and `reference_audio` (audio) inputs **in the order the model should read them**:
  the order labels them in the prompt (`<Picture 1>`, `<Video 1>`, `<Audio 1>`) and sets the audio/video clock, so a different order is a different
  request. An upload must have the media type of its role (`image/*`, `video/*`, `audio/*`; otherwise `400`). An audio reference cannot be the
  only reference. Nothing is fetched from a URL by the server or the worker: a reference is always an upload.
- `params.loras` (inside `params`, like `seed`): `[{"id": "<id from the manifest>", "scale": 0.8}]` (at most `capabilities.loras.max`, `scale` defaults to 1.0, no
  repeated id). An unknown id, a `scale` out of range or an extra field (e.g. `path`) ⇒ `400 invalid_request` before queueing. A job
  without `loras` gives a result **identical** to a server that never loaded a LoRA.
- Video: `params` holds `aspect_ratio`, `duration_s`, `short_edge`, `seed`, and `steps` (only for a model that declares `capabilities.steps`;
  absent means its default). A model with `capabilities.sizes` gets the exact `width`, `height` and `num_frames` from that table; the
  client never sends a free width or height for video (`unknown params`).
- `client_request_id` (optional): sending the same value again within 10 minutes returns the original job and does not run it twice.
- A parameter outside `capabilities` ⇒ `400 invalid_request` **before** queueing; the server never silently stretches or resizes.
- `429 busy` is not only `limits.max_queue_per_client`. A server may also refuse because its whole queue is full, because
  the client created too many jobs in the last minute, or because too many uploads are stored for the client. Those numbers are
  the operator's configuration and are **not** in the manifest. A client waits for `retry_after_s` (and the `Retry-After`
  header) and tries again; it never retries in a tight loop.

## 6. Job lifecycle

```text
queued → loading_model → running → succeeded
                                 ↘ failed
(every non-final state) → cancelled
```

```json
GET /v1/jobs/job_…
{
  "id": "job_…",
  "status": "running",
  "progress": {"step": 12, "total_steps": 40, "fraction": 0.3},
  "queue_position": null,
  "results": [],
  "error": null,
  "created_at": "2026-11-22T10:00:00Z"
}
```

When `succeeded`, each element of `results[]` is:
`{"index": 0, "kind": "image", "media_type": "image/png", "width": 2048, "height": 2048, "bytes": 5123456, "sha256": "…", "seed": 42, "loras": [{"id": "ink", "scale": 0.8, "sha256": "…"}]}`
(the server rejects a PNG result whose size differs from the accepted size — the job is `failed`, never silently resized; Qwen
images are the RGBA PNG exactly as the model returns it);
video adds `duration_s`, `fps`, `has_audio`, and, for a model with `capabilities.sizes`, `width` and `height` (the server fails a job
whose worker reports another size than the accepted one, or none, exactly as it does for images).

- A job can be read or cancelled only by the token that created it; any other token gets `404` (its existence is not revealed).
- Cancellation takes effect at the next step boundary; after `cancelled` there is never a result.
- `POST /v1/jobs/{id}/cancel` answers `200` with the job. The server waits a short, bounded time (a few seconds) for the job to
  stop before it answers, so the body is normally `cancelled`; if the job has not stopped yet it is returned in its current
  state, and the client keeps polling `GET /v1/jobs/{id}` until the status is final. Cancelling a job that is already final
  changes nothing and answers `200` with that job.
- Results are kept for at least `limits.result_ttl_s` and then deleted.

## 7. Pairing (optional)

```text
POST /v1/enroll        {"client_name": "Laptop An", "client_kind": "aha-studio"}
  → 201 {"id": "enr_…", "enroll_code": "K7Q4-2M", "expires_at": "…"}
GET /v1/enroll/enr_…   → {"status": "pending"}
                       | {"status": "approved", "token": "…"}   (the token is returned exactly once)
                       | {"status": "denied"} | {"status": "expired"}
```

## 8. Error codes

| Code | HTTP | Meaning |
|---|---|---|
| `invalid_request` | 400 | Missing/wrong field, parameter outside `capabilities` |
| `unsupported_task` | 400 | The model does not do this task |
| `upload_incomplete` | 400 | The upload was not completed or its sha256 is wrong |
| `unauthorized` | 401 | Missing/wrong token |
| `forbidden` | 403 | No permission, `Origin` blocked |
| `not_found` | 404 | No such job/upload (or it does not belong to this token) |
| `job_not_ready` | 409 | Downloading the result of a job that has not `succeeded` (running, failed or cancelled) |
| `model_not_installed` | 409 | The model is not installed |
| `unsupported_hardware` | 409 | The hardware cannot run the model |
| `busy` | 429 | The client's or the server's queue is full; has `retry_after_s` |
| `model_loading` | 503 | The model is loading; has `retry_after_s` |
| `gpu_out_of_memory` | (in a job) | Out of VRAM |
| `user_cancelled` | (in a job) | Cancelled |
| `generation_failed` | (in a job) | Any other pipeline error |

## 9. Network profiles (`profile` in the manifest)

| Profile | Listens on | Protocol | Authentication |
|---|---|---|---|
| `local` | `127.0.0.1` | HTTP | token (on by default) |
| `lan` | an interface chosen by the operator | **HTTPS** | one token per client |
| `internet` | `127.0.0.1` (behind a Cloudflare Tunnel) | HTTPS at the edge | Cloudflare Access + one token per client |

A client (for example AHA Studio) accepts `http://` only for `127.0.0.1`/`localhost`.

## 10. Versioning rules

Adding a field/route/task/error code/profile = **minor** (`1.x`). Changing a meaning or removing one = **major** (`/v2`).
