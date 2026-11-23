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
- The `internet` profile adds two Cloudflare Access headers: `CF-Access-Client-Id` and `CF-Access-Client-Secret`
  (Cloudflare checks them at the edge; AHA open does not need to read them).
- Every error: an HTTP status plus `{"error": {"code": "<code>", "message": "<sentence for a person>", "retry_after_s": <number|null>}}`.
- A request carrying a browser `Origin` header is rejected with `403`, unless the operator has explicitly allowed it.
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
- A model that offers variants (the field is general; Qwen-Image-2.1 offers `int8` only) also declares `variant` (the variant running on this server) and
  `variants` = `{ "<name>": {"label", "disk_gib", "vram_gib": {"1k", "native"}, "selected", "recommended"} }`.
  `recommended` is a hint derived from the server's hardware (display only; the server never switches variant by itself).
  `vram_gib` holds measured numbers (`null` = not measured yet).
- Video capabilities: `durations_s` (a list or `{min, max}`), `fps`, `short_edge`, `aspect_ratios`,
  `audio` (`{channels, sample_rate}` or `null`), `first_frame`, `last_frame`, `max_reference` (`{image, video, audio, total}`).
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
- `loras`: `[{"id": "<id from the manifest>", "scale": 0.8}]` (at most `capabilities.loras.max`, `scale` defaults to 1.0, no
  repeated id). An unknown id, a `scale` out of range or an extra field (e.g. `path`) ⇒ `400 invalid_request` before queueing. A job
  without `loras` gives a result **identical** to a server that never loaded a LoRA.
- Video: `params` holds `aspect_ratio`, `duration_s`, `short_edge`, `seed`.
- `client_request_id` (optional): sending the same value again within 10 minutes returns the original job and does not run it twice.
- A parameter outside `capabilities` ⇒ `400 invalid_request` **before** queueing; the server never silently stretches or resizes.

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
video adds `duration_s`, `fps`, `has_audio`.

- A job can be read or cancelled only by the token that created it; any other token gets `404` (its existence is not revealed).
- Cancellation takes effect at the next step boundary; after `cancelled` there is never a result.
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
