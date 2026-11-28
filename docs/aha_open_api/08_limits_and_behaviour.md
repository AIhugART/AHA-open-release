# Limits and behaviour — what to design your app around

The things that do not show in a first happy-path test and bite later. They describe **AHA open** as it is now; the neutral standard is
[`../03_api_v1.md`](../product/03_api_v1.md).

## 1. Limits

The numbers are AHA open's defaults. The operator can change some of them, so **do not write them into your code**: read the ones in
the manifest, and for the rest react to the answer (`busy` → wait `retry_after_s`).

| Limit | Default | In the manifest? | When you hit it |
|---|---|---|---|
| Upload part size | 50 MiB | `limits.part_size` | `400` on a part of another size |
| Largest upload | 1 GiB | `limits.max_upload_bytes` | `400` |
| Waiting or running jobs per client | 4 | `limits.max_queue_per_client` | `429 busy`, retry after 10 s |
| Waiting or running jobs on the server | 32 | no | `429 busy`, retry after 15 s |
| Jobs created per client per minute | 30 | no | `429 busy`, retry after 30 s |
| **Stored uploads per client** | 32 | no | `429 busy`, retry after 30 s |
| How long a result is kept after the job ends | 3600 s | `limits.result_ttl_s` | `404 not_found` |
| How long an upload is kept after it is created | 3600 s | no | see section 2 |
| Longest a job may run | 3600 s | no | job `failed`, `generation_failed` |
| Longest a model may take to load | 900 s | no | job `failed` |
| Prompt | 8000 characters | no | `400` |
| `client_request_id` | 1–128 characters, remembered 600 s | no | `400` |
| Pairing code lifetime / requests waiting | 300 s / 5 | no | `expired` / `429 busy` |

There is no route that lists your jobs or uploads, and none that deletes them. Keep the ids yourself.

## 2. Uploads expire, accepted jobs keep their inputs

An upload is deleted a set time after it was **created**, whether or not a job has used it. A job that was accepted keeps its own
inputs, however long it waits in the queue.

- Create the job before the upload expires; afterwards its `upload_id` answers `404`.
- Within the lifetime you may use one `upload_id` in several jobs. Do that for a batch: the cap of 32 stored uploads counts every
  upload until it expires, used ones included.

**An older server** (a build from before 2026-11-26, which includes the `0.1.0.dev1` pre-release) lets a job
that is still queued lose its input when the upload expires: it ends `failed` with `generation_failed`. Against such a server upload
just before you create the job, and on that failure upload again and create a new job.

## 3. Retrying safely: `client_request_id`

Sending the same `client_request_id` again within 10 minutes returns the **original job**, with `202`, even when the rest of the body
is different, and even when that job has `failed` or was `cancelled`.

- One id per thing the person asked for; reuse it only to repeat the *same* request after a timeout or a `busy`.
- The person presses *Try again* or changes the prompt → **new id**. Otherwise they get the old job back.

## 4. Validation: what is refused, and what is filled in

Refused with `400` before anything is queued: an unknown field in `params`; a size, length, ratio or step count outside
`capabilities`; an input role the task or model does not take; an upload whose media type does not match its role; an unknown LoRA; a
model id the server has not configured or has switched off.

Filled in when you leave it out: image size (the first declared one), `steps` (the model's default), `seed` (random, reported back),
video `aspect_ratio`, `duration_s` (the shortest) and `short_edge` (**the first declared, which is the largest** for
`minimax-h3-fl2va`).

## 5. The queue

- Each GPU has one queue and runs one job at a time. `queue_position` starts at 1.
- Clients are served in turn: one app with many jobs does not starve another.
- The server prefers the model already in memory, but switches once a job for another model has waited 60 s.
- A failed job is never run again by the server.
- When the server restarts, jobs that were waiting or running become `failed` ("the server restarted…").
- `queued` lasts as long as the jobs ahead of you; behind videos that is tens of minutes.

## 6. Cancel

`POST /v1/jobs/{id}/cancel` waits up to about 6 s for the job to stop, then answers with its current state. Set your HTTP timeout above
that. If the answer is not final yet, keep polling. A running worker that ignores the cancel is stopped after 15 s. A cancelled job has
no result, ever.

## 7. HTTP details

- JSON bodies use `Content-Type: application/json`. A body that cannot be read is `400 invalid_request`.
- Every response of an accepted `Host` and `Origin` has `Cache-Control: no-store`.
- `/v1/health` also reports each GPU's `compute_cap` and `active_model` (the model in memory, or `null`). Use them for display only.
- Tokens begin with `aho_`. Treat the whole token as opaque.
- Ignore fields you do not know.

## 8. What to store for each job in your app

| Store | Why |
|---|---|
| `client_request_id`, before sending | to repeat the request safely after a crash or timeout |
| the job `id`, before polling | to find the job again after a restart |
| model id, `variant`, prompt and `params` as sent | to explain or reproduce the result |
| `results[].seed`, `results[].loras` | the rest of the recipe |
| the downloaded file and its `sha256` | the server deletes its copy after `result_ttl_s` |
| for MiniMax H3: that the clip is machine-made | your license duty ([07_models.md](07_models.md) section 5) |

## 9. Generating types

The machine-readable form is [`../../schema/aha_provider_v1.openapi.yaml`](../../schema/aha_provider_v1.openapi.yaml) (OpenAPI 3.1). A
generator will give you the routes and the job, upload, result and error types. `capabilities` and `params` are open objects there
because they differ per model: write those types from [07_models.md](07_models.md) and [`../03_api_v1.md`](../product/03_api_v1.md) §4.
