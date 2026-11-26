# Errors and troubleshooting, in plain words

Two kinds of trouble look alike and need different answers, so this page separates them:

1. **The API answered with an error** (an HTTP status and `{"error": {"code", "message", "retry_after_s"}}`) → [section 1](#1-the-api-answered-with-an-error).
2. **A job was accepted and then failed** (`status: "failed"` with `error: {code, message}`) → [section 2](#2-a-job-failed).
3. **Nothing answered**, or something other than AHA open answered → [section 3](#3-nothing-answered-or-the-wrong-thing-answered).

Definitions: [`../03_api_v1.md`](../03_api_v1.md) §8. Branch on `code`; show `message` to a person; wait `retry_after_s` when it is a number.

## 1. The API answered with an error

| Code | HTTP | What happened | What to tell the person | What the client should do |
|---|---|---|---|---|
| `invalid_request` | 400 | A field is missing or wrong, or a parameter is outside what the model's `capabilities` allow. Nothing was queued. | "That setting is not allowed: *message*." (the message names the field, for example the sizes the model accepts) | Fix the request. Do not resend it unchanged. Build your controls from the manifest so this cannot happen |
| `unsupported_task` | 400 | This model does not do that task (for example `video.generate` on an image model). | "This model cannot do that." | Pick a model whose `tasks` include it |
| `upload_incomplete` | 400 | `complete` was called with a part missing, or the assembled file does not match the `sha256`/size you announced; or a job used an upload that was never completed. | "The file did not arrive intact. Trying again." | Upload again from `POST /v1/uploads` (after a bad checksum the server has already deleted that upload) |
| `unauthorized` | 401 | The token is missing, wrong, or was revoked. | "AHA open did not accept this app's token. Ask the person who runs it for a new one." | Stop; ask for a new token. Retrying cannot help |
| `forbidden` | 403 | Allowed to talk, not allowed to do this: a browser `Origin` the operator did not allow, a `Host` name the server does not answer to, or a model the operator turned off. | "This server refuses requests from here: *message*." | Read `message`. Origin: call from a backend instead of a web page ([03_build_a_client.md](03_build_a_client.md)). Host: use the address the operator gave you |
| `not_found` | 404 | No such job/upload **for this token** (another token's job looks the same), the result has expired, or `index` is out of range. | "That job is gone: it never existed, belongs to another app, or its result was deleted after `result_ttl_s`." | Do not retry. Save results before they expire |
| `job_not_ready` | 409 | You asked for the result of a job that has not succeeded (still running, failed or cancelled). | "The picture is not ready." | Poll the job; download only after `succeeded`. A cancelled job never has a result |
| `model_not_installed` | 409 | The model is known but its files are not on the server. | "This model is not installed on the server yet. The person who runs AHA open can install it (the web page has a *Download model* button)." | Stop; do not retry. Re-read the manifest later |
| `unsupported_hardware` | 409 | This server's graphics card cannot run the model (too little video memory, or no NVIDIA GPU). | "This computer cannot run that model." | Stop; offer another model if the manifest lists one |
| `busy` | 429 | A queue or rate limit was reached: this client's queue, the whole server's queue, too many jobs in the last minute, or too many stored uploads. `retry_after_s` says when to come back. | "The server is busy. Trying again in *N* seconds." | Wait `retry_after_s` (and the `Retry-After` header), then send the **same** request (same `client_request_id`). Never loop without waiting |
| `model_loading` | 503 | The model is still loading and the server asked you to come back. AHA open itself does not send this to a job request: it accepts the job and shows `loading_model`. Another provider may send it. | "The model is loading. Trying again shortly." | Wait `retry_after_s`, try again |
| `gpu_out_of_memory` | in a job | The graphics card ran out of video memory (see section 2). | see section 2 | see section 2 |
| `user_cancelled` | in a job | The job was cancelled (see section 2). | "Cancelled." | Nothing |
| `generation_failed` | in a job | Anything else that went wrong while making the result (see section 2). | see section 2 | see section 2 |

Rows 12–14 never come back as an HTTP error; they are the `error.code` inside a `failed` or `cancelled` job.

## 2. A job failed

`GET /v1/jobs/{id}` → `"status": "failed"`, `"error": {"code": "…", "message": "…"}`. A failed job cost GPU time, so **nothing is retried automatically**: show the message and let the person decide.

| `error.code` and message | Cause | What to do |
|---|---|---|
| `gpu_out_of_memory` | The card ran out of video memory, usually because another program (a photo or video editor, a game, a second model) holds it, or the size is too big. | Close the other programs, pick a smaller size (for Qwen-Image-2.1 the 1k tier needs about 9.5 GiB; the native tier measured 26.9 GiB, more than a 24 GiB card has), then try again |
| `generation_failed`: "the server restarted while this job was waiting or running" | AHA open was stopped or restarted while the job was in it. | Wait until it is back (`GET /v1/health`), then create the job again if still wanted |
| `generation_failed`: "the worker is not reachable" / "while generating: …" / "while loading the model: …" (a crash or out-of-memory exit code and a log file name) | The model's worker process died. The message says whether the operating system reported a crash (the request is not the cause) or running out of memory (free memory). | Try again once. If it repeats, give the operator the message and the named log (`worker-<model family>.log` in AHA open's `logs` folder; [../06_deployment.md](../06_deployment.md) §6) |
| `generation_failed`: "the job exceeded job_timeout_s" or "loading the model timed out" | It took longer than the operator's limit (`job_timeout_s`, 3600 s by default). A heavy size or a slow first load can do this; a MiniMax H3 `native` clip measured 1 309 s on an RTX 3090 Ti, so a much slower card can pass an hour. | Try a smaller size (MiniMax H3: `preview`); if it is only the first job after a start, try once more |
| `gpu_out_of_memory`: "the GPU ran out of memory: close other programs that use the GPU (games, other AI tools) or choose a smaller size, then try again" (MiniMax H3) | The video worker refuses to use more GPU memory than is free when the job starts, because on Windows a full card does not fail, it silently becomes 3× slower. How much a clip needs depends on the build (13–21 GiB, [07_models.md](07_models.md) section 3.4). | Close what uses the GPU, or ask for `preview`, then create the job again. The operator can also switch to a smaller build |
| `generation_failed`: a message about an input that is missing or cannot be read, after a long wait in the queue | An older server deleted the input while the job was still waiting ([08_limits_and_behaviour.md](08_limits_and_behaviour.md) section 2). | Upload again and create a new job; ask the operator to update AHA open |
| `generation_failed`: "the worker did not report the size of the video it produced" / "output is WxH but WxH was accepted" (video) | The model returned another size than the one accepted, or none. AHA open refuses to deliver it. | Report it to the operator with the job id |
| `model_not_installed`: "model files are missing: …; run `aha-open models install` again" (MiniMax H3) | One of the four large files (transformer, text encoder, two VAEs) is not where the worker looks. The message names the files, never a server path. | The operator runs the install again; it continues where it stopped and checks every file's sha256 |
| `generation_failed`: "output is WxH but WxH was accepted" / "the worker produced an image that is not a PNG" | The model returned something other than what was asked. AHA open refuses to deliver a wrongly sized picture instead of stretching it. | Report it to the operator with the job id; do not work around it |
| `model_not_installed` | The files disappeared after the job was accepted. | Install the model again, then create a new job |
| `user_cancelled` / `status: "cancelled"` | Someone cancelled it. | Nothing |

## 3. Nothing answered, or the wrong thing answered

Your client library usually gives one sentence for these (the example clients say `cannot reach <url>`); the cause decides what to tell the person.

| You see | Most likely | Check, in this order |
|---|---|---|
| Connection refused / `cannot reach http://127.0.0.1:8890` | AHA open is not running, or listens on another address or port. | Start it (Windows: Start menu ▸ **AHA open**; source: `aha-open serve`). `curl http://127.0.0.1:8890/v1/health` should answer. Wrong port? Set `AHA_OPEN_URL` |
| `/v1/health` answers, but `/v1/manifest` says `401` | The server is fine; your token is not accepted. | A new token: `aha-open clients create --name "…"` on the server, or the local token file ([01_quickstart.md](01_quickstart.md)) |
| `status: "starting"` | The gateway is still starting. | Wait a few seconds |
| The manifest says a model is `loading` | The model is being read into the card. | Wait about two minutes (first load on the test machine took about 120 s) |
| `403` before any JSON, with a Cloudflare page or a redirect (`internet` profile) | Cloudflare Access stopped you; AHA open never saw the request. | Check `CF-Access-Client-Id` / `CF-Access-Client-Secret` and that the Access policy is *Service Auth* for that hostname |
| `403 forbidden` with a JSON body, "this Host is not served here" | You used a name the server does not answer to (a machine name, a different IP). | Use the address the operator gave you; the operator lists extra names in `hosts` in `config.toml` |
| `403 forbidden`, "requests from this Origin are not allowed" | A web page called AHA open directly. | Call from a backend or a native app ([03_build_a_client.md](03_build_a_client.md) "Web apps") |
| Certificate error on `https://…` (`lan`) | The server's certificate is self-signed and your machine does not trust it. | Give your HTTP library the server's `.crt` (and compare its fingerprint with `aha-open tls fingerprint`). Do not turn certificate checking off |
| `200` but not JSON, or JSON that is not a `/v1` document | The address is not AHA open (another program on that port, or a proxy page). | Check the address and port |

### Two commands that say which layer is broken

On any machine that has AHA open installed:

```bash
aha-open check http://127.0.0.1:8890 --token <token>      # server up? token accepted? which models are ready? (any profile; add --cf-id/--cf-secret, --cacert as needed)
aha-open doctor                                            # on the server: GPU, configuration, state of each model
```

`check` prints one line per layer, `ok` or `FAIL`, and what to do about the failing one. Keep the token out of shared terminals and shell history.

### Reporting a problem

Give: the HTTP status, the `error.code` and `message` (they are written for people and contain no secrets), the job id, and the time. Do **not** paste your token, your prompt or your pictures
unless you mean to share them; AHA open does not write prompts or tokens to its logs.
