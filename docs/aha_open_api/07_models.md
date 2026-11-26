# Model reference — Qwen-Image-2.1 and MiniMax H3

What an app developer needs to know about each model AHA open serves: what it accepts, what it returns, what it costs, and what its
license asks of **your app**. Measurements and license quotations come from [`../05_models.md`](../05_models.md), which has the method
behind every number. The size lists, variant numbers and request examples on this page are checked against the server's model catalog
by the conformance suite.

**The manifest still decides.** This page tells you what to design for. At run time read `GET /v1/manifest`: a server lists only the
models its operator configured, and each one has a `state` of its own.

## 1. Which models exist

| Model id | Does | Tasks | Output | Commercial use | Smallest GPU |
|---|---|---|---|---|---|
| `qwen-image-2.1` | Text to image, image editing, reference images, transparency, LoRA | `image.generate`, `image.edit` | PNG (RGBA) | `not_allowed` | 9216 MiB (ConvRot variants); 10240 MiB (`int8`) |
| `minimax-h3-fl2va` | Text to video with sound | `video.generate` | MP4 | `allowed_with_conditions` | 16 GB card |
| `minimax-h3-ref2va` | One reference image to video with sound | `video.generate` | MP4 | `allowed_with_conditions` | 18 GB or more |

All of them need an NVIDIA GPU. The model files are **not** shipped with AHA open: the operator downloads them and accepts the license.
`example-image` and `example-video` are synthetic test models; a normal server does not list them.

### What `state` means for your app

| `state` | A job is… | Show the person |
|---|---|---|
| `ready` | accepted | the model, enabled |
| `loading` | accepted; it shows `loading_model` until the model is in memory | "Loading" |
| `not_installed` | refused, `409 model_not_installed` | "Not installed on the server"; `state_detail` says what is missing |
| `unsupported_hardware` | refused, `409 unsupported_hardware` | "This server's graphics card cannot run it"; `state_detail` says what is needed |
| `error` | refused, `403 forbidden` | "Disabled on this server" (MiniMax H3 on the `internet` profile until the operator accepts the hosted-service duties) |

One GPU holds **one model at a time**. When jobs for Qwen and for MiniMax H3 are mixed, the server swaps models between them, and a
swap costs a model load. `ready` means "can run", not "already in memory".

## 2. Qwen-Image-2.1 (`qwen-image-2.1`)

### 2.1 Sizes

14 exact sizes. Send `width` and `height` from one row, or only `aspect_ratio` (you then get the `1k` tier), or nothing (1024×1024).

| `aspect_ratio` | `1k` tier | `native` tier |
|---|---|---|
| `1:1` | 1024×1024 | 2048×2048 |
| `4:3` | 1152×864 | 2400×1792 |
| `3:4` | 864×1152 | 1792×2400 |
| `3:2` | 1248×832 | 2528×1696 |
| `2:3` | 832×1248 | 1696×2528 |
| `16:9` | 1376×768 | 2752×1536 |
| `9:16` | 768×1376 | 1536×2752 |

On the reference machine (RTX 3090 Ti 24 GiB) a `1k` image takes about 36–55 s depending on the variant. `native` was measured with
the `int8` variant only: 269–326 s and 26.9 GiB, more than the card has. Not measured for the ConvRot variants.

### 2.2 Parameters

| `params` field | Values | If you leave it out |
|---|---|---|
| `width`, `height` | together, from the table above | see 2.1 |
| `aspect_ratio` | one of the seven above | `1:1` |
| `steps` | 10–60 | 40 |
| `seed` | 0 or a positive integer | the server picks one and reports it in `results[].seed` |
| `num_outputs` | 1 | 1 |
| `transparent` | `true` / `false` | `false` |
| `loras` | up to 3 of `{"id", "scale"}`, 0 < scale ≤ 2 | no LoRA |

`prompt` is required for `image.generate`, at most 8000 characters. **There is no** negative prompt, guidance scale, edit strength,
sampler or output format setting: an unknown field in `params` is `400`.

Keep seeds below 2^53 if your client is JavaScript, or the number you store is not the number that was used. Seeds the server picks
are below 2^31.

### 2.3 Requests

Text to image:

```json
{"task": "image.generate", "model": "qwen-image-2.1", "prompt": "a lighthouse on a cliff at sunset, oil painting",
 "params": {"width": 1024, "height": 1024, "seed": 42, "steps": 40}, "client_request_id": "6f1c2f0e9a3b4d27"}
```

Edit one picture, with a reference:

```json
{"task": "image.edit", "model": "qwen-image-2.1", "prompt": "give the first picture the colours of the second",
 "inputs": [{"role": "source", "upload_id": "up_source"}, {"role": "reference", "upload_id": "up_reference"}],
 "params": {"width": 1152, "height": 864}}
```

Transparent background with a LoRA (the `id` must be one the manifest lists under `capabilities.loras.available`):

```json
{"task": "image.generate", "model": "qwen-image-2.1", "prompt": "a red apple, product photo",
 "params": {"aspect_ratio": "1:1", "transparent": true, "loras": [{"id": "my-style", "scale": 0.8}]}}
```

### 2.4 Inputs and output

- `image.edit` needs exactly one `source`. Up to 10 `reference` images. **No mask** (`mask: false`): a `mask` input is `400`. To point
  at an area, draw on the source picture or describe it in the prompt.
- The output has the size you asked for, not the size of the source.
- The server reads the source first and then the references, whatever order you listed them in.
- **Input transparency is dropped:** each input is converted to RGB. **EXIF rotation is not applied:** turn phone photos upright
  before you upload them. (Both are read from the worker's code, not tried.)
- The result is a PNG, `image/png`, RGBA as the model returns it, even when you did not ask for transparency.
- `transparent: true` makes the server add the transparent-background wording to your prompt; do not add it yourself.
- The same model, variant, prompt, size, steps and seed gave the same bytes on the reference machine. A different variant gives
  different bytes.

### 2.5 Variants

The operator chooses the variant. A job has **no** variant field. `manifest.models[].variant` says which one is running.

| Variant | Disk, GB | Minimum GPU, MiB | Peak GPU at 1k, GiB | Seconds per 1k image | System RAM, GiB |
|---|---|---|---|---|---|
| `int8` | 18 | 10240 | 9.5 | 39.8 | 20.0 |
| `int8-convrot` | 17 | 9216 | 8.2 | 51.3 | 23.0 |
| `mixed-p010` | 17 | 9216 | 8.2 | 54.7 | 23.0 |
| `mixed-p015` | 15 | 9216 | 8.2 | 52.3 | 22.0 |

LoRA has been run on `int8`, `int8-convrot` and `mixed-p015`. The manifest reports `loras: true` only for `int8` and `null` ("not
recorded") for the others: do not hide the LoRA control because of a `null`.

The first image after the server starts also loads the model: about 120 s on the first measurement, 21–55 s for the ConvRot variants.

### 2.6 License duties for your app

Qwen RESEARCH LICENSE AGREEMENT: *"'Non-Commercial' shall mean for research or evaluation purposes only."* and *"You shall not use the
Materials for any commercial purpose without obtaining a separate commercial license from us."*

- Show `license.name`, `commercial_use: not_allowed` and `license.notes` where the person picks the model.
- Do not ship it as part of a commercial product without a separate license from the licensor.
- §4.c: *"You shall not use "Qwen" as the primary name or identifier of any derivative works or products"*. Do not name your app or
  feature after it.

## 3. MiniMax H3, text to video (`minimax-h3-fl2va`)

### 3.1 Sizes and clip

Choose a size by `aspect_ratio` **and** `short_edge`. You never send `width` or `height` for video (`400 unknown params`).

| `tier` | `aspect_ratio` | `short_edge` | Output |
|---|---|---|---|
| `preview` | `16:9` | 352 | 608×352 |
| `native` | `16:9` | 768 | 1344×768 |

**Always send `short_edge`.** If you leave it out you get `native`, which takes about 23 minutes instead of about 5.

The clip is 5 s only (`duration_s: 5`), 124 frames at 24 fps = 5.17 s, with stereo sound at 32 kHz.

| `params` field | Values | If you leave it out |
|---|---|---|
| `aspect_ratio` | `16:9` | `16:9` |
| `short_edge` | 352 or 768 | **768** |
| `duration_s` | 5 | 5 |
| `steps` | 20 | 20 |
| `seed` | 0 or a positive integer | the server picks one |
| `num_outputs` | 1 | 1 |

```json
{"task": "video.generate", "model": "minimax-h3-fl2va",
 "prompt": "a lighthouse on a cliff at sunset, waves crashing on the rocks, seagulls calling, cinematic",
 "params": {"aspect_ratio": "16:9", "short_edge": 352, "duration_s": 5, "seed": 42}}
```

Describe the sound in the prompt too: the model makes picture and sound together. The publisher lists stable speech in 11 languages;
Vietnamese is not one of them.

### 3.2 What it does not do here

`first_frame: false`, `last_frame: false`, no references (`max_reference.total: 0`), no other length or aspect ratio, no 2K. The
publisher's description is wider than this; AHA open declares only what has been run, and refuses the rest with `400`.

### 3.3 Result

An MP4 (`video/mp4`, H.264 with AAC stereo sound) carrying the comment `AI-generated (MiniMax H3)`. `results[0]` has `kind: "video"`,
`width`, `height`, `duration_s`, `fps`, `has_audio`, `bytes`, `sha256` and `seed`.

### 3.4 Variants

| Variant | Disk, GB | Minimum GPU, MiB | Peak GPU `preview`, GiB | Peak GPU `native`, GiB |
|---|---|---|---|---|
| `int8` | 50 | 23552 | 21.4 | — |
| `mixed` | 46 | 18432 | 16.7 | 19.6 |
| `nvfp4` | 43 | 15872 | 13.1 | 16.4 |

`int8` ran out of memory at `native` on the 24 GiB card. The manifest says it with `fit_by_tier` and `too_small_up_to_gib`: **hide or
disable a tier whose `fit_by_tier` is `too_small`** on the variant that is `selected`. MiniMax H3 variants have no `seconds`,
`ram_gib` or `loras` fields; treat every variant field as optional.

| Whole job, RTX 3090 Ti | `int8` | `mixed` | `nvfp4` |
|---|---|---|---|
| `preview` | 278 s | 293 s | 242 s |
| `native` | out of memory | 1376 s | 1365 s |

Other graphics cards: to be measured. `nvfp4` at `preview` was run with PyTorch capped at 15.5 GiB, not on a real 16 GB card.

### 3.5 Progress, cancel and failure

- `progress.total_steps` is what the model reports (19 passes for `steps: 20`). Draw the bar from `progress.fraction`.
- The bar stays at 0 for the first 1–2 minutes (prompt encoding and model load), then moves once per pass, then waits again while the
  video is decoded. Do not treat a bar that is not moving as a hang.
- Every job loads the model again; nothing stays on the GPU between jobs.
- Cancel takes effect at the next pass; a worker that does not stop within 15 s is stopped by the server.
- `gpu_out_of_memory` usually means another program is using the card. `model_not_installed` inside a job names the missing file.
- Give your own wait at least 40 minutes, and add the time the job spends queued.

## 4. MiniMax H3, reference to video (`minimax-h3-ref2va`)

A **separate model id** with its own download. It declares `reference_required: true`: a job without a reference is `400` (use
`minimax-h3-fl2va` for text only).

### 4.1 What it accepts

- Exactly **one reference image**: `max_reference` is `{image: 1, video: 0, audio: 0, total: 1}`.
- Only the `preview` size.

| `tier` | `aspect_ratio` | `short_edge` | Output |
|---|---|---|---|
| `preview` | `16:9` | 352 | 608×352 |

Clip length, frame rate, sound, `steps`, `seed` and the result file are the same as in section 3.

```json
{"task": "video.generate", "model": "minimax-h3-ref2va",
 "prompt": "the building in <Picture 1> at night under a starry sky, the camera moving closer",
 "inputs": [{"role": "reference", "upload_id": "up_reference"}],
 "params": {"aspect_ratio": "16:9", "short_edge": 352, "duration_s": 5}}
```

`<Picture 1>` names the first reference in the prompt. The reference is a picture to take content from; it is **not** a first frame.

### 4.2 Variants

| Variant | Disk, GB | Minimum GPU, MiB | Peak GPU `preview`, GiB | Peak GPU `native`, GiB |
|---|---|---|---|---|
| `mixed` | 45 | 18432 | 17.1 | — |
| `nvfp4` | 43 | 18432 | 14.6 | — |

A whole job took 665 s (`mixed`) and 672 s (`nvfp4`) on the RTX 3090 Ti. `int8` ran out of memory with a reference and is not offered.
`nvfp4` ran out of memory under a 15.5 GiB cap, so 16 GB cards are not supported. The disk figure is for a first install; a server that
already has `minimax-h3-fl2va` downloads only the transformer (about 12–15 GB).

## 5. MiniMax H3 license duties for your app

The manifest carries `license.required_ui_label`, `license.excluded_territories` and `commercial_use: allowed_with_conditions`.
Quotations below are from the MiniMax H3 Community License Agreement as recorded in [`../05_models.md`](../05_models.md) §2.1. Some
sentences were cut when they were read: **read the full text before you ship**; this table is not legal advice.

| Duty | What your app does |
|---|---|
| §IV.2: *"You shall prominently display "MiniMax H3"on the user interface of commercial product or service"* | Show the exact text `MiniMax H3` (take it from `required_ui_label`) wherever the model can be picked and where its videos appear |
| Excluded territories: *"the European Union, the United Kingdom, the Republic of Korea and the United States of America."* | Do not offer the model to people there; show `excluded_territories` |
| §IV.1: a separate permission is needed *"if your commercial products and services generate more than 20 million US dollars (or equivalent in other currencies) in yearly revenue."* | Check before you sell |
| Hosted Services include *"…application programming interfaces (APIs), web access…"* | If you serve other people: your own user terms, safety measures and a way to report violations, before the operator enables the model on the `internet` profile |
| Exhibit A.12: no publishing *"without clearly and prominently disclosing that such information and/or content is machine-generated"* (cut) | Tell viewers a clip is machine-made. Keep the comment in the MP4; re-encoding may drop it |
| §VI.4 / §V.3: *"MiniMax claims no rights over the Outputs you generate."*; outputs must not be used to improve other AI models | Do not use the clips as training data |
| §VI.2: *"the Licensee may not use any name or mark owned by or associated with MiniMax"* (cut) | No product or feature name containing "MiniMax"; the required label is what the license itself asks for |

## 6. Installing (the operator's job)

Your app does not install models: `/admin/*` and the web page are not part of the API. Give the person who runs the server these
commands.

```bash
aha-open models install qwen-image-2.1 --accept-license qwen-research            # --variant int8 | int8-convrot | mixed-p010 | mixed-p015
aha-open models install minimax-h3-fl2va --accept-license minimax-h3 --confirm-territory    # --variant int8 | mixed | nvfp4
aha-open models install minimax-h3-ref2va --accept-license minimax-h3 --confirm-territory   # --variant mixed | nvfp4
aha-open models list                                                             # what is installed, what is in use, what fits
aha-open models use minimax-h3-fl2va nvfp4                                       # switch variant, then restart the server
```

The web page (`local` profile) has *Download a Model* and *Manage Models* panels that do the same.

## 7. Evidence

| Claim | Level |
|---|---|
| Qwen text to image at all 7 `1k` and all 7 `native` sizes through a gateway (`int8`) | ✅ 2026-11-23 |
| Qwen ConvRot variants through a gateway, with LoRA on `int8-convrot` and `mixed-p015` | ✅ 2026-11-25 |
| Qwen `native` on the ConvRot variants; several references in one real edit | ⬜ |
| MiniMax `fl2va` (`mixed`) through a gateway at `preview`: job, file checked, cancel | ✅ 2026-11-25 |
| MiniMax `ref2va` (`mixed`, one image) through a gateway and the web page | ✅ 2026-11-25 |
| The MiniMax build tables (sizes × variants, peaks, times) | ✅ 2026-11-26, on one RTX 3090 Ti |
| Other graphics cards; a real 16 GB card; cancel of a `native` clip in the middle of sampling | ⬜ |
