# Recipes — how do I…

Short answers to common tasks. Each names the file that does it; every file runs in the project's tests against a live gateway, and the ones marked
**✅ real** were also run against the real Qwen-Image-2.1 model (RTX 3090 Ti, 2026-11-24). Settings for all of them: `AHA_OPEN_URL`, `AHA_OPEN_TOKEN`
([01_quickstart.md](01_quickstart.md)). In the paths below `examples/` is `docs/aha_open_api/examples/`.

| I want to… | Recipe |
|---|---|
| Make a picture from a sentence | [1](#1-text-to-image) |
| Get the same picture again | [2](#2-the-same-picture-again-seed) |
| Pick a size | [3](#3-pick-a-size) |
| Change a picture I already have | [4](#4-edit-a-picture) |
| Borrow style or content from other pictures | [5](#5-reference-images) |
| Edit only one area | [6](#6-edit-only-one-area-mask) |
| Get a see-through background | [7](#7-transparent-background) |
| Use a style file (LoRA) | [8](#8-use-a-lora) |
| Make a video | [9](#9-make-a-video) |
| Stop a job | [10](#10-cancel-a-job) |
| Download a result again, or part of it | [11](#11-download-again-or-only-part) |
| Make several pictures at once | [12](#12-several-pictures-from-one-job) |

## 1. Text to image

**✅ real.** [examples/python/generate.py](examples/python/generate.py), [examples/powershell/generate.ps1](examples/powershell/generate.ps1),
[examples/node/generate.mjs](examples/node/generate.mjs), [examples/curl/generate.sh](examples/curl/generate.sh). The body they send:

```json
{"task": "image.generate", "model": "qwen-image-2.1", "prompt": "a lighthouse on a cliff at sunset, oil painting",
 "params": {"width": 1024, "height": 1024, "seed": 42, "steps": 40}, "client_request_id": "<random>"}
```

`prompt` is required and at most 8000 characters. `steps` is optional (Qwen-Image-2.1: 10–60, default 40; more is slower, not always better).

## 2. The same picture again (`seed`)

**✅ real.** The same model, prompt, size, steps and `seed` give the same picture **byte for byte**: the Python, PowerShell and Node examples all produced a PNG of
2 312 944 bytes with sha256 `c142df9e…ba45` for the prompt in recipe 1. Leave `seed` out and the server picks one and reports it in `results[].seed`; keep that number.
(Different graphics cards or driver versions are not covered by this measurement: **to be measured**.)

## 3. Pick a size

The model lists exact sizes in the manifest (`capabilities.sizes`); send `width` **and** `height` from one entry, or only `aspect_ratio`. Anything else is `400`, because AHA open
never stretches. For Qwen-Image-2.1 the **1k** tier (about 1 megapixel, 1024×1024, 1152×864, 1376×768 and their flips) takes about 36–42 s on the test machine; the **native**
tier (about 4 megapixels, 2048×2048 and others) took 269–326 s and needs more video memory than a 24 GiB card has (26.9 GiB measured; [../05_models.md](../product/05_models.md) §3.1). Other cards: **to be measured**.

## 4. Edit a picture

**✅ real.** [examples/python/edit.py](examples/python/edit.py) uploads the picture and sends:

```json
{"task": "image.edit", "model": "qwen-image-2.1", "prompt": "make the apple blue",
 "inputs": [{"role": "source", "upload_id": "up_…"}],
 "params": {"width": 1024, "height": 1024, "seed": 5}}
```

```bash
python edit.py "make the apple blue" --source apple.png --seed 5 --out apple_blue.png
```

Exactly one `source` is required. The output size is the size you ask for, not the size of the source: ask for one the model declares (the script uses the first). On the test
machine this took about 56 s and the result differed clearly from the source (mean absolute pixel difference 53 of 255).

## 5. Reference images

Add `{"role": "reference", "upload_id": …}` inputs (Qwen-Image-2.1: up to 10; the manifest's `max_reference_images`):

```bash
python edit.py "put the style of the second picture on the first" --source me.png --reference style.png
```

Tested against a live gateway with a synthetic model (🧪); a real run with several references: ⬜ not tried.

## 6. Edit only one area (mask)

Only when the model's `capabilities.mask` is `true`. **Qwen-Image-2.1 declares `mask: false`**: it has no mask input, and a job with a `mask` is refused with `400`.
`edit.py --mask` therefore stops with a sentence *before* uploading anything. What works on Qwen: paint or draw the area on the `source` picture, or describe it in the prompt.
With a model that supports masks, `--mask mask.png` sends `{"role": "mask", "upload_id": …}` (tested with the synthetic model, 🧪).

## 7. Transparent background

**✅ real.** [examples/python/recipes.py](examples/python/recipes.py) `transparent`:

```bash
python recipes.py transparent "a red apple, product photo" --out apple.png
```

It sends `"params": {…, "transparent": true}` to a model whose `capabilities.rgba` is true (Qwen-Image-2.1 is). The server adds the transparent-background description to your prompt.
Real result: a 1024×1024 PNG of color type 6 (RGBA) in which 603 773 of 1 048 576 pixels are not fully opaque (alpha from 0 to 255), about 46 s. A model without `rgba`
refuses `transparent: true` with `400 "this model cannot produce transparency"`; the script says which models it looked at.

## 8. Use a LoRA

A LoRA is a small style file that **the operator** puts on the server (`aha-open loras add my-style.safetensors --name "My style" --trigger mystyle`); a client cannot upload one.

```bash
python recipes.py loras                                   # what does this server offer? (reads the manifest)
python recipes.py lora "a portrait of a fox, mystyle" --lora my-style --scale 0.8 --seed 1 --out fox.png
```

The body carries `"params": {…, "loras": [{"id": "my-style", "scale": 0.8}]}` (up to `capabilities.loras.max`, 3 for Qwen; `scale` above 0 and at most `scale.max`, default 1.0; no repeated id).
Use the LoRA's trigger word in the prompt if it has one. The result records which LoRAs were used, with their hashes, in `results[].loras`; a job **without** `loras` is identical to a server that never had one.
Tested with the synthetic model and a synthetic LoRA file (🧪, the manifest updates without a restart). Real LoRAs on the real model were run in an earlier session ([../05_models.md](../product/05_models.md) §3.2);
this machine had none installed on 2026-11-24, so recipe 8 itself was **not run for real** (⬜).

## 9. Make a video

[examples/python/recipes.py](examples/python/recipes.py) `video`:

```bash
python recipes.py video "a lighthouse on a cliff at sunset, waves crashing" --tier preview --out lighthouse.mp4
python recipes.py video "waves on a beach" --duration 4 --out waves.mp4        # a model whose length is free
```

It needs a model whose tasks include `video.generate` and whose `state` is `ready`. It takes `duration_s`, `aspect_ratio` and `short_edge` from that model's capabilities
(it checks the duration first, saying "exactly 5 seconds" for a model with one length). A model with `capabilities.sizes` is asked for a size **by name**: `--tier preview`
or `--tier native` (for MiniMax H3; the default is the smallest, so a first try takes minutes), and a name that is not on offer is answered with the list. The script also prints
the licence duty (`required_ui_label`, `excluded_territories`) and waits up to an hour (`--timeout`). The result's `kind` is `video`, with `width`, `height`, `duration_s`, `fps`
and `has_audio`.

**✅ real, MiniMax H3 on an RTX 3090 Ti:** a 5 s clip, 124 frames at 24 fps with stereo sound at 32 kHz, takes about **4–5 minutes at
`preview` (608×352) and 23 minutes at `native` (1344×768)**, depending on the build ([07_models.md](07_models.md) section 3.4). The
whole `/v1` path through a gateway was run at `preview` on 2026-11-25 ([../05_models.md](../product/05_models.md) §3.3). The recipe script
itself is tested with a synthetic model that has the same shape (🧪, `example-video-sized`: tier choice, default, wrong name, fixed
length) and has not been run against the real model (⬜).

**References** (a model that declares `max_reference`, such as `minimax-h3-ref2va`): `--reference FILE` uploads each file and sends it as a `reference`
(image), `reference_video` or `reference_audio` input **in the order given**; the recipe then picks a model that takes that many references, and without
`--reference` it skips models that declare `reference_required` (they would answer `400`):

```bash
python recipes.py video "the person in the photo waves and says hello" --reference person.png --tier preview --out wave.mp4
```

What was run for real on the reference machine is in [../05_models.md](../product/05_models.md) §2.4; the manifest only offers what was.

## 10. Cancel a job

**✅ real.** `POST /v1/jobs/{id}/cancel`. [examples/python/recipes.py](examples/python/recipes.py) `cancel` starts a job, waits, cancels it and shows what happened:

```bash
python recipes.py cancel "a very detailed city at night" --after 8
# job job_… started; cancelling in 8 s
# final status: cancelled; results: 0
# asking for the result now gives: job_not_ready
```

A cancelled job never has a result. The server waits a few seconds for the job to stop before it answers; if it still says `running`, keep polling.

## 11. Download again, or only part

`GET /v1/jobs/{id}/result?index=0` can be repeated until `limits.result_ttl_s` has passed. `Range` works, so you can read just the start of a big file (the first 8 bytes of a PNG are its signature):

```bash
curl -H "Authorization: Bearer $AHA_OPEN_TOKEN" -H "Range: bytes=0-7" http://127.0.0.1:8890/v1/jobs/job_…/result    # answers 206
```

(For a real script prefer the examples above: typing a token on a command line puts it in your shell history. `read -rs AHA_OPEN_TOKEN` first, as in the quickstart.)

## 12. Several pictures from one job

`params.num_outputs` up to the model's `capabilities.max_outputs`, then download `?index=0`, `?index=1`… Qwen-Image-2.1 has `max_outputs: 1`: ask for several pictures by creating several jobs
(each with its own `client_request_id`). The server's queue and the per-client limit apply; a `busy` answer means wait, not hurry.
