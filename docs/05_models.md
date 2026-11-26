# Models in AHA open v1

> Every figure below names its source. "To be measured" = not yet run on the project's hardware. The P0 measurements are in §3.
> Evidence levels: ✅ run for real · 🧪 code and tests only · ⬜ not checked.

## 1. Qwen-Image-2.1 (images)

| Item | Value | Source |
|---|---|---|
| Type | One model for text-to-image and image editing; 7B image generator (32-layer DiT) | official README |
| `/v1` tasks | `image.generate`, `image.edit` | |
| Reference images | ≤ 10 | official README |
| Local editing | circling, drawn annotations, or a separate mask | official README |
| Transparency | RGBA (the prompt starts with the template sentence "This is an RGBA image with transparency…") | official README |
| Recommended sizes | 1:1 2048×2048 · 4:3 2400×1792 · 3:4 1792×2400 · 3:2 2528×1696 · 2:3 1696×2528 · 16:9 2752×1536 · 9:16 1536×2752 | official README |
| Prompt expansion | Two prompt-rewriting models (Qwen3.5-VL 9B, T2I and I2I) — optional, not part of v1 | official README |
| INT8 build (community `ixim/Image21-INT8`) | 18.632 GB of weights (BF16 33.116 GB); CUDA peak 12.09 GiB (BF16 19.08 GiB), RTX 5090, 1024², 40 steps, CPU offload | INT8 README |
| INT8 environment | Python 3.13, PyTorch 2.10.0+cu128, `transformers==5.17.0`, diffusers pinned to commit `80c7ed2…` | INT8 README + requirements |
| Known limits (stated by the INT8 build) | Editing at 1024px over-sharpens; adds stray text on Chinese posters; replacing a background at 2048px fails at both precisions | INT8 README |
| On an RTX 3090 Ti 24 GB | ✅ measured, see §3: INT8 1024² ≈ 40 s / 9.5 GiB; native 2048² overflows VRAM | §3 |
| Variants AHA open offers | **Four**, all quantized (RTX 3090 Ti, 1024², 40 steps; the numbers and their sources are in §1.1 and §3.4): `int8` (community bitsandbytes build above: 18 GB, 9.5 GiB, 40 s), `int8-convrot` (17 GB, 8.2 GiB, 51 s), `mixed-p010` (17 GB, 8.2 GiB, 55 s), `mixed-p015` (15 GB, 8.2 GiB, 52 s). The three ConvRot builds share one folder (text encoder, VAE, configs) and differ only in the transformer file, so a second variant downloads only its own file. The full BF16 model (~33 GB) was evaluated and dropped (project owner, 2026-11-23): it needs twice the disk, was not faster on the measuring machine, and only pays off on GPUs ≥ 40 GiB that were never available to measure. The BF16 measurements stay in §3 as the evidence for that decision | design note R25; plan 2026-11-25 |
| Sizes AHA open declares | 14 sizes: 7 native (`tier: native`) + 7 of ~1 MP (`tier: 1k`, every side divisible by **32** — the pipeline rounds down to a multiple of 32: asking for 1200×896 returns 1184×896). Giving only `aspect_ratio` ⇒ the `1k` tier | design note R20 |
| LoRA | The operator puts a `.safetensors` file in `<data_dir>/loras/` (`aha-open loras add FILE`); the client chooses by id + `scale` (≤ 3 at a time, 0 < scale ≤ 2). ✅ run for real on `int8` with two real LoRAs (see §3.2) and on the ConvRot variants (§3.4: `int8-convrot` and `mixed-p015`, through a forward hook because peft cannot wrap the ConvRot layer); a LoRA with a layer the model lacks is **rejected**, never half-applied | design note R23 |
| `mask` | **Not declared** (`mask: false`): the README only says "separate masks" and gives no parameter; no run has shown how it behaves | design note R21 |

**License — Qwen RESEARCH LICENSE AGREEMENT** (Release Date: September 20, 2026; read in full; quoted verbatim):

- *"'Non-Commercial' shall mean for research or evaluation purposes only."*
- *"…to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Materials FOR
  NON-COMMERCIAL PURPOSES ONLY."*
- *"You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us."*
- The definition of "Qwen" includes *"…inference-enabling code…"* — which is why AHA open's adapter only calls diffusers'
  public API and copies no code from the Qwen repository.
- Redistribution must carry a Notice file: *"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026
  Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."* — AHA open does **not** distribute the weights, so
  it does not carry this file itself, but the model install command prints this notice again.

- §2.b: commercial use needs a separate license from *model-business@notice.qwencloud.com*.
- §4.c: *"You shall not use "Qwen" as the primary name or identifier of any derivative works or products; reasonable
  descriptive use (e.g., "fine-tuned from Qwen Image") is permitted."*
- §8: Chinese law, Hangzhou court.

Consequence: the AHA open operator is responsible for using Qwen within its license (the model install command requires
`--accept-license`). A client only shows `commercial_use: not_allowed` from the manifest and has no Qwen-specific logic.

**Installing from the web page (`/ui/`, local profile only).** The model is **not** downloaded when AHA open is installed; the first time the
page is opened on the machine that runs it, a *Download a Model* panel shows the licence name, `commercial_use`, the licence's own `notes` text
("Research or evaluation use only; a separate license is needed for commercial use."), a link to the full text, and a checkbox
*"I have read the licence and I accept it."* The *Download model* button stays disabled until the box is ticked; ticking it is the web
equivalent of `--accept-license qwen-research`, and the server refuses a request that does not carry that exact key. The page then shows the
phase (checking → downloading with bytes of ~18 GB → setting up the image engine → finishing), can cancel (what was downloaded is kept), and
says why it cannot start before anything is pressed: no NVIDIA GPU or less than ~10 GiB of VRAM, no `uv`, no download library, or less than
~20 GB of free disk. How it works and the API: [04_architecture.md](04_architecture.md) section 6.1.

**Choosing a variant, and the *Models* panel.** The first download picks the variant for the machine (the best measured quality that fits: `int8-convrot`, then `mixed-p010`,
`mixed-p015`, `int8`) and still needs one tick and one click; a drop-down lets the person pick another. Once a model has a variant on disk the page shows a **Models** panel: one row per
variant with its size, the measured GPU memory and time, whether it fits this GPU (`fits` / `tight` / `too_small`; a number that was not measured never refuses), the publisher's quality
note, and **Use this**, **Download**, **Delete** (asks first, says what it frees; the text encoder and VAE other variants share stay until the last variant goes, and are offered separately) and
**Cancel download**. A second variant is only added: the running one is never switched behind the operator's back. Switching or deleting is refused with `409 model_busy` while a job of the
model waits or runs. The command line has the same: `models install --variant`, `models list`, `models use`, `models remove` ([04_architecture.md](04_architecture.md)).

Evidence: 🧪 the API (conformance tests with a fake install script) and ✅ the page driven in a real browser (Chromium via Playwright, a real
gateway, a fake install script: licence and consent, progress phases, cancel button, network-failure sentence, low-disk sentence, no loop
when an install ends with the model not ready). **Correction from the first real download:** the progress bar first measured the weights
folder and stayed at 0% for minutes while data was arriving (`hf-xet` assembles files only at the end), a defect the fake script could not
show because it wrote straight into the folder; the bar now uses the downloader's own byte count. ✅ **A complete real download through the
page** (2026-11-24, RTX 3090 Ti, the Windows installer into a clean folder, running the installed code, not the checkout): the first
attempt was cut off on purpose at 9.35 GiB, a second press continued it, and the model (INT8, 18 GB) plus its PyTorch environment were ready
about 17 minutes after the second press; the first job loaded the model in ~120 s, the next took 42 s. `a lighthouse on a cliff at sunset, oil painting`,
seed 42, 1:1: 1024×1024, 2 312 944 bytes, **sha256 `c142df9e…ba45` — identical to the P0 reference**, checked on the bytes downloaded from
`/v1/jobs/{id}/result` and by looking at the picture. The connection gave 5–13 MB/s.

Manifest: `"license": {"name": "Qwen RESEARCH LICENSE", "commercial_use": "not_allowed"}`.

### 1.1 The ConvRot variants (`int8-convrot`, `mixed-p010`, `mixed-p015`)

Sources, each file pinned by revision, size and sha256 in `aha_open/install.py` and checked after the download:

| Part | Repository (revision) | Files |
|---|---|---|
| Configs, tokenizer, VAE | `Qwen/Qwen-Image-2.1` (`d26bb612…`) | small files + `vae/diffusion_pytorch_model.safetensors` (1.35 GB) |
| Text encoder (Qwen3-VL 8B, INT8 ConvRot) | `Comfy-Org/Qwen-Image-2.1` (`aacf31a6…`) | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (9.35 GB) |
| Transformer | `JoaoZaokk/ConvRot-Mixed-INT8-for-Qwen-Image-2.1` (`ea9b6618…`) | `qwen_image_2.1_bf16_int8_convrot` 7.26 GB · `…_mixed_p010_int8` 6.55 GB · `…_mixed_p015_int8` 4.61 GB (+ `.quant.json`, `NOTICE.txt`, `LICENSE_QWEN_RESEARCH.txt`) |

**Licence.** The publisher states, verbatim: *"Non-commercial use only (Qwen Research License, see below). These files are modified, quantized versions of the original
transformer weights; they are not the original model."* and its `NOTICE.txt` reads *"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026
Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved. The files in this repository are modified (quantized) versions of the Qwen-Image 2.1 diffusion
transformer … They are not the original weights. Changes made by JoaoZaokk, 2026-10."* The install command prints that notice after the download; the model's licence key
and `commercial_use: not_allowed` are unchanged. AHA open still downloads nothing by itself and ships no weights.

**What the file is.** Every one of the 192 linear layers of the transformer is stored as INT8 ConvRot (`int8_tensorwise`: one INT8 weight and one float32 scale per output row)
or as ConvRot W4A4 (`convrot_w4a4`: two signed INT4 per byte, low nibble first); the format of each layer is in the safetensors header
(`__metadata__._quantization_metadata`). The weight was rotated by a Hadamard transform over groups of 256 input features, so the stored matrix is `W·H` with `H`
symmetric and `H·H = I`. The mixes keep the layers with the largest 4-bit error in INT8: `p015` keeps 120 layers in 4 bits and 72 in INT8, `p010` 68 and 124.

**Quality, in the publisher's words (one rater, three blind rounds of 24 scenes; not repeated here):** full INT8 *"was never picked as the worst version and was repeatedly
called "nearly identical" to BF16"*; `p010` *"is the cheapest mix that never looked bad in the review"*; `p012` and `p015` *"fail the same hard scenes (motion blur, crowded
interiors) in 6 of 24 scenes"*. The same publisher warns that PSNR against BF16 does not rank the versions. These notes are shown on the page next to each variant; AHA open has
**not** repeated the blind review.

**How AHA open runs it** (`workers/qwen_image/qwen_models.py`, `convrot.py`, `convrot_lora.py`; no ComfyUI code was read or copied). The loader builds the transformer on the
`meta` device from the official config, replaces the 192 layers, splits the fused `img_mlp.gate_up` into diffusers' `gate_layer` (first half) and `proj` (second half), loads
the other tensors as they are, and refuses a file with a tensor left over, missing or still on `meta`. The product is computed as `(x·H)·Sᵀ`: the **activation** is rotated (cheap) instead of
un-rotating the weight (which cost `N·K·256` per call), quantized per row to INT8, and multiplied through `torch._int_mm` with the weight passed as a column-major **view**
(`weight.t()`): that layout is what cuBLASLt wants, and a contiguous copy measured 3.4× slower (§3.4). A GPU without INT8 tensor cores (< sm75) decodes the weight to bf16
instead. Model-level CPU offload is always used. Needs `comfy-kitchen==0.2.42` (Apache-2.0) in the worker environment, which is why the lock changed; `bitsandbytes` stays for `int8`.

## 2. MiniMax H3 (video with audio)

> Status: **runs for real, installed from the command line** (2026-11-25). The community build was first run in a reference runtime
> (§2.2); AHA open's own worker (`workers/minimax_h3/`) now loads it without ComfyUI (§2.3) and two models are served through `/v1`:
> `minimax-h3-fl2va` (text to video) and `minimax-h3-ref2va` (one reference image, §2.4). Both can be installed from the command line or
> from the web page's *Download a Model* panel, which asks for the licence **and** for a separate confirmation that the person is not in a
> territory the licence excludes ([04_architecture.md](04_architecture.md) section 6.1).

| Item | Value | Source |
|---|---|---|
| Type | Multimodal video + audio generation model | official README |
| Duration | 4–15 seconds | README |
| Frame rate | 24 fps | README |
| Resolution | Default short edge 768 px; 2K through an "H3-Regenerate-2K" step (re-runs the same model) | README |
| Aspect ratios | Many, including 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 | README |
| Audio | Stereo 32 kHz; stable speech in 11 languages (Vietnamese is not in the list) | README |
| FL2VA checkpoint | Text→video; 1 image (first or last frame); 2 images (first + last) | README |
| Ref2VA checkpoint | References: ≤ 9 images, ≤ 3 videos, ≤ 3 audio clips (each clip 2–15 s, total ≤ 15 s), at most 12 files | README |
| Precision | BF16; "33B" Omni-Transformer | README, requirements |
| Official serving (example) | `sglang serve --num-gpus 4` | README |
| Environment | diffusers pinned to commit `f53d552…`, `transformers>=4.57`, torch ≥ 2.4 | requirements |
| "H3-Context-IR" prompt expansion | Calls **MiniMax's cloud API** (needs a token), does not run locally. AHA open does **not** enable it by default — sending prompts to a third party goes against the data principle | script `full-2k-t2va-h3-context-ir.sh` |
| Encoder | Qwen3-VL-32B (Apache-2.0) | LICENSE notes |
| On an RTX 3090 Ti 24 GB | ✅ a 5 s clip runs in a reference runtime: 3–3.5 min at 608×352, **about 19–20 min at 1344×768**; see §2.2 | measured, §2.2 |

`/v1` mapping: `minimax-h3-fl2va` (`video.generate`, `first_frame`, `last_frame`), `minimax-h3-ref2va`
(`video.generate`, `max_reference`). Result: an mp4 with audio (`has_audio: true`).

### 2.0 The one build AHA open will use (decided 2026-11-23, project owner)

> Superseded on 2026-11-26: three builds per model are offered now (section 2.5); this section keeps why the `mixed` build came first.

The full model (~133 GB: transformer 66.28 GB + encoder 66.71 GB, read from the index file) is **dropped**. The only MiniMax H3
build AHA open will support is the community quantized collection **`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`**.
What was read from its public page and file list (2026-11-23; ✅ read, nothing downloaded or run):

| Item | Value |
|---|---|
| Revision looked at | `908eccad7e68751190d04c171956f163bfeed741` (last modified 2026-08-15) |
| License field | `license: other`, `license_name: minimax-h3-community-license-agreement` (the base model's license, §2.1) |
| Base model | `MiniMaxAI/MiniMax-H3`, `base_model_relation: quantized` |
| FL2VA files | `…FL2VA_pruned_mixed_int4_int8_convrot.safetensors` 14.81 GiB · `…_pruned_int8_convrot` 19.53 GiB · `…_pruned_nvfp4` 11.67 GiB (NVFP4 needs a Blackwell GPU) |
| Ref2VA files | `…Ref2VA_pruned_mixed_int4_int8_convrot` 14.06 GiB · `…_pruned_int8_convrot` 19.53 GiB · `…_pruned_nvfp4` 11.67 GiB · `…_nvfp4_mixed` 22.76 GiB |
| VAEs (required by every build) | video 4.85 GiB + audio 0.56 GiB |
| Page claims (third party, **not verified**) | the mixed INT4/INT8 file "requires 15+ VRAM (~15.5 GB)"; INT8 "24GB+ VRAM"; text encoders in `/text_encoders` (15.0–27.1 GB) |

#### Safetensors headers read (✅ 2026-11-23, HTTP Range only, nothing downloaded)

Read: 8-byte length + the JSON header (95–103 KB) of three FL2VA files, by `Range` requests against `resolve/main`.

| File | Tensors | Quantization (from the files themselves) | Dtypes |
|---|---|---|---|
| `…FL2VA_pruned_int8_convrot` | 932 | per-layer `*.comfy_quant` = `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` (**corrected 2026-11-24**: every sampled quantized layer of this file, read from the blobs themselves; the first reading wrongly gave it the `convrot_w4a4` format) — INT8 weights rotated with a Hadamard transform, one scale per output row | I8 ×200, U8 ×200, F32 ×210, BF16 ×220, F16 ×102 |
| `…FL2VA_pruned_mixed_int4_int8_convrot` | 932 | same format; metadata `convrot_w4a4_mixed`: `"w4a4_int8mm_layers": 117, "int8_layers": 83, "linear_dtype": "int4"`, `"paper": "arXiv:2512.03673"` | same counts; `blocks.0.attn.out_proj.weight` I8 `[5376, 3584]` |
| `…FL2VA_pruned_nvfp4` | same layout | `*.comfy_quant` = `{"format": "nvfp4"}`; `weight` U8, `weight_scale` F8_E4M3, `weight_scale_2` F32 | U8, F8_E4M3, BF16, F16, F32 |

**Where the files come from (✅ 2026-11-24, all three checked against the publisher's sha256).** The Abiray repository has **no text encoder**
(its last commit is titled "Delete text_encoders"), and the encoder is required (*"The H3-Encoder uses the full pretrained weights of
Qwen3-VL-32B"*). The publisher's own example workflow (`ref2.json`) takes the VAEs and the encoder from `Comfy-Org/MiniMax-H3`. So AHA open uses:

| Part | Repository @ revision | File | sha256 (first 16) |
|---|---|---|---|
| Transformer, mixed | `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` @ `908eccad…` | `MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors` (14.81 GiB) | `541c7797d815a165` |
| Transformer, INT8 | same | `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors` (19.53 GiB) | `f07a54277ca85b59` |
| VAE video / audio | same | `vae/minimax_h3_video_vae_fp16.safetensors` (4.85 GiB) / `vae/minimax_h3_audio_vae_fp32.safetensors` (0.56 GiB) | `88539d729fd7e9f9` / `83043bf3afccf4ca` |
| Text encoder | `Comfy-Org/MiniMax-H3` @ `4b53a351…` (**only because Abiray has none**) | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (25.28 GiB) | `bc2ced0fbea64757` |

Not used: `yuivs/Minimax-H3-nvfp4-INT4-INT8-Convrot` (a byte-identical copy of the Abiray files, same hashes); the transformer and VAE files
of Comfy-Org (they differ byte for byte from Abiray's, e.g. video VAE `7c1f1314…` vs `88539d72…`). The encoder file is INT8 per output row
(`weight` I8 + `weight_scale [rows, 1]`), HuggingFace key names, only the first 50 layers, and its metadata says the output is *"unnormalized_hidden_after_layer_50"*.
Downloads must be per file with a pinned revision and a hash check; a `git clone` of the Abiray repository fetches all seven checkpoints
(~115 GiB) and an interrupted one left 76 GB of partial files on the measuring machine.

Facts: tensor names are ComfyUI-style (`blocks.N.attn.qkv_proj`, `blocks.N.mlp.fc1`, `blocks.N.adaln_proj.linear`, `token_refiner.*`,
`adaln_t_table`), every quantized linear has a `*.comfy_quant` JSON blob, and the mixed file's metadata `config` gives `hidden_size 5376`,
`num_layers 50`, `num_attention_heads 56`, `latents_dim 24`, `audio_latents_dim 32`, `patch_size [1,2,2]`. Dtypes are all standard.
Inference (not observed): the mixed file's `out_proj` width 3584 = 7168/2 suggests two INT4 values packed per byte.

The Ref2VA files have the same format (`…Ref2VA_pruned_mixed_int4_int8_convrot`: 932 tensors, same `comfy_quant` blob; `…Ref2VA_nvfp4_mixed`: 951 tensors,
F8_E4M3 scales, an extra `time_embedder`). Not read: the INT4-only files and the VAEs. The repository has **no `text_encoders/` folder** (top level
of the tree: the seven checkpoints, `README.md`, `ref2.json`, `Amy`, `vae/`) although its README lists three text-encoder files.

**Consequence (✅ from the headers; the fix is §2.3):** these are not diffusers checkpoints. Their key names and `comfy_quant` layout are
not what `from_single_file` / the Qwen worker's diffusers code expects, and running ConvRot W4A4 needs a loader plus INT8/INT4 kernels that
are not part of diffusers. "Load them with the diffusers worker" is therefore **refuted as stated**. This paused MiniMax H3 support until a runtime for this
format was chosen and built (§2.3: a loader of AHA open's own for the diffusers pipeline).

**What exists for this format** (✅ read from public sources on 2026-11-23, nothing downloaded):

- It is implemented by ComfyUI (`Comfy-Org/ComfyUI`, **GPL-3.0** per the GitHub API; the H3 model is in `comfy/ldm/minimax/model.py`,
  `convrot_w4a4` in `comfy/quant_ops.py`) and by the library `comfy-kitchen` (**Apache-2.0**, on PyPI: quantized tensor types and kernels, no
  model code). The official MiniMax repository (`MiniMax-AI/MiniMax-H3`) ships the VAE code; its license field is "other".
- The official ComfyUI-format distribution is **`Comfy-Org/MiniMax-H3`** (license field `minimax-h3-community-license-agreement`, not gated). It has
  everything the pipeline needs: `diffusion_models/` (pruned INT8-ConvRot 19.53 GiB, pruned W6A8 14.89, pruned FP8 19.52, full INT8 31.70),
  `text_encoders/` (INT8-ConvRot 25.28 GiB, NVFP4-AWQ 14.61 for Blackwell, BF16 47.97) and `vae/` (video 4.85 GiB, audio 0.56 GiB).
- Not measured, and the reason it was not tried here: a 24 GiB GPU / 31.7 GiB RAM machine would need the INT8 transformer (19.5 GiB) plus the INT8
  text encoder (25.3 GiB) plus the VAEs (5.4 GiB), about 50 GiB of downloads, and the two big parts cannot sit in RAM together, so the runtime
  must load them one after the other (ComfyUI does).

**Open questions as they stood on 2026-11-24 — answered in §2.3 and §2.4 (✅ 2026-11-25); the ones still open are marked** (⬜):
- The files are **single-file ComfyUI checkpoints of a "pruned" model**, not a diffusers pipeline folder. The header read above shows the
  diffusers worker cannot load them as they are; which runtime can (and how much quality the pruning costs) has not been tried.
- ✅ Which file the worker uses: the mixed INT4/INT8 file (the only variant offered; NVFP4 excludes everything but Blackwell).
- ~~The text encoders named by the page do not appear in the file list~~ — resolved 2026-11-24: use the one from `Comfy-Org/MiniMax-H3` (table above).
- Time per clip and the peak the runtime chose: measured in §2.2. The peak AHA open's own loader needs is measured in §2.3 (17.24 GiB at
  608×352, 19.61 GiB at 1344×768), which is where `min_vram_mib` (18 432) comes from. ⬜ Quality against the unquantized model is not compared.
- ✅ Which runtime the worker uses without a separate ComfyUI install: a diffusers-based worker with a loader of its own over `comfy-kitchen`
  (direction approved 2026-11-24, built and checked against the §2.2 clips in §2.3).
- The license text is the base model's; a community re-upload does not change §2.1. The operator's duties (territory, UI label, hosted
  service) are the same.

### 2.2 Reference runs on the RTX 3090 Ti (✅ 2026-11-24)

What was run: ComfyUI 0.39.0 (commit `b26625f`, started as a separate loopback-only process with custom nodes disabled and its own output
folders) loaded the files of §2.0 and generated text→video clips: prompt *"a lighthouse on a cliff at sunset, waves crashing on the rocks,
seagulls calling, cinematic"*, seed 42, 124 frames (about 5.2 s at 24 fps), 20 steps, `res_multistep`, `simple` schedule, one guider without
a negative prompt (the chain of the publisher's example, minus its SageAttention and RTX-upscaler custom nodes). The driver is
`tools/h3_oracle/run_oracle.py` (development machine only; it reads the process table and `nvidia-smi`, nothing else). Machine: RTX 3090 Ti 24 GiB, 31.7 GiB RAM.

| Transformer | Size | Job time | Peak VRAM (whole GPU) | Peak RAM of the process | Lowest free RAM of the machine |
|---|---|---|---|---|---|
| mixed INT4/INT8 | 608×352 | 199 s | 23 811 MiB | 8.95 GiB | 11.13 GiB |
| INT8 | 608×352 | 179 s | 23 774 MiB | 9.06 GiB | 7.49 GiB |
| mixed INT4/INT8 | 1344×768 | **1 180 s** (19 min 40 s) | 23 852 MiB | 13.65 GiB | 6.69 GiB |
| INT8 | 1344×768 | **1 130 s** (18 min 49 s) | 23 818 MiB | 16.83 GiB | 3.15 GiB |

How to read it: the first run of this driver reported 0 MiB and 0.01 GiB because its sampler thread had died on a psutil API difference
without a trace; the driver now counts rounds and errors and writes `sampler_trustworthy` (true in every row above, ≥ 196 rounds, 0 errors).
**Peak VRAM is the whole GPU's used memory with ComfyUI's dynamic offloading: it fills the card, so ~23.8 GiB is what the runtime chose
to use, not the minimum the model needs.** The minimum has not been measured. The INT8 file is not slower than the smaller mixed file here
(its kernels are plain INT8 matrix products); the mixed file saves disk (14.81 vs 19.53 GiB) and RAM at the larger size, not time.

Content, checked on the decoded frames and audio rather than on "job succeeded": every clip is H.264 at exactly the requested size
(608×352 and 1344×768), 24 fps, 124 frames, 5.167 s, with an AAC stereo 32 kHz track whose RMS is 0.010–0.018 (not silence); the first and last
frame differ (mean absolute difference 5–19 of 255) and the middle frames show breaking waves and moving birds. Looked at by eye: a lighthouse
on a cliff at sunset with the sea, as prompted. **No connection to a non-loopback address** was seen from the ComfyUI process tree during
any of the four runs (socket table sampled every second). Not checked: the first/last-frame inputs, the audio's content, repeatability of a
seed, and anything about quality compared with the unquantized model.

Consequence for the product: at the model's default short edge (768) one clip takes about 19 minutes on this GPU, so the job + poll design
is not optional and the manifest must say so; 608×352 takes about 3 minutes. The AHA open worker was written afterwards (plan
2026-11-24_minimax_h3_plan.md, phases M0′–M2) and is compared against these clips in §2.3.

### 2.3 AHA open's own loader, no ComfyUI (✅ 2026-11-25, same RTX 3090 Ti, mixed build)

`workers/minimax_h3/` builds the diffusers MiniMax H3 pipeline from the same files (design and checks: [04_architecture.md](04_architecture.md)
§6.2). Same prompt and seed 42, 124 frames, 20 steps (19 transformer calls: the scheduler's grid has 19 steps), three stages
(encode, sample, decode) so only one large part is on the GPU at a time.

| Size | Encode | Load transformer | Sample | Decode | **Total** | Peak CUDA allocated |
|---|---|---|---|---|---|---|
| 608×352 | 53.5 s | 28.9 s | 174.2 s (9.2 s/step) | 20.8 s | **283 s** | 17.24 GiB |
| 1344×768 | 55.0 s | 30.5 s | 1 160.4 s (61 s/step) | 50.0 s | **1 309 s** (21 min 49 s) | 19.61 GiB |

For comparison, the ComfyUI reference of §2.2 took 199 s and 1 180 s for the same two sizes: AHA open's loader is 42 % slower at
608×352 and 11 % slower at 1344×768, where the sampling itself (61 s/step against ComfyUI's 59 s/step) is on par and the rest is
the encoder read from disk each time (55 s) and the transformer loaded each time (30 s). Content is checked on decoded pixels and
audio, not on "done": 124 frames of exactly the requested size, a stereo 32 kHz sound track (RMS 0.008–0.014), and the frames show
the prompt at both sizes (looked at by eye). The mp4 writer (OpenH264) reproduces the generated frames at 39 dB PSNR.
Not compared: the clip itself against ComfyUI's (different sampler, so no bit-for-bit comparison exists), first/last-frame input.

**A defect found and fixed on the way (root cause, not a workaround).** The first 1344×768 run took 2 385 s: 118 s per step. A real
block timed in isolation took 1.19 s with room on the card and **3.95 s when the card was full** (same code, 0 allocator
retries): on Windows the NVIDIA driver can move GPU memory to system RAM instead of failing, silently. The fix has three parts:
the weight is decoded to bf16 just before each product (the INT8 product of `comfy-kitchen` runs its rotation and quantization as
separate kernels on GPUs without cuBLASLt and measured slower than a bf16 product); the feed-forward runs in slices of 8 192 tokens;
and PyTorch is capped at the GPU memory that is free when a job starts, so going over is an error. Decoding the weight also leaves
the activation unquantized: the linear product is within 0.23 % of the dense product with the decoded weight (the INT8 activation
path gave 1 %), and the decoded weight equals `comfy-kitchen`'s own decoder exactly.

### 2.4 Reference images: `minimax-h3-ref2va` (✅ 2026-11-25, same RTX 3090 Ti, mixed build)

The same community build also has an "omni-reference" checkpoint (`MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors`,
15 093 774 276 bytes, sha256 `434c372cbb68ff055ffb5aa2c35de42211d19ebacc2ae8e9ae9202b00ee48afb`, both **checked after the download**). AHA open
runs it with the same loader through the diffusers `ref2va` workflow (`transformer_ref`; its `config.json` is identical to the text model's).
The encoder, both VAEs and the small configs are the same files as the text model's (all 20 small files of `FL2VA/` and `Ref2VA/` have the same
hash on the Hub), so the reference model is installed **into the text model's folder** and a machine that has the text model downloads only
the 14 GiB transformer (`aha-open models install minimax-h3-ref2va --accept-license minimax-h3 --confirm-territory`).

| Measured (one 256×160 reference image, preview 608×352, 124 frames, 20 steps) | Result |
|---|---|
| Whole job through a real gateway and the harness (`real_ref2va_one_image`) | **PASS 20/20, 665 s**: the downloaded file is 608×352, 124 frames, 24 fps, stereo 32 kHz, label `AI-generated (MiniMax H3)`; size, fps, duration, audio, sha256 and bytes equal what the job reported |
| Same job through the web page (Playwright) | a 608×352, 5.175 s clip in the `<video>` element, result line `… 24 fps, with sound, seed … - MiniMax H3` |
| Stages of a job (cold) | encode 82.7 s (prompt + vision tower + reference VAE) · load transformer 39.4 s · sampling **499 s** (19 passes, ~26 s each: the reference adds rows) · decode 26.3 s |
| GPU memory | peak **17.07 GiB** allocated (text model at the same size: 17.2) |
| Worker RAM | peak RSS 1.7 GiB; the machine's free RAM stayed above 18 GiB (stage-line samples) |
| Does the reference matter? (same prompt, seed, 8 steps, 384×224; a red image vs a blue image) | mean R−B of the clip **+37.3** vs **−71.6**: a difference of 108.9 on a 0–255 scale (one seed) |
| Text model after the change (`real_video_h3_preview`) | still PASS 19/19, 290 s (283 s before): no slowdown |

Two defects were found on the way and fixed at the cause (plan section 3.10):

1. The vision tower's rotary table is a computed buffer that is not in the file; left on the `meta` device it made the job fail before it
   started. It is rebuilt like the language model's.
2. Qwen3-VL's grouped-query attention asks PyTorch for `enable_gqa=True`. Only the flash and math kernels accept that flag and the Windows
   builds have no flash kernel, so one reference image (6 544 tokens after the image is brought to a short edge of 2 048) needed a
   64 × 6 544² float32 score matrix: 10.21 GiB, out of memory in the *encoder*. The worker now registers an "sdpa" attention that repeats the
   key/value heads first (same numbers, tested to 1e-5), which keeps the memory-efficient kernel (encoder peak 9.7 GiB with a reference).

**Declared (only what was run):** one reference **image**, size `preview` only. **Not run, so refused with `400`:** several images, reference
video, reference audio (the code path exists in the worker; nothing was measured), native 1344×768 with a reference, first/last frame.
The publisher's README lists ≤ 9 images, ≤ 3 videos, ≤ 3 audio clips; widening `max_reference` is a measurement, not a switch.

### 2.5 The builds an operator can choose (✅ 2026-11-26, RTX 3090 Ti 24 GiB, torch 2.10.0+cu128)

The Hub repository `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` (revision `908ecca…`) holds three pruned transformer builds per
model. Its README also lists a pure INT4 build and INT4 / NVFP4 text encoders: **no such files are in the repository**, so they are
not offered. All builds use the same INT8 text encoder (Comfy-Org) and the same two VAEs.

| Build | FL2VA file / sha256 | GiB | Format |
|---|---|---|---|
| `int8` | `…_pruned_int8_convrot` `f07a5427…` | 19.53 | INT8 + ConvRot, 200 quantized layers |
| `mixed` | `…_pruned_mixed_int4_int8_convrot` `541c7797…` | 14.81 | INT4/INT8 + ConvRot |
| `nvfp4` | `…_pruned_nvfp4` `6ab7f0c4…` | 11.67 | FP4 (E2M1), high nibble first; FP8 scale per 16 inputs in cuBLAS' tiled layout; one float32 scale per tensor; no rotation |

Reference-model files: `int8` `ed7e9aa5…` (19.53 GiB), `mixed` `434c372c…` (14.06), `nvfp4` `3e1be702…` (11.67). The three FL2VA
files, the text encoder, both VAEs and Ref2VA `int8` were checked against these hashes after the download.

**How NVFP4 was verified.** The layout above was found by decoding one layer four ways and comparing with the same layer of the
INT8 file: the right one differs by 9.4–9.5 % (4-bit noise), the others by 44–146 %. The loader's decoder equals comfy-kitchen's
reference exactly on a synthetic tensor (test). The publisher made the build for Blackwell cards; AHA open decodes it to bf16 before
each product, so it runs on any NVIDIA card (here an Ampere one, compute capability 8.6).

Measured (text to video, 124 frames = 5.17 s, 20 steps, seed 42, same prompt; every clip: exact size, 24 fps, stereo 32 kHz,
`AI-generated (MiniMax H3)` label, frames looked at by eye):

| Build | Size | Peak GiB (PyTorch allocated) | Whole job | Result |
|---|---|---|---|---|
| `int8` | 608×352 | 21.40 | 278 s | ✅ |
| `int8` | 1344×768 | — | — | ❌ out of memory before the first step (22.07 GiB allocated, cap 22.3) |
| `mixed` | 608×352 | 16.71 | 293 s | ✅ |
| `mixed` | 1344×768 | 19.60 | 1 376 s | ✅ |
| `nvfp4` | 608×352 | 13.08 | 242 s | ✅ |
| `nvfp4` | 608×352, memory capped at 15.5 GiB | 13.08 | 317 s | ✅ same output as uncapped (a 16 GB card imitated; the whole process took ~14.1 GiB) |
| `nvfp4` | 1344×768 | 16.43 | 1 365 s | ✅ (needs a 24 GB card) |

**Which card runs what** (from the peaks; `[gpu] memory_limit_gib` narrows it): 24 GB: `mixed` and `nvfp4` at both sizes, `int8` at
608×352 only. 16 GB: `nvfp4` at 608×352 (imitated with a cap, **not tried on a real 16 GB card**; the publisher's README says
*"Do not download these for 30/40 series cards!"* about NVFP4, which is about NVIDIA's FP4 kernels, not about this decoder).
Below 16 GB MiniMax H3 is not offered. System memory: the worker's peak RSS was 20–22 GiB (memory-mapped file pages are counted in
it); a 16 GB-RAM machine has not been tried.

Two changes were needed for this (root causes, plan 2026-11-25_minimax_h3_variants_plan.md §3.3): the feed-forward now runs in slices
of 2 048 tokens (with the INT8 build's 19.5 GiB of weights resident, 8 192 ran out of memory on the second step), and the NVFP4 block
scales stay in FP8 (as float32 they added 3.4 GiB to the peak). The `mixed` peaks above supersede those of §2.3; the reference model's
`mixed` peak (§2.4, 17.07 GiB) was measured with the 8 192-token slices and is therefore an upper bound.

**Reference model (`minimax-h3-ref2va`), one reference image, 608×352, 124 frames, 20 steps** (same card):

| Build | Peak GiB | Whole job | Result |
|---|---|---|---|
| `mixed` | 17.07 | 665 s | ✅ (§2.4; measured with the 8 192-token slices: an upper bound) |
| `int8` | — | — | ❌ out of memory on the 24 GiB card (21.69 GiB allocated): **not offered** |
| `nvfp4` | 14.55 | 672 s | ✅ |
| `nvfp4`, memory capped at 15.5 GiB | — | — | ❌ out of memory at the first denoising step: the reference model is **not offered on 16 GB cards** |

The `nvfp4` reference build was checked on the pixels: with a frame of a lighthouse (white tower, red-roofed houses on a headland) as
the reference and a prompt asking for that building at night, the clip shows the same tower and houses under a starry sky, the camera
moving closer. (A plain-colour reference with a neutral prompt shows the colour in the first frame only: mean colour is not a good
measure of whether a reference is used.) The `nvfp4` file was downloaded through the web page's *Download* button (12.3 GB, the
shared files already on disk were not fetched again) and matched its published sha256.

### 2.1 License — "MiniMax H3 COMMUNITY LICENSE AGREEMENT"

License date: August 2, 2026. Licensor: Nanonoble Pte. Ltd. Source: `huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE`
(read on 2026-11-22 through a web tool; long sentences were cut at 125 characters, marked "(cut)" — **re-check the full text before the MiniMax worker ships**).

| Clause | Quotation | Consequence for AHA open |
|---|---|---|
| Territory (§II, Def. 3, Def. 5) | *"Solely within the Applicable Territory…"* (cut); Excluded Territories: *"the European Union, the United Kingdom, the Republic of Korea and the United States of America."* | Not usable in the EU/UK/Korea/US. The install command requires `--confirm-territory` |
| Commercial use (§IV.1) | A separate permission is needed *"if your commercial products and services generate more than 20 million US dollars (or equivalent in other currencies) in yearly revenue."* | Below the threshold: commercial use is allowed. `commercial_use: allowed_with_conditions` |
| Showing the name (§IV.2) | *"You shall prominently display "MiniMax H3"on the user interface of commercial product or service"* | `required_ui_label: "MiniMax H3"`; the client shows it verbatim |
| Service to others (Def. 8, §V.2, §V.5) | Hosted Services include *"…application programming interfaces (APIs), web access…"*; must *"bind each recipient or user to enforceable terms…"* (cut); safety measures + *"mechanism for reporting suspected violations"* (cut) | Off by default on the `internet` profile; enabling it needs the operator to accept the obligations |
| Output (§VI.4, §V.3) | *"MiniMax claims no rights over the Outputs you generate."*; outputs must not be used to improve other AI models | Stated in user documentation |
| Disclosing machine-made content (Exhibit A.12, §III.3.b) | Forbids publishing *"without clearly and prominently disclosing that such information and/or content is machine-generated"* (cut); encourages adding an AI marker to files | Write an AI label into mp4 metadata (never the prompt) |
| Names/marks (§VI.2) | *"the Licensee may not use any name or mark owned by or associated with MiniMax"* (cut) | No product/command name carries "MiniMax"; `minimax-h3-*` is only a descriptive id |
| Distribution (§III.1–§III.4) | Include the Agreement, mark modified files, the NOTICE file *"MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved."* | AHA open does not distribute the weights; the install command prints the NOTICE |
| Prohibited use (Exhibit A) | 20 items, including breaking the law, harm to children, election disinformation, impersonation, high-risk automated decisions, military purposes | The install command prints the list; the operator is responsible |
| Law (§IX) | Hong Kong law | |

Note: MiniMax H3's encoder is **Qwen3-VL-32B** (Apache-2.0), not the Qwen RESEARCH LICENSE.

Manifest: `"license": {"name": "MiniMax H3 Community License", "commercial_use": "allowed_with_conditions",
"excluded_territories": ["EU","GB","KR","US"], "required_ui_label": "MiniMax H3"}`.

## 3. P0 measurements

Measuring machine (2026-11-23, ✅ really measured): RTX 3090 Ti 24 GB (23.99 GiB), 31.7 GB of RAM, Windows 11, PyTorch 2.10.0+cu128, diffusers
`80c7ed2`, 40 steps, same prompt/seed 42, `enable_model_cpu_offload`. Tool: `workers/qwen_image/p0_bench.py` (one separate
process per cell). The table's numbers come from runs made **while the machine was idle**; an earlier run, made while tests were
running in parallel, gave skewed timings (BF16 second run: 268 s) and is **not** used.

| Model | Variant | Size | Peak VRAM (allocated / reserved) | Peak RAM | Load | Image (1st / 2nd run) | Notes |
|---|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | INT8 | 1024² | **9.47 / 9.53 GiB** | 21.3 GiB | 35 s | **41.2 s / 38.6 s** | Fits in VRAM. The result is deterministic (same sha256 twice) |
| Qwen-Image-2.1 | INT8 | 2048² | **26.89 / 32.44 GiB** | 24.3 GiB | 33 s | **264 s** (1 run) | **Exceeds the 24 GiB of physical memory**: the driver spills into RAM; it runs but about 6.5× slower |
| Qwen-Image-2.1 | BF16 | 1024² | **16.37 / 16.54 GiB** | 24.3 GiB | 26 s | **95.1 s / 323.7 s** | Fits in VRAM but system memory is tight (33 GB of weights, 31.7 GB of RAM). The second run is ~3.4× slower than the first, **repeatable on an idle machine**; the cause is not isolated (hypothesis: the file cache is pushed out of RAM) |
| Qwen-Image-2.1 | BF16 | 2048² | to be measured | | | to be measured | |
| Qwen-Image-2.1 | `int8-convrot` / `mixed-p010` / `mixed-p015` (model offload) | 1024² | **8.17 GiB** (all three) | 22–23 GiB | 21–55 s | 51.3 / 54.7 / 52.3 s (2nd run) | §3.4; ✅ through the gateway |
| MiniMax H3 FL2VA | community mixed INT4/INT8 build (§2.0), AHA open's own loader | 5 s · 608×352 / 1344×768 | **17.24 / 19.61 GiB** (allocated) | worker RSS 2.2 GiB | 29 s (transformer, loaded per job) | **283 s / 1 309 s** | §2.3; ✅ through the gateway §3.3. The ~133 GB full model is dropped |
| MiniMax H3 Ref2VA | same build, one reference image | 5 s · 608×352 | **17.07 GiB** (allocated) | worker RSS 1.7 GiB | 39 s (transformer, loaded per job) | **665 s** (cold, whole job) | §2.4; 1344×768 with a reference not measured |

Reading the table: on a 24 GB GPU, **INT8 at 1024² is the comfortable choice** (~40 s, 9.5 GiB). The BF16 rows are kept as the evidence
for dropping the full model: it runs here but is not faster and takes twice the disk. A GPU of ≥ 40 GiB that would hold it without
offload was never available, so nothing was measured there.

### 3.1 Size sweep and LoRA through the real gateway (✅ 2026-11-23, `int8`, seed 11, 40 steps)

| Tier | Sizes | Result | Time per image |
|---|---|---|---|
| 1k | 1024×1024 · 1152×864 · 864×1152 · 1248×832 · 832×1248 · 1376×768 · 768×1376 | ✅ exactly the requested size | 36–38 s |
| native | 2048×2048 · 2400×1792 · 1792×2400 · 2528×1696 · 1696×2528 · 2752×1536 · 1536×2752 | ✅ exactly the requested size | 269–326 s (VRAM overflows) |
| rejected | 1200×896 · 896×1200 · 1264×848 · 848×1264 | the model returns 1184×896 / 896×1184 / 1248×832 / 832×1248: **the pipeline rounds down to a multiple of 32**; the gateway rejects the job instead of delivering a wrongly sized image | — |

LoRA (synthetic, only to check the loading path — not a real style): `synthetic-a`, rank 8 on `to_q/to_k/to_v/to_out.0` of 32 blocks.
`scale` 0.5 / 1.0 / 2.0 gives a mean deviation from the base image of 10 / 11 / 13 on a 255 scale (the image is not empty, no NaN);
**a job without a LoRA run right afterwards gives exactly the base image's sha256** (`625baaeab50e2cc6`) ⇒ nothing is left behind between jobs.
Quality with real (trained) LoRAs: see §3.2.

### 3.2 Real LoRAs through the gateway (✅ 2026-11-23, `int8`, 1024², 40 steps)

Two real files supplied by the project owner (not in git): `natural-skin` (ModelScope, rank 16, 448 tensors, layer names without
prefix, with `.default.`) and `restore` (ai-toolkit 0.13.23, rank 32, 384 tensors, `diffusion_model.` prefix).

| Test | Result |
|---|---|
| `natural-skin` 0.8, portrait prompt, seed 21 | the image differs clearly from the no-LoRA image (closer framing, skin with pores) ⇒ the LoRA has an effect; judged by eye, **no quantitative measure of "better"** |
| `restore` 1.0 on an image blurred + JPEG q12 (`image.edit`) | mean deviation LoRA on/off 2.85/255; against the original image: off 10.34, on 9.73. The base model already restores fairly well so the effect is **small**; one image, one seed — not enough to say "better" |

**Bug found (R24):** on the first run peft only wrote one log line, `Loading adapter weights ... unexpected keys ... img_mlp.gate_up`, and dropped **64 of the
384 tensors** of `restore`. Cause: ai-toolkit fuses two layers into `img_mlp.gate_up` (rows of B = 2 × 12288), while this model
has two layers, `gate_layer` and `proj` (`transformer_qwenimage21.py:206-212`); the job still `succeeded`, the LoRA ran on 5/6 of its layers, with no error. Fixed at the root:
the worker reads the file, strips the prefix/`.default`, folds in `alpha`, splits `gate_up` into `gate_layer` (first half) + `proj` (second half), and **rejects** the whole LoRA if
any tensor still targets a layer that does not exist (`workers/qwen_image/worker.py::lora_state_for`). The first-half/second-half order is **quoted from another
project's code** (an independent open-source Qwen-Image LoRA loader: "first half of the rows ... `gate_layer`, second half `proj`"), not from upstream
documentation; the only independent evidence at that time was that after the fix the on/off effect rose from 1.85 to 2.85 and the warning line disappeared from the log.
**Closed 2026-11-25 (§3.4):** the order is now proved from the weights, by decoding the two halves of the community ConvRot file's fused `img_mlp.gate_up` and comparing them with the separate `gate_layer` and `proj` of the bitsandbytes build of the same model (cosine 1.007 against 0.000 across, `tests/test_qwen_models_real.py`). `natural-skin` gives **exactly the same image sha256** before and after the fix (`b699c70c77c4`) ⇒ the new path does not change the result of a file that already worked.
Regression gate: `tests/test_lora_state.py` (4 tests, sabotage-verified: swapping the halves, dropping `alpha`, dropping the unknown-layer check ⇒ exactly that test turns red).

Consequence for the manifest (design note R20): Qwen's size table has two tiers — `1k` (~1 MP, same ratios, fast; ✅ all 7 sizes generated for real) and `native`
(the model's own ~4 MP, slow when VRAM < ~27 GiB). Qwen images are always returned as RGBA PNG (even when transparency was not requested).
Cloudflare Tunnel (request body limit, timeout): **⬜ not measured** — needs a real Cloudflare account, see `docs/06_deployment.md`.

A model that cannot run on a GPU in the table ⇒ the manifest reports `unsupported_hardware` on that machine, with the minimum
hardware measured elsewhere (if any).

### 3.3 MiniMax H3 through the real gateway (✅ 2026-11-25, RTX 3090 Ti, `mixed`, seed 42, `short_edge` 352)

Driven by the `aha-open-mcp` scenarios `real_video_h3_preview` and `cancel_real_video_h3` against a real `aha-open serve` with the real worker.

| Check | Result |
|---|---|
| Whole job, submit to a fetchable file | 283 s (model loaded for this job, nothing left on the GPU afterwards: 950 MiB used by the desktop) |
| The downloaded file, read from its own mp4 boxes | 608x352, 124 frames, 24 fps, stereo 32 kHz sound, comment `AI-generated (MiniMax H3)`; size, fps, duration, audio, sha256 and byte count all equal what the job reported (19/19 checks) |
| Cancel, 2 s after the job became `running` | `cancelled`, no result (`/result` is HTTP 409); the GPU is free afterwards |

Worker memory, measured later from the worker's own stage lines (§2.4): peak RSS 2.2 GiB (text model) and 1.7 GiB (reference model) while the machine's free RAM stayed above 18 GiB.

**Not measured:** cancel in the middle of denoising at native size through the gateway (this cancel landed while the model was loading; step-level cancel is covered by the worker test, M2), other durations or aspect ratios, first/last-frame input.

### 3.4 The ConvRot variants (✅ 2026-11-25, RTX 3090 Ti, torch 2.10.0+cu128, 1024², 40 steps, seed 42, prompt `A neon sign reading "CREATE WITH LIGHT", rainy night`)

Tools: `workers/qwen_image/p0_convrot.py` (one cell per process) and the product worker driven over its stdin/stdout protocol.

**Why the first version was slow, and what fixed it (p015).** (ii) `comfy_kitchen.int8_linear`: 75.2 s — the `cuda` backend of comfy-kitchen 0.2.42 does not list `int8_linear` among its capabilities,
so the call falls to the `eager` backend (many small kernels); (i) decode the weight to bf16 with the rotation: 66 s, then 62 s when the *activation* is rotated instead; (iv) rotate the activation, quantize per row,
`torch._int_mm`: **44.8 s**. The root cause of the remaining gap was the operand layout: for a 4608×4096 by 4096×12288 product `torch._int_mm(q, w.t())` takes **1.74 ms (267 TOPS)**,
`torch._int_mm(q, w.t().contiguous())` 5.93 ms and a bf16 `linear` 5.68 ms. One layer's error against a float64 reference: 0.90 % for the INT8 product (the W8A8 the checkpoints were made for),
0.29 % for the bf16 decode (both at bf16 noise level).

**Placement.** Everything resident on the GPU peaked at 19.65 GiB (p015), 21.46 GiB (p010) and 22.12 GiB (int8-convrot) of 24: the last two then spilled into system memory
(46 → 61 s and 49 → 66 s between the first and second image). With model-level offload all three peak at **8.17 GiB**. Same seed, different placement gives a different sha256 (the float summation order differs);
each configuration is repeatable (two runs, same sha256).

| Variant | Image, offload (1st / 2nd) | Peak VRAM | Peak RAM | sha256 (first 16) | Through the gateway |
|---|---|---|---|---|---|
| `int8` (bitsandbytes, reference) | 54.5 / 39.8 s (earlier tool, §3) | 9.47 GiB | 20.1 GiB | `625baaeab50e2cc6` | ✅ identical |
| `int8-convrot` | 64.2 / 51.3 s | 8.17 GiB | 23.0 GiB | `1db934655ac9bb91` | ✅ identical |
| `mixed-p010` | 58.6 / 54.7 s | 8.17 GiB | 22.9 GiB | `0f8d6882bc38665e` | ✅ identical |
| `mixed-p015` | 55.6 / 52.3 s | 8.17 GiB | 22.0 GiB | `bdb17b3c16ce4e45` | ✅ identical |

RAM is the real constraint of the ConvRot builds (the worker holds the transformer and the text encoder in system memory for the offload; the machine has 32 GB and 0.4–1.4 GiB were free while loading): **not optimized**.
The image of each variant was looked at and reads "CREATE WITH LIGHT" correctly; the publisher's quality ranking was not reproduced.

**The order of the fused layers (D5).** `img_mlp.gate_up` of the ConvRot file decoded and compared with `gate_layer` and `proj` of the bitsandbytes build (block 0, float64): first half ↔ `gate_layer` cosine 1.007,
second half ↔ `proj` 1.008, crosswise 0.000 / 0.000 (values above 1 are float32 rounding over 50 million elements in the first probe; the test computes in float64 and asserts > 0.99 and < 0.1).
`tests/test_qwen_models_real.py` (skipped without the files), and the loader's order is pinned by an AST test; both sabotage-verified.

**LoRA through the gateway (`int8-convrot`, seed 21, 1024², 40 steps).** `natural-skin` 0.8: `9a037797f65bb9b7`; the same job without a LoRA: `0b013a31f413ca11`; **a no-LoRA job run after the LoRA is
bit-identical to the one before** (also `1db93465…` for the neon image). On `mixed-p015` (which has W4A4 layers) the same holds: with LoRA `09a5a8fa391d95d5`, without `981d5343b093f47c` before and after.
`restore` 1.0 on a blurred + JPEG q12 image (`image.edit`, `int8-convrot`, 30 steps): mean deviation from the original 8.55 (degraded) → 8.68 without the LoRA, **7.87** with it; one image, one seed: **not enough to say "better"**.
`image.edit` through the gateway (a 512² synthetic source, `mixed-p015`, 1024², 20 steps) returned a valid PNG of the requested size; its quality was not judged.

**Install and web page, for real.** 6.55 GB (`mixed-p010`) downloaded through the *Models* panel in a real browser against a real gateway: sha256 matched, the variant appeared as installed, the running variant was not switched;
*Cancel* killed the whole process tree; *Use this* wrote the configuration; *Delete* (cancel, then confirm) removed only that variant's file; deleting the last one offered the shared files separately. With a job running,
`POST /admin/models/{id}/use` answered `409 model_busy` and the configuration did not change (✅ on the real gateway). The delete route was not pointed at the real model files for this check (a refused request would have been the
only thing standing between it and a real deletion); it is covered by unit tests and by the run on a throw-away data folder.

⬜ Not done: the publisher's blind review is not repeated; no quantitative comparison of ConvRot against `int8` image quality; `image.edit` quality against `int8`; RAM reduction; GPUs other than the RTX 3090 Ti (the INT8 path is
chosen from the CUDA capability, the bf16 fallback is covered only by a unit test); a TLS note: on the measuring machine a company CA intercepts HTTPS and Python's `certifi` does not trust it (`CERTIFICATE_VERIFY_FAILED`
from `snapshot_download`); the installer of the product downloaded fine in the run above, so the difference is not explained. The page now says `certificate_untrusted` instead of "no internet" when it happens.

## 4. Adding a new model

1. Measure it using the shape of the §3 table on at least one GPU.
2. Read the license and quote the clauses about commercial use verbatim in a new section of this file.
3. Create `workers/<name>/` with a `pyproject` that pins the upstream exactly.
4. Declare in the manifest exactly what was measured; run the conformance suite.
