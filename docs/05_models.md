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
| On an RTX 3090 Ti 24 GB | ✅ measured, see §3: INT8 1024² ≈ 40 s / 9.5 GiB; BF16 1024² runs (16.4 GiB) but slowly; native 2048² overflows VRAM | §3 |
| Variants AHA open lets you choose | `int8` (weaker machines, ~18 GB of disk) and `bf16` (the full model, ~33 GB of disk, strong machines ≥ 40 GiB of VRAM — this threshold is inferred, not measured). `aha-open models install` suggests one from the GPU and says why | `aha_open/hardware.py` |
| Sizes AHA open declares | 14 sizes: 7 native (`tier: native`) + 7 of ~1 MP (`tier: 1k`, every side divisible by **32** — the pipeline rounds down to a multiple of 32: asking for 1200×896 returns 1184×896). Giving only `aspect_ratio` ⇒ the `1k` tier | design note R20 |
| LoRA | The operator puts a `.safetensors` file in `<data_dir>/loras/` (`aha-open loras add FILE`); the client chooses by id + `scale` (≤ 3 at a time, 0 < scale ≤ 2). ✅ run for real on `int8` with two real LoRAs (see §3.2); a LoRA with a layer the model lacks is **rejected**, never half-applied; `bf16`: not tried, not refused | design note R23 |
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

Manifest: `"license": {"name": "Qwen RESEARCH LICENSE", "commercial_use": "not_allowed"}`.

## 2. MiniMax H3 (video with audio)

> Status: **deferred** — the worker is not part of the current release; this section records what was read from the
> publisher's files so that it is not lost.

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
| On an RTX 3090 Ti 24 GB | **to be measured** — inferred: hard to run without offload/heavy quantization (33B transformer + 32B encoder) | |

`/v1` mapping: `minimax-h3-fl2va` (`video.generate`, `first_frame`, `last_frame`), `minimax-h3-ref2va`
(`video.generate`, `max_reference`). Result: an mp4 with audio (`has_audio: true`).

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
| MiniMax H3 FL2VA | BF16 + offload | 4 s · 768p | to be measured | | | to be measured | **Deferred** (project owner, 2026-11-23). From the index file that was read: transformer 66.28 GB + encoder 66.71 GB ≈ 133 GB for one checkpoint |

Reading the table: on a 24 GB GPU, **INT8 at 1024² is the comfortable choice** (~40 s, 9.5 GiB). BF16 runs but is not faster and
takes twice the disk; the BF16 build suits large GPUs (≥ 40 GiB) where no offload is needed — **not measured on such a GPU** (no
such hardware was available), so the 40 GiB threshold in `hardware.py` is inferred from the weight size (33 GB) plus activations, not a measurement.

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
documentation; the only independent evidence here is that after the fix the on/off effect rose from 1.85 to 2.85 and the warning line disappeared from the log — no test
yet tells the right order from the reversed one. `natural-skin` gives **exactly the same image sha256** before and after the fix (`b699c70c77c4`) ⇒ the new path does not change the result of a file that already worked.
Regression gate: `tests/test_lora_state.py` (4 tests, sabotage-verified: swapping the halves, dropping `alpha`, dropping the unknown-layer check ⇒ exactly that test turns red).

Consequence for the manifest (design note R20): Qwen's size table has two tiers — `1k` (~1 MP, same ratios, fast; ✅ all 7 sizes generated for real) and `native`
(the model's own ~4 MP, slow when VRAM < ~27 GiB). Qwen images are always returned as RGBA PNG (even when transparency was not requested).
Cloudflare Tunnel (request body limit, timeout): **⬜ not measured** — needs a real Cloudflare account, see `docs/06_deployment.md`.

A model that cannot run on a GPU in the table ⇒ the manifest reports `unsupported_hardware` on that machine, with the minimum
hardware measured elsewhere (if any).

## 4. Adding a new model

1. Measure it using the shape of the §3 table on at least one GPU.
2. Read the license and quote the clauses about commercial use verbatim in a new section of this file.
3. Create `workers/<name>/` with a `pyproject` that pins the upstream exactly.
4. Declare in the manifest exactly what was measured; run the conformance suite.
