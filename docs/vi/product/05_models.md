# Model trong AHA open v1

> Mọi số liệu dưới đây có ghi nguồn. "Cần đo" = chưa chạy trên phần cứng của dự án. Bảng đo P0 sẽ được điền ở §3.

## 1. Qwen-Image-2.1 (ảnh)

| Mục | Giá trị | Nguồn |
|---|---|---|
| Loại | Model thống nhất tạo ảnh + sửa ảnh, thành phần sinh ảnh 7B (32 lớp DiT) | README chính thức |
| Task `/v1` | `image.generate`, `image.edit` | |
| Ảnh tham chiếu | ≤ 10 | README chính thức |
| Sửa cục bộ | khoanh tròn, vẽ chú thích, hoặc mask riêng | README chính thức |
| Trong suốt | RGBA (prompt bắt đầu bằng câu mẫu "This is an RGBA image with transparency…") | README chính thức |
| Kích thước khuyến nghị | 1:1 2048×2048 · 4:3 2400×1792 · 3:4 1792×2400 · 3:2 2528×1696 · 2:3 1696×2528 · 16:9 2752×1536 · 9:16 1536×2752 | README chính thức |
| Mở rộng prompt | Hai model viết lại prompt riêng (Qwen3.5-VL 9B, T2I và I2I) — tuỳ chọn, chưa đưa vào v1 | README chính thức |
| Bản INT8 (cộng đồng `ixim/Image21-INT8`) | Trọng số 18,632 GB (BF16 33,116 GB); đỉnh CUDA 12,09 GiB (BF16 19,08 GiB), RTX 5090, 1024², 40 bước, CPU offload | README INT8 |
| Môi trường bản INT8 | Python 3.13, PyTorch 2.10.0+cu128, `transformers==5.17.0`, diffusers ghim commit `80c7ed2…` | README + requirements INT8 |
| Hạn chế đã biết (bản INT8 tự ghi) | Sửa ảnh 1024px làm quá nét; thêm chữ thừa ở poster tiếng Trung; thay nền 2048px thất bại ở cả hai độ chính xác | README INT8 |
| Trên RTX 3090 Ti 24 GB | ✅ đã đo, xem §3: INT8 1024² ≈ 40 s / 9,5 GiB; cỡ gốc 2048² tràn VRAM | §3 |
| Biến thể AHA open cung cấp | **Bốn**, đều lượng tử hoá (RTX 3090 Ti, 1024², 40 bước; số liệu và nguồn ở §1.1 và §3.4 của bản tiếng Anh): `int8` (bản bitsandbytes của cộng đồng ở trên: 18 GB, 9,5 GiB, 40 s), `int8-convrot` (17 GB, 8,2 GiB, 51 s), `mixed-p010` (17 GB, 8,2 GiB, 55 s), `mixed-p015` (15 GB, 8,2 GiB, 52 s). Ba bản ConvRot dùng chung một thư mục (text encoder, VAE, config) và chỉ khác file transformer, nên biến thể thứ hai chỉ tải file riêng của nó. Bản BF16 đầy đủ (~33 GB) đã được đánh giá và bỏ (chủ sản phẩm, 2026-11-23); số đo BF16 giữ ở §3 làm bằng chứng | ghi chú thiết kế R25; plan 2026-11-25 |
| Cỡ ảnh AHA open khai | 14 cỡ: 7 cỡ gốc (`tier: native`) + 7 cỡ ~1 MP (`tier: 1k`, mọi cạnh chia hết cho **32** — pipeline làm tròn xuống bội của 32: yêu cầu 1200×896 ra 1184×896). Chỉ cho `aspect_ratio` ⇒ hạng `1k` | design note R20 |
| LoRA | Người vận hành đặt file `.safetensors` vào `<data_dir>/loras/` (`aha-open loras add FILE`); client chọn theo id + `scale` (≤ 3 cùng lúc, 0 < scale ≤ 2). ✅ chạy thật trên `int8` với hai LoRA thật (xem §3.2); LoRA có lớp lạ bị **từ chối**, không áp dụng một phần | design note R23 |
| `mask` | **Không khai** (`mask: false`): README chỉ nói "separate masks", không có tham số nào; chưa có phép chạy nào cho thấy nó hoạt động ra sao | design note R21 |

**Giấy phép — Qwen RESEARCH LICENSE AGREEMENT** (Release Date: September 20, 2026; toàn văn đã đọc; trích nguyên văn):

- *"'Non-Commercial' shall mean for research or evaluation purposes only."*
- *"…to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Materials FOR
  NON-COMMERCIAL PURPOSES ONLY."*
- *"You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us."*
- Định nghĩa "Qwen" bao gồm *"…inference-enabling code…"* — vì vậy adapter của AHA open chỉ gọi API công khai của
  diffusers, không chép mã từ repo Qwen.
- Phân phối lại phải kèm file Notice: *"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026
  Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."* — AHA open **không** phân phối trọng số nên
  không tự mang file này, nhưng lệnh cài model in lại thông báo này.

- §2.b: muốn dùng thương mại phải xin giấy phép riêng tại *model-business@notice.qwencloud.com*.
- §4.c: *"You shall not use "Qwen" as the primary name or identifier of any derivative works or products; reasonable
  descriptive use (e.g., "fine-tuned from Qwen Image") is permitted."*
- §8: luật Trung Quốc, toà án Hàng Châu.

Hệ quả: người vận hành AHA open chịu trách nhiệm dùng Qwen đúng giấy phép (lệnh cài model đòi `--accept-license`).
Client như AHA Studio chỉ hiện `commercial_use: not_allowed` từ manifest, không có logic riêng cho Qwen.

**Cài từ trang web (`/ui/`, chỉ hồ sơ local).** Model **không** được tải khi cài AHA open; lần đầu mở trang trên máy chạy nó, panel *Download a Model* hiện tên
giấy phép, `commercial_use`, đoạn `notes` của chính giấy phép ("Research or evaluation use only; a separate license is needed for commercial use."), liên kết tới toàn văn,
và một ô tick *"I have read the licence and I accept it."* Nút *Download model* bị khoá cho tới khi tick ô đó; tick là phần tương đương trên web của
`--accept-license qwen-research`, và server từ chối request không mang đúng khoá đó. Sau đó trang hiện giai đoạn (kiểm tra → tải kèm số byte trên ~18 GB → dựng engine của model → hoàn tất),
có thể huỷ (phần đã tải được giữ lại), và nói vì sao không thể bắt đầu trước khi bấm bất cứ gì: không có GPU NVIDIA hoặc dưới ~10 GiB VRAM, không có `uv`, không có thư viện tải,
hoặc còn dưới ~20 GB đĩa trống. Cách hoạt động và API: [04_architecture.md](04_architecture.md) mục 6.1.

Bằng chứng: 🧪 API (test conformance với một script cài giả) và ✅ trang được lái trong trình duyệt thật (Chromium qua Playwright, gateway thật, script cài giả: giấy phép và sự đồng ý,
các giai đoạn tiến độ, nút huỷ, câu khi mất mạng, câu khi hết đĩa, không lặp khi một lần cài kết thúc mà model chưa sẵn sàng). **Đính chính từ lần tải thật đầu tiên:** thanh tiến độ
ban đầu đo thư mục trọng số và đứng ở 0% hàng phút trong khi dữ liệu vẫn đang về (`hf-xet` chỉ ráp tệp ở cuối), một lỗi mà script giả không thể cho thấy vì nó ghi thẳng vào thư mục;
thanh nay dùng bộ đếm byte của chính trình tải. ✅ **Một lần tải thật hoàn chỉnh qua trang** (2026-11-24, RTX 3090 Ti, trình cài Windows vào một thư mục sạch, chạy mã đã cài chứ không phải checkout): lần thử đầu cố ý
bị cắt ở 9,35 GiB, lần bấm thứ hai tiếp tục, và model (INT8, 18 GB) cùng môi trường PyTorch của nó sẵn sàng khoảng 17 phút sau lần bấm thứ hai; job đầu nạp model trong ~120 s, job kế tiếp mất 42 s.
`a lighthouse on a cliff at sunset, oil painting`, seed 42, 1:1: 1024×1024, 2 312 944 byte, **sha256 `c142df9e…ba45` — giống hệt tham chiếu P0**, kiểm trên các byte tải từ
`/v1/jobs/{id}/result` và bằng cách nhìn bức ảnh. Kết nối cho 5–13 MB/s.

Manifest: `"license": {"name": "Qwen RESEARCH LICENSE", "commercial_use": "not_allowed"}`.

## 2. MiniMax H3 (video có âm thanh)

> Trạng thái: **chạy thật, cài từ dòng lệnh hoặc từ trang web** (2026-11-25). Bản cộng đồng lúc đầu được chạy trong một runtime tham chiếu (§2.2); worker riêng của AHA open
> (`workers/minimax_h3/`) nay nạp nó mà không cần ComfyUI (§2.3) và hai model được phục vụ qua `/v1`: `minimax-h3-fl2va` (chữ sang video) và `minimax-h3-ref2va`
> (một ảnh tham chiếu, §2.4). Cả hai cài được từ dòng lệnh hoặc từ panel *Download a Model* của trang web, panel này hỏi giấy phép **và** một xác nhận riêng rằng người dùng
> không ở một vùng mà giấy phép loại trừ ([04_architecture.md](04_architecture.md) mục 6.1).

| Mục | Giá trị | Nguồn |
|---|---|---|
| Loại | Model sinh video + âm thanh đa phương thức | README chính thức |
| Thời lượng | 4–15 giây | README |
| Khung hình | 24 fps | README |
| Độ phân giải | Cạnh ngắn mặc định 768 px; 2K qua bước "H3-Regenerate-2K" (chạy lại chính model) | README |
| Tỉ lệ | Nhiều, gồm 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 | README |
| Âm thanh | Stereo 32 kHz; lời thoại ổn định ở 11 ngôn ngữ (không có tiếng Việt trong danh sách) | README |
| Checkpoint FL2VA | Text→video; 1 ảnh (khung đầu hoặc cuối); 2 ảnh (đầu + cuối) | README |
| Checkpoint Ref2VA | Tham chiếu: ≤ 9 ảnh, ≤ 3 video, ≤ 3 audio (mỗi clip 2–15 s, tổng ≤ 15 s), tối đa 12 tệp | README |
| Độ chính xác | BF16; Omni-Transformer "33B" | README, requirements |
| Phục vụ chính thức (ví dụ) | `sglang serve --num-gpus 4` | README |
| Môi trường | diffusers ghim commit `f53d552…`, `transformers>=4.57`, torch ≥ 2.4 | requirements |
| Mở rộng prompt "H3-Context-IR" | Gọi **API cloud của MiniMax** (cần token), không chạy local. AHA open **không** bật mặc định — gửi prompt ra bên thứ ba trái nguyên tắc dữ liệu | script `full-2k-t2va-h3-context-ir.sh` |
| Bộ mã hoá | Qwen3-VL-32B (Apache-2.0) | ghi chú LICENSE |
| Trên RTX 3090 Ti 24 GB | ✅ một clip 5 s chạy được trong một runtime tham chiếu: 3–3,5 phút ở 608×352, **khoảng 19–20 phút ở 1344×768**; xem §2.2 | đã đo, §2.2 |

Ánh xạ `/v1`: `minimax-h3-fl2va` (`video.generate`, `first_frame`, `last_frame`), `minimax-h3-ref2va`
(`video.generate`, `max_reference`). Kết quả: mp4 có âm thanh (`has_audio: true`).

### 2.0 Bản duy nhất AHA open sẽ dùng (chủ sản phẩm quyết 2026-11-23)

Model đầy đủ (~133 GB: transformer 66,28 GB + bộ mã hoá 66,71 GB, đọc từ tệp index) **bị bỏ**. Bản MiniMax H3 duy nhất AHA open sẽ hỗ trợ là bộ lượng tử hoá
cộng đồng **`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`**. Điều đã đọc từ trang công khai và danh sách tệp của nó (2026-11-23; ✅ chỉ đọc, chưa tải hay chạy gì):

| Mục | Giá trị |
|---|---|
| Revision đã xem | `908eccad7e68751190d04c171956f163bfeed741` (sửa lần cuối 2026-08-15) |
| Trường giấy phép | `license: other`, `license_name: minimax-h3-community-license-agreement` (giấy phép của model gốc, §2.1) |
| Model gốc | `MiniMaxAI/MiniMax-H3`, `base_model_relation: quantized` |
| Tệp FL2VA | `…FL2VA_pruned_mixed_int4_int8_convrot.safetensors` 14,81 GiB · `…_pruned_int8_convrot` 19,53 GiB · `…_pruned_nvfp4` 11,67 GiB (NVFP4 cần GPU Blackwell) |
| Tệp Ref2VA | `…Ref2VA_pruned_mixed_int4_int8_convrot` 14,06 GiB · `…_pruned_int8_convrot` 19,53 GiB · `…_pruned_nvfp4` 11,67 GiB · `…_nvfp4_mixed` 22,76 GiB |
| VAE (mọi bản đều cần) | video 4,85 GiB + audio 0,56 GiB |
| Điều trang tự nhận (bên thứ ba, **chưa kiểm**) | tệp mixed INT4/INT8 "requires 15+ VRAM (~15.5 GB)"; INT8 "24GB+ VRAM"; các bộ mã hoá văn bản trong `/text_encoders` (15,0–27,1 GB) |

#### Phần đầu safetensors đã đọc (✅ 2026-11-23, chỉ HTTP Range, không tải gì)

Đã đọc: 8 byte độ dài + phần đầu JSON (95–103 KB) của ba tệp FL2VA, bằng các request `Range` tới `resolve/main`.

| Tệp | Số tensor | Lượng tử hoá (từ chính các tệp) | Kiểu dữ liệu |
|---|---|---|---|
| `…FL2VA_pruned_int8_convrot` | 932 | mỗi lớp `*.comfy_quant` = `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` (**đính chính 2026-11-24**: mọi lớp lượng tử được lấy mẫu của tệp này, đọc từ chính các blob; lần đọc đầu nhầm cho nó định dạng `convrot_w4a4`) — trọng số INT8 xoay bằng biến đổi Hadamard, một hệ số tỉ lệ cho mỗi hàng đầu ra | I8 ×200, U8 ×200, F32 ×210, BF16 ×220, F16 ×102 |
| `…FL2VA_pruned_mixed_int4_int8_convrot` | 932 | cùng định dạng; metadata `convrot_w4a4_mixed`: `"w4a4_int8mm_layers": 117, "int8_layers": 83, "linear_dtype": "int4"`, `"paper": "arXiv:2512.03673"` | cùng số lượng; `blocks.0.attn.out_proj.weight` I8 `[5376, 3584]` |
| `…FL2VA_pruned_nvfp4` | cùng bố cục | `*.comfy_quant` = `{"format": "nvfp4"}`; `weight` U8, `weight_scale` F8_E4M3, `weight_scale_2` F32 | U8, F8_E4M3, BF16, F16, F32 |

**Các tệp đến từ đâu (✅ 2026-11-24, cả ba đã kiểm với sha256 của nhà phát hành).** Repo Abiray **không có bộ mã hoá văn bản** (commit cuối của nó có tiêu đề "Delete text_encoders"),
mà bộ mã hoá là bắt buộc (*"The H3-Encoder uses the full pretrained weights of Qwen3-VL-32B"*). Workflow ví dụ của chính nhà phát hành (`ref2.json`) lấy các VAE và bộ mã hoá từ
`Comfy-Org/MiniMax-H3`. Vì vậy AHA open dùng:

| Phần | Repo @ revision | Tệp | sha256 (16 ký tự đầu) |
|---|---|---|---|
| Transformer, mixed | `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` @ `908eccad…` | `MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors` (14,81 GiB) | `541c7797d815a165` |
| Transformer, INT8 | như trên | `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors` (19,53 GiB) | `f07a54277ca85b59` |
| VAE video / audio | như trên | `vae/minimax_h3_video_vae_fp16.safetensors` (4,85 GiB) / `vae/minimax_h3_audio_vae_fp32.safetensors` (0,56 GiB) | `88539d729fd7e9f9` / `83043bf3afccf4ca` |
| Bộ mã hoá văn bản | `Comfy-Org/MiniMax-H3` @ `4b53a351…` (**chỉ vì Abiray không có**) | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (25,28 GiB) | `bc2ced0fbea64757` |

Không dùng: `yuivs/Minimax-H3-nvfp4-INT4-INT8-Convrot` (bản sao giống hệt từng byte của các tệp Abiray, cùng băm); các tệp transformer và VAE của Comfy-Org (khác Abiray từng byte, vd
VAE video `7c1f1314…` so với `88539d72…`). Tệp bộ mã hoá là INT8 theo từng hàng đầu ra (`weight` I8 + `weight_scale [rows, 1]`), tên khoá của HuggingFace, chỉ 50 lớp đầu, và metadata của nó nói
đầu ra là *"unnormalized_hidden_after_layer_50"*. Việc tải phải theo từng tệp với revision được ghim và kiểm băm; `git clone` repo Abiray kéo cả bảy checkpoint (~115 GiB) và một lần bị ngắt đã để
lại 76 GB tệp dở trên máy đo.

Sự kiện: tên tensor theo kiểu ComfyUI (`blocks.N.attn.qkv_proj`, `blocks.N.mlp.fc1`, `blocks.N.adaln_proj.linear`, `token_refiner.*`, `adaln_t_table`), mỗi lớp tuyến tính lượng tử có một blob JSON
`*.comfy_quant`, và `config` trong metadata của tệp mixed cho `hidden_size 5376`, `num_layers 50`, `num_attention_heads 56`, `latents_dim 24`, `audio_latents_dim 32`, `patch_size [1,2,2]`. Mọi kiểu dữ liệu đều chuẩn.
Suy diễn (chưa quan sát): độ rộng 3584 = 7168/2 của `out_proj` trong tệp mixed gợi ý hai giá trị INT4 được đóng gói trong một byte.

Các tệp Ref2VA có cùng định dạng (`…Ref2VA_pruned_mixed_int4_int8_convrot`: 932 tensor, cùng blob `comfy_quant`; `…Ref2VA_nvfp4_mixed`: 951 tensor, hệ số F8_E4M3, thêm một `time_embedder`). Chưa đọc: các tệp chỉ INT4 và
các VAE. Repo **không có thư mục `text_encoders/`** (cấp cao nhất của cây: bảy checkpoint, `README.md`, `ref2.json`, `Amy`, `vae/`) dù README của nó liệt kê ba tệp bộ mã hoá văn bản.

**Hệ quả (✅ từ các phần đầu tệp; cách sửa là §2.3):** đây không phải checkpoint diffusers. Tên khoá và bố cục `comfy_quant` của chúng không phải điều `from_single_file` / mã diffusers của worker Qwen mong đợi, và chạy
ConvRot W4A4 cần một bộ nạp cùng các kernel INT8/INT4 không nằm trong diffusers. "Nạp chúng bằng worker diffusers" do đó **bị bác bỏ như phát biểu**. Điều này tạm dừng hỗ trợ MiniMax H3 cho tới khi một runtime cho định dạng này
được chọn và xây (§2.3: một bộ nạp riêng của AHA open cho pipeline diffusers).

**Những gì đã có cho định dạng này** (✅ đọc từ nguồn công khai ngày 2026-11-23, chưa tải gì):

- Nó được hiện thực bởi ComfyUI (`Comfy-Org/ComfyUI`, **GPL-3.0** theo GitHub API; model H3 nằm ở `comfy/ldm/minimax/model.py`, `convrot_w4a4` ở `comfy/quant_ops.py`) và bởi thư viện `comfy-kitchen`
  (**Apache-2.0**, trên PyPI: kiểu tensor lượng tử và kernel, không có mã model). Repo chính thức của MiniMax (`MiniMax-AI/MiniMax-H3`) kèm mã VAE; trường giấy phép của nó là "other".
- Bản phân phối ở định dạng ComfyUI chính thức là **`Comfy-Org/MiniMax-H3`** (trường giấy phép `minimax-h3-community-license-agreement`, không bị khoá). Nó có mọi thứ pipeline cần: `diffusion_models/` (pruned INT8-ConvRot 19,53 GiB,
  pruned W6A8 14,89, pruned FP8 19,52, INT8 đầy đủ 31,70), `text_encoders/` (INT8-ConvRot 25,28 GiB, NVFP4-AWQ 14,61 cho Blackwell, BF16 47,97) và `vae/` (video 4,85 GiB, audio 0,56 GiB).
- Chưa đo, và lý do chưa thử ở đây: một máy GPU 24 GiB / RAM 31,7 GiB sẽ cần transformer INT8 (19,5 GiB) cộng bộ mã hoá văn bản INT8 (25,3 GiB) cộng các VAE (5,4 GiB), khoảng 50 GiB cần tải, và hai phần lớn không thể cùng nằm trong RAM,
  nên runtime phải nạp chúng lần lượt (ComfyUI làm vậy).

**Các câu hỏi mở như tình trạng ngày 2026-11-24 — đã được trả lời ở §2.3 và §2.4 (✅ 2026-11-25); những câu vẫn còn mở được đánh dấu** (⬜):
- Các tệp là **checkpoint ComfyUI một tệp của một model "pruned"**, không phải thư mục pipeline diffusers. Việc đọc phần đầu ở trên cho thấy worker diffusers không nạp được chúng nguyên trạng; runtime nào nạp được (và cắt tỉa làm mất bao nhiêu chất lượng) chưa được thử.
- ✅ Worker dùng tệp nào: tệp mixed INT4/INT8 (biến thể duy nhất được cung cấp; NVFP4 loại trừ mọi GPU trừ Blackwell).
- ~~Các bộ mã hoá văn bản mà trang nêu không có trong danh sách tệp~~ — đã giải quyết 2026-11-24: dùng bộ từ `Comfy-Org/MiniMax-H3` (bảng trên).
- Thời gian mỗi clip và đỉnh mà runtime đã chọn: đo ở §2.2. Đỉnh mà bộ nạp riêng của AHA open cần được đo ở §2.3 (17,24 GiB ở 608×352, 19,61 GiB ở 1344×768), đó là nguồn của `min_vram_mib` (18 432).
  ⬜ Chất lượng so với model chưa lượng tử chưa được so sánh.
- ✅ Runtime nào worker dùng mà không cần cài ComfyUI riêng: một worker dựa trên diffusers với bộ nạp riêng trên `comfy-kitchen` (hướng đã duyệt 2026-11-24, đã xây và kiểm với các clip của §2.2 ở §2.3).
- Giấy phép là của model gốc; một bản tải lại của cộng đồng không đổi §2.1. Nghĩa vụ của người vận hành (lãnh thổ, nhãn UI, dịch vụ cho người khác) vẫn như vậy.

### 2.2 Các lần chạy tham chiếu trên RTX 3090 Ti (✅ 2026-11-24)

Đã chạy gì: ComfyUI 0.39.0 (commit `b26625f`, khởi động như một tiến trình riêng chỉ nghe loopback, tắt custom node, thư mục đầu ra riêng) nạp các tệp ở §2.0 và sinh các clip chữ→video: prompt *"a lighthouse on a cliff at sunset, waves crashing on the rocks,
seagulls calling, cinematic"*, seed 42, 124 khung (khoảng 5,2 s ở 24 fps), 20 bước, `res_multistep`, lịch `simple`, một guider không có negative prompt (chuỗi trong ví dụ của nhà phát hành, trừ các custom node SageAttention và RTX-upscaler).
Công cụ chạy là `tools/h3_oracle/run_oracle.py` (chỉ trên máy phát triển; nó đọc bảng tiến trình và `nvidia-smi`, không gì khác). Máy: RTX 3090 Ti 24 GiB, RAM 31,7 GiB.

| Transformer | Kích thước | Thời gian job | VRAM đỉnh (cả GPU) | RAM đỉnh của tiến trình | RAM trống thấp nhất của máy |
|---|---|---|---|---|---|
| mixed INT4/INT8 | 608×352 | 199 s | 23 811 MiB | 8,95 GiB | 11,13 GiB |
| INT8 | 608×352 | 179 s | 23 774 MiB | 9,06 GiB | 7,49 GiB |
| mixed INT4/INT8 | 1344×768 | **1 180 s** (19 phút 40 s) | 23 852 MiB | 13,65 GiB | 6,69 GiB |
| INT8 | 1344×768 | **1 130 s** (18 phút 49 s) | 23 818 MiB | 16,83 GiB | 3,15 GiB |

Cách đọc: lần chạy đầu của công cụ này báo 0 MiB và 0,01 GiB vì luồng lấy mẫu của nó chết do khác biệt API psutil mà không để lại dấu vết; công cụ nay đếm số vòng và lỗi và ghi `sampler_trustworthy`
(true ở mọi dòng trên, ≥ 196 vòng, 0 lỗi). **VRAM đỉnh là bộ nhớ đã dùng của cả GPU với cơ chế offload động của ComfyUI: nó lấp đầy card, nên ~23,8 GiB là mức runtime chọn dùng, không phải mức tối thiểu model cần.**
Mức tối thiểu chưa được đo. Tệp INT8 ở đây không chậm hơn tệp mixed nhỏ hơn (kernel của nó là các phép nhân INT8 thường); tệp mixed tiết kiệm đĩa (14,81 so với 19,53 GiB) và RAM ở kích thước lớn hơn, không phải thời gian.

Nội dung, kiểm trên các khung và âm thanh đã giải mã chứ không phải trên "job thành công": mọi clip là H.264 đúng kích thước yêu cầu (608×352 và 1344×768), 24 fps, 124 khung, 5,167 s, với một track AAC stereo 32 kHz có RMS 0,010–0,018 (không phải im lặng);
khung đầu và khung cuối khác nhau (sai khác tuyệt đối trung bình 5–19 trên 255) và các khung giữa cho thấy sóng vỗ và chim bay. Nhìn bằng mắt: một ngọn hải đăng trên vách đá lúc hoàng hôn với biển, đúng như prompt. **Không thấy kết nối tới địa chỉ ngoài loopback**
từ cây tiến trình ComfyUI trong cả bốn lần chạy (bảng socket lấy mẫu mỗi giây). Chưa kiểm: đầu vào khung đầu/cuối, nội dung của âm thanh, khả năng lặp lại của một seed, và mọi điều về chất lượng so với model chưa lượng tử.

Hệ quả cho sản phẩm: ở cạnh ngắn mặc định của model (768), một clip mất khoảng 19 phút trên GPU này, nên thiết kế job + thăm dò là bắt buộc và manifest phải nói vậy; 608×352 mất khoảng 3 phút. Worker của AHA open được viết sau đó
(plan 2026-11-24_minimax_h3_plan.md, các phase M0′–M2) và được so với các clip này ở §2.3.

### 2.3 Bộ nạp riêng của AHA open, không ComfyUI (✅ 2026-11-25, cùng RTX 3090 Ti, bản mixed)

`workers/minimax_h3/` dựng pipeline MiniMax H3 của diffusers từ cùng các tệp đó (thiết kế và các phép kiểm: [04_architecture.md](04_architecture.md) mục 6.2). Cùng prompt và seed 42, 124 khung, 20 bước (19 lần gọi transformer: lưới của scheduler có 19 bước),
ba giai đoạn (mã hoá, lấy mẫu, giải mã) để mỗi lúc chỉ một phần lớn nằm trên GPU.

| Kích thước | Mã hoá | Nạp transformer | Lấy mẫu | Giải mã | **Tổng** | CUDA cấp phát đỉnh |
|---|---|---|---|---|---|---|
| 608×352 | 53,5 s | 28,9 s | 174,2 s (9,2 s/bước) | 20,8 s | **283 s** | 17,24 GiB |
| 1344×768 | 55,0 s | 30,5 s | 1 160,4 s (61 s/bước) | 50,0 s | **1 309 s** (21 phút 49 s) | 19,61 GiB |

So sánh, bản tham chiếu ComfyUI ở §2.2 mất 199 s và 1 180 s cho hai kích thước đó: bộ nạp của AHA open chậm hơn 42% ở 608×352 và 11% ở 1344×768, nơi việc lấy mẫu (61 s/bước so với 59 s/bước của ComfyUI) ngang nhau
và phần còn lại là bộ mã hoá đọc từ đĩa mỗi lần (55 s) và transformer nạp mỗi lần (30 s). Nội dung được kiểm trên pixel và âm thanh đã giải mã, không phải trên "xong": 124 khung đúng kích thước yêu cầu, một track âm thanh stereo 32 kHz (RMS 0,008–0,014),
và các khung cho thấy prompt ở cả hai kích thước (nhìn bằng mắt). Bộ ghi mp4 (OpenH264) tái tạo các khung đã sinh ở 39 dB PSNR. Chưa so: chính clip với clip của ComfyUI (sampler khác, nên không có phép so từng bit), đầu vào khung đầu/cuối.

**Một lỗi tìm ra và sửa dọc đường (nguyên nhân gốc, không phải giải pháp tình thế).** Lần chạy 1344×768 đầu mất 2 385 s: 118 s mỗi bước. Một khối thật đo riêng mất 1,19 s khi còn chỗ trên card và **3,95 s khi card đầy** (cùng mã, 0 lần thử lại của bộ cấp phát):
trên Windows driver NVIDIA có thể chuyển bộ nhớ GPU sang RAM hệ thống thay vì báo lỗi, một cách âm thầm. Cách sửa có ba phần: trọng số được giải mã sang bf16 ngay trước mỗi phép nhân (phép nhân INT8 của `comfy-kitchen` chạy phép xoay và lượng tử hoá thành các kernel riêng trên GPU
không có cuBLASLt và đo chậm hơn phép nhân bf16); feed-forward chạy theo các lát 8 192 token; và PyTorch bị giới hạn ở bộ nhớ GPU còn trống khi job bắt đầu, nên vượt quá là một lỗi. Giải mã trọng số cũng để kích hoạt không bị lượng tử hoá: phép nhân tuyến tính sai khác 0,23% so với phép nhân dày
với trọng số đã giải mã (đường kích hoạt INT8 cho 1%), và trọng số giải mã bằng đúng bộ giải mã của chính `comfy-kitchen`.

### 2.4 Ảnh tham chiếu: `minimax-h3-ref2va` (✅ 2026-11-25, cùng RTX 3090 Ti, bản mixed)

Cùng bản cộng đồng đó còn có một checkpoint "omni-reference" (`MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors`, 15 093 774 276 byte, sha256 `434c372cbb68ff055ffb5aa2c35de42211d19ebacc2ae8e9ae9202b00ee48afb`, cả hai **đã kiểm sau khi tải**).
AHA open chạy nó bằng cùng bộ nạp qua workflow `ref2va` của diffusers (`transformer_ref`; `config.json` của nó giống hệt của model chữ). Bộ mã hoá, cả hai VAE và các cấu hình nhỏ là cùng các tệp với model chữ (cả 20 tệp nhỏ của `FL2VA/` và `Ref2VA/` có cùng băm trên Hub), nên model tham chiếu được cài
**vào thư mục của model chữ** và một máy đã có model chữ chỉ tải transformer 14 GiB (`aha-open models install minimax-h3-ref2va --accept-license minimax-h3 --confirm-territory`).

| Đã đo (một ảnh tham chiếu 256×160, preview 608×352, 124 khung, 20 bước) | Kết quả |
|---|---|
| Cả job qua một gateway thật và harness (`real_ref2va_one_image`) | **PASS 20/20, 665 s**: tệp tải về là 608×352, 124 khung, 24 fps, stereo 32 kHz, nhãn `AI-generated (MiniMax H3)`; kích thước, fps, thời lượng, âm thanh, sha256 và byte bằng điều job đã báo |
| Cùng job qua trang web (Playwright) | một clip 608×352, 5,175 s trong phần tử `<video>`, dòng kết quả `… 24 fps, with sound, seed … - MiniMax H3` |
| Các giai đoạn của một job (khi nguội) | mã hoá 82,7 s (prompt + tháp thị giác + VAE tham chiếu) · nạp transformer 39,4 s · lấy mẫu **499 s** (19 lượt, ~26 s mỗi lượt: tham chiếu thêm hàng) · giải mã 26,3 s |
| Bộ nhớ GPU | đỉnh **17,07 GiB** cấp phát (model chữ ở cùng kích thước: 17,2) |
| RAM của worker | RSS đỉnh 1,7 GiB; RAM trống của máy luôn trên 18 GiB (lấy mẫu ở các dòng giai đoạn) |
| Tham chiếu có quan trọng không? (cùng prompt, seed, 8 bước, 384×224; ảnh đỏ so với ảnh xanh) | R−B trung bình của clip **+37,3** so với **−71,6**: chênh 108,9 trên thang 0–255 (một seed) |
| Model chữ sau thay đổi (`real_video_h3_preview`) | vẫn PASS 19/19, 290 s (trước là 283 s): không chậm đi |

Hai lỗi tìm ra dọc đường và sửa tại nguyên nhân (plan mục 3.10):

1. Bảng rotary của tháp thị giác là một buffer tính toán không có trong tệp; để trên thiết bị `meta` nó làm job hỏng trước khi bắt đầu. Nay nó được dựng lại như của mô hình ngôn ngữ.
2. Attention theo nhóm truy vấn (grouped-query) của Qwen3-VL đòi PyTorch `enable_gqa=True`. Chỉ kernel flash và math nhận cờ đó mà bản Windows không có kernel flash, nên một ảnh tham chiếu (6 544 token sau khi ảnh được đưa lên cạnh ngắn 2 048) cần một ma trận điểm float32
   64 × 6 544²: 10,21 GiB, hết bộ nhớ trong *bộ mã hoá*. Worker nay đăng ký một attention "sdpa" lặp các head key/value trước (cùng số học, đã kiểm tới 1e-5), giữ được kernel tiết kiệm bộ nhớ (đỉnh bộ mã hoá 9,7 GiB với một tham chiếu).

**Đã khai báo (chỉ những gì đã chạy):** một **ảnh** tham chiếu, chỉ cỡ `preview`. **Chưa chạy, nên bị từ chối bằng `400`:** nhiều ảnh, video tham chiếu, audio tham chiếu (đường mã có trong worker; chưa đo gì), native 1344×768 với tham chiếu, khung đầu/cuối.
README của nhà phát hành liệt kê ≤ 9 ảnh, ≤ 3 video, ≤ 3 audio; mở rộng `max_reference` là một phép đo, không phải một cái công tắc.

### 2.5 Các biến thể người vận hành chọn được (✅ 2026-11-26, RTX 3090 Ti 24 GiB; bản đầy đủ: tiếng Anh §2.5)

Repo `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` (@ `908ecca…`) có ba bản transformer đã tỉa cho mỗi model: `int8` (19,53 GiB),
`mixed` (14,81 / 14,06) và `nvfp4` (11,67). README còn liệt kê bản INT4 thuần và bộ mã hoá INT4/NVFP4 nhưng **các tệp đó không có**.
Mọi bản dùng chung bộ mã hoá INT8 (Comfy-Org) và hai VAE. Nhà phát hành làm NVFP4 cho Blackwell; AHA open giải mã nó sang 16 bit
nên chạy được trên GPU NVIDIA bất kỳ (đo trên Ampere).

| Model, bản | 608×352 | 1344×768 | Thẻ |
|---|---|---|---|
| FL2VA `int8` | 21,40 GiB · 278 s | ❌ hết bộ nhớ | 24 GB |
| FL2VA `mixed` | 16,71 GiB · 293 s | 19,60 GiB · 1 376 s | 24 GB |
| FL2VA `nvfp4` | 13,08 GiB · 242 s (trần 15,5 GiB: ✅) | 16,43 GiB · 1 365 s | 16 GB (cỡ nhỏ), 24 GB |
| Ref2VA `mixed` | 17,07 GiB · 665 s (đo bằng mã cũ: cận trên) | chưa đo | 18 GB trở lên |
| Ref2VA `int8` | ❌ hết bộ nhớ trên 24 GB: **không cung cấp** | — | — |
| Ref2VA `nvfp4` | 14,55 GiB · 672 s (trần 15,5 GiB: ❌) | chưa đo | 18 GB trở lên |

Thẻ dưới 16 GB: MiniMax H3 không được cung cấp. Số "trần 15,5 GiB" là giả lập thẻ 16 GB trên 3090 Ti, **chưa thử trên thẻ 16 GB thật**.
RAM: đỉnh RSS của worker 20–22 GiB (gồm trang tệp được ánh xạ); máy 16 GB RAM chưa thử.

### 2.1 Giấy phép — "MiniMax H3 COMMUNITY LICENSE AGREEMENT"

License date: August 2, 2026. Bên cấp phép: Nanonoble Pte. Ltd. Nguồn: `huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE`
(đọc 2026-11-22 qua công cụ web; câu dài bị cắt ở 125 ký tự, đánh dấu "(cắt)" — **đối chiếu lại toàn văn trước khi worker MiniMax được phát hành**).

| Điều | Trích dẫn | Hệ quả cho AHA open |
|---|---|---|
| Lãnh thổ (§II, Đn 3, Đn 5) | *"Solely within the Applicable Territory…"* (cắt); Excluded Territories: *"the European Union, the United Kingdom, the Republic of Korea and the United States of America."* | Không dùng được ở EU/Anh/Hàn/Mỹ. Lệnh cài đòi `--confirm-territory` |
| Thương mại (§IV.1) | Cần cho phép riêng *"if your commercial products and services generate more than 20 million US dollars (or equivalent in other currencies) in yearly revenue."* | Dưới ngưỡng: được thương mại. `commercial_use: allowed_with_conditions` |
| Hiện tên (§IV.2) | *"You shall prominently display "MiniMax H3"on the user interface of commercial product or service"* | `required_ui_label: "MiniMax H3"`; AHA Studio hiện nguyên văn |
| Dịch vụ cho người khác (Đn 8, §V.2, §V.5) | Hosted Services gồm *"…application programming interfaces (APIs), web access…"*; phải *"bind each recipient or user to enforceable terms…"* (cắt); biện pháp an toàn + *"mechanism for reporting suspected violations"* (cắt) | Tắt mặc định ở hồ sơ `internet`; bật cần người vận hành chấp nhận nghĩa vụ |
| Output (§VI.4, §V.3) | *"MiniMax claims no rights over the Outputs you generate."*; không dùng output để cải tiến model AI khác | Ghi trong tài liệu người dùng |
| Công bố nội dung máy tạo (Exhibit A.12, §III.3.b) | Cấm phát tán công khai *"without clearly and prominently disclosing that such information and/or content is machine-generated"* (cắt); khuyến khích thêm nhận diện AI vào file | Ghi nhãn AI vào metadata mp4 (không ghi prompt) |
| Tên/nhãn hiệu (§VI.2) | *"the Licensee may not use any name or mark owned by or associated with MiniMax"* (cắt) | Không tên sản phẩm/lệnh nào mang "MiniMax"; `minimax-h3-*` chỉ là id mô tả |
| Phân phối (§III.1–§III.4) | Kèm bản Agreement, đánh dấu file sửa, file NOTICE *"MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved."* | AHA open không phân phối trọng số; lệnh cài in lại NOTICE |
| Cấm dùng (Exhibit A) | 20 mục, gồm vi phạm pháp luật, hại trẻ em, tin giả ảnh hưởng bầu cử, mạo danh, quyết định tự động rủi ro cao, mục đích quân sự | Lệnh cài in danh sách; người vận hành chịu trách nhiệm |
| Luật (§IX) | Luật Hồng Kông | |

Ghi chú: bộ mã hoá của MiniMax H3 dùng **Qwen3-VL-32B** (Apache-2.0), không phải Qwen RESEARCH LICENSE.

Manifest: `"license": {"name": "MiniMax H3 Community License", "commercial_use": "allowed_with_conditions",
"excluded_territories": ["EU","GB","KR","US"], "required_ui_label": "MiniMax H3"}`.

## 3. Bảng đo P0

Máy đo (2026-11-23, ✅ đo thật): RTX 3090 Ti 24 GB (23,99 GiB), RAM 31,7 GB, Windows 11, PyTorch 2.10.0+cu128, diffusers
`80c7ed2`, 40 bước, cùng prompt/seed 42, `enable_model_cpu_offload`. Công cụ: `workers/qwen_image/p0_bench.py` (mỗi ô một
tiến trình riêng). Số liệu trong bảng lấy từ lần chạy **khi máy rảnh**; một lần chạy trước đó, lúc tôi đang chạy test
song song, cho số thời gian lệch (BF16 lần 2: 268 s) nên **không** được dùng.

| Model | Biến thể | Kích thước | VRAM đỉnh (cấp phát / giữ) | RAM đỉnh | Nạp | Ảnh (lần 1 / lần 2) | Ghi chú |
|---|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | INT8 | 1024² | **9,47 / 9,53 GiB** | 21,3 GiB | 35 s | **41,2 s / 38,6 s** | Vừa VRAM. Kết quả xác định (cùng sha256 hai lần) |
| Qwen-Image-2.1 | INT8 | 2048² | **26,89 / 32,44 GiB** | 24,3 GiB | 33 s | **264 s** (1 lần) | **Vượt 24 GiB vật lý**: driver tràn sang RAM, chạy được nhưng chậm gấp ~6,5 lần |
| Qwen-Image-2.1 | BF16 | 1024² | **16,37 / 16,54 GiB** | 24,3 GiB | 26 s | **95,1 s / 323,7 s** | Vừa VRAM nhưng bộ nhớ hệ thống căng (33 GB trọng số, RAM 31,7 GB). Lần 2 chậm hơn lần 1 ~3,4 lần, **lặp lại được khi máy rảnh**; nguyên nhân chưa tách được (giả thuyết: bộ đệm tệp bị đẩy ra khỏi RAM) |
| Qwen-Image-2.1 | BF16 | 2048² | cần đo | | | cần đo | |
| MiniMax H3 FL2VA | bản mixed INT4/INT8 cộng đồng (§2.0), bộ nạp riêng của AHA open | 5 s · 608×352 / 1344×768 | **17,24 / 19,61 GiB** (cấp phát) | RSS worker 2,2 GiB | 29 s (transformer, nạp mỗi job) | **283 s / 1 309 s** | §2.3; ✅ qua gateway ở §3.3. Model đầy đủ ~133 GB bị bỏ |
| MiniMax H3 Ref2VA | cùng bản, một ảnh tham chiếu | 5 s · 608×352 | **17,07 GiB** (cấp phát) | RSS worker 1,7 GiB | 39 s (transformer, nạp mỗi job) | **665 s** (nguội, cả job) | §2.4; 1344×768 với tham chiếu chưa đo |

Đọc bảng: trên GPU 24 GB, **INT8 ở 1024² là lựa chọn thoải mái** (~40 s, 9,5 GiB). Các dòng BF16 giữ lại làm bằng chứng cho việc bỏ
model đầy đủ: chạy được nhưng không nhanh hơn và tốn gấp đôi đĩa. GPU ≥ 40 GiB chứa được mà không cần offload chưa từng có để đo.

### 3.1 Quét cỡ ảnh và LoRA qua gateway thật (✅ 2026-11-23, `int8`, seed 11, 40 bước)

| Hạng | Cỡ | Kết quả | Thời gian/ảnh |
|---|---|---|---|
| 1k | 1024×1024 · 1152×864 · 864×1152 · 1248×832 · 832×1248 · 1376×768 · 768×1376 | ✅ đúng cỡ yêu cầu | 36–38 s |
| native | 2048×2048 · 2400×1792 · 1792×2400 · 2528×1696 · 1696×2528 · 2752×1536 · 1536×2752 | ✅ đúng cỡ yêu cầu | 269–326 s (VRAM tràn) |
| bị bác | 1200×896 · 896×1200 · 1264×848 · 848×1264 | model trả 1184×896 / 896×1184 / 1248×832 / 832×1248: **pipeline làm tròn xuống bội số của 32**; gateway từ chối job thay vì giao ảnh sai cỡ | — |

LoRA (tổng hợp, chỉ để kiểm đường nạp — không phải một style thật): `synthetic-a` rank 8 trên `to_q/to_k/to_v/to_out.0` của 32 khối.
`scale` 0,5 / 1,0 / 2,0 cho độ lệch trung bình so với ảnh gốc 10 / 11 / 13 trên thang 255 (ảnh không rỗng, không NaN);
**job không LoRA chạy ngay sau đó cho đúng sha256 của ảnh gốc** (`625baaeab50e2cc6`) ⇒ không để lại dấu vết giữa các job.
Chất lượng với LoRA thật (huấn luyện): xem §3.2.

### 3.2 LoRA thật qua gateway (✅ 2026-11-23, `int8`, 1024², 40 bước)

Hai file thật do chủ sản phẩm đưa (`reference/Qwen-Image-2.1_LoRA/`, không nằm trong git): `natural-skin` (ModelScope, rank 16,
448 tensor, tên lớp không tiền tố, có `.default.`) và `restore` (ai-toolkit 0.13.23, rank 32, 384 tensor, tiền tố `diffusion_model.`).

| Phép thử | Kết quả |
|---|---|
| `natural-skin` 0,8, prompt chân dung, seed 21 | ảnh khác hẳn ảnh không LoRA (khung gần hơn, da có lỗ chân lông) ⇒ LoRA có tác dụng; xem bằng mắt, **chưa có thước đo định lượng về "đẹp hơn"** |
| `restore` 1,0 trên ảnh bị làm mờ + JPEG q12 (`image.edit`) | độ lệch trung bình bật/tắt LoRA 2,85/255; so với ảnh gốc: tắt 10,34, bật 9,73. Mô hình gốc đã tự khôi phục khá tốt nên tác dụng **nhỏ**, một ảnh một seed — không đủ để nói "tốt hơn" |

**Lỗi tìm ra (R24):** lần chạy đầu, peft chỉ ghi một dòng log `Loading adapter weights ... unexpected keys ... img_mlp.gate_up` và bỏ **64 trên
384 tensor** của `restore`. Nguyên nhân: ai-toolkit gộp hai lớp thành `img_mlp.gate_up` (hàng của B = 2 × 12288), còn model
này có hai lớp `gate_layer` và `proj` (`transformer_qwenimage21.py:206-212`); job vẫn `succeeded`, LoRA chạy với 5/6 lớp, không báo lỗi. Sửa tại gốc:
worker đọc file, bỏ tiền tố/`.default`, gấp `alpha`, tách `gate_up` thành `gate_layer` (nửa đầu) + `proj` (nửa sau) rồi **từ chối** cả LoRA nếu
còn tensor nhắm vào lớp không có (`workers/qwen_image/worker.py::lora_state_for`). Thứ tự nửa đầu/nửa sau là **trích dẫn từ mã của một dự
án khác** (một bộ nạp LoRA mã nguồn mở độc lập cho Qwen-Image: "first half of the rows ... `gate_layer`, second half `proj`"), không phải từ tài liệu
upstream; bằng chứng độc lập duy nhất ở đây là sau sửa tác dụng bật/tắt tăng từ 1,85 lên 2,85 và log không còn dòng cảnh báo — chưa có phép thử
nào phân biệt thứ tự đúng với thứ tự đảo. `natural-skin` cho **đúng cùng sha256 ảnh** trước và sau sửa (`b699c70c77c4`) ⇒ đường mới không đổi kết quả của file đã chạy đúng.
Cổng chống tái phát: `tests/test_lora_state.py` (4 test, đã sabotage-verify: đảo nửa, bỏ `alpha`, bỏ kiểm lớp lạ ⇒ đúng test đó đỏ).

Hệ quả cho manifest: bảng cỡ của Qwen có hai hạng — `1k` (~1 MP cùng tỉ lệ, nhanh; ✅ cả 7 cỡ đã sinh ảnh thật) và `native`
(cỡ gốc ~4 MP, chậm khi VRAM < ~27 GiB). Ảnh Qwen trả về luôn là PNG RGBA (cả khi không yêu cầu trong suốt).
Cloudflare Tunnel (giới hạn thân request, timeout): **⬜ chưa đo** — cần tài khoản Cloudflare thật, xem `docs/product/06_deployment.md`.

Model không chạy được trên GPU nào trong bảng ⇒ manifest báo `unsupported_hardware` trên máy đó, kèm phần cứng tối
thiểu đã đo được ở nơi khác (nếu có).

### 3.3 MiniMax H3 qua gateway thật (✅ 2026-11-25, RTX 3090 Ti, `mixed`, seed 42, `short_edge` 352)

Chạy bằng các kịch bản `aha-open-mcp` `real_video_h3_preview` và `cancel_real_video_h3` trên một `aha-open serve` thật với worker thật.

| Phép kiểm | Kết quả |
|---|---|
| Cả job, từ lúc gửi tới khi có tệp tải được | 283 s (model nạp cho job này, không có gì ở lại GPU sau đó: 950 MiB do desktop dùng) |
| Tệp tải về, đọc từ chính các hộp mp4 của nó | 608x352, 124 khung, 24 fps, âm thanh stereo 32 kHz, chú thích `AI-generated (MiniMax H3)`; kích thước, fps, thời lượng, âm thanh, sha256 và số byte đều bằng điều job đã báo (19/19 phép kiểm) |
| Huỷ, 2 s sau khi job thành `running` | `cancelled`, không có kết quả (`/result` là HTTP 409); GPU được giải phóng sau đó |

Bộ nhớ của worker, đo sau đó từ chính các dòng giai đoạn của worker (§2.4): RSS đỉnh 2,2 GiB (model chữ) và 1,7 GiB (model tham chiếu) trong khi RAM trống của máy luôn trên 18 GiB.

**Chưa đo:** huỷ giữa lúc khử nhiễu ở kích thước native qua gateway (lần huỷ này rơi vào lúc model đang nạp; huỷ theo từng bước đã có test của worker, M2), các thời lượng hay tỉ lệ khác, đầu vào khung đầu/cuối.

## 4. Thêm một model mới

1. Đo theo khuôn bảng §3 trên ít nhất một GPU.
2. Đọc giấy phép, trích nguyên văn các điều về thương mại vào mục mới ở file này.
3. Tạo `workers/<tên>/` với `pyproject` ghim đúng upstream.
4. Khai khả năng trong manifest đúng những gì đã đo; chạy bộ kiểm hợp chuẩn.

### Các biến thể ConvRot (tóm tắt; bản đầy đủ có trích dẫn nguyên văn và số đo ở `docs/product/05_models.md` §1.1 và §3.4)

- **Nguồn (ghim revision, kích thước, sha256):** cấu hình/tokenizer/VAE từ `Qwen/Qwen-Image-2.1`; text encoder INT8 ConvRot từ `Comfy-Org/Qwen-Image-2.1`; transformer từ `JoaoZaokk/ConvRot-Mixed-INT8-for-Qwen-Image-2.1`.
- **Giấy phép:** nhà phát hành nêu: *Non-commercial use only (Qwen Research License)*; các file là bản đã sửa (lượng tử hoá), không phải model gốc. Lệnh cài in lại NOTICE của họ; khoá giấy phép và `commercial_use: not_allowed` không đổi.
- **Chất lượng** là lời của nhà phát hành (một người chấm, ba vòng mù, mỗi vòng 24 cảnh) — AHA open **không** lặp lại bài chấm mù đó; trang web hiện các ghi chú ấy cạnh từng biến thể.
- **Cách chạy:** xoay *activation* thay vì xoay lại trọng số, lượng tử theo hàng rồi nhân qua `torch._int_mm` với trọng số truyền dạng *view* cột-chính (`weight.t()`) — bố cục này nhanh gấp 3,4 lần bản copy liên tục (1,74 ms so với 5,93 ms); GPU không có lõi INT8 (< sm75) giải mã sang bf16. Luôn dùng offload cấp model: đỉnh VRAM **8,17 GiB** cho cả ba bản (đặt tất cả lên GPU thì 19,7–22,1 GiB và bản lớn tràn sang RAM).
- **Đo qua gateway thật (RTX 3090 Ti, seed 42):** int8-convrot `1db93465…`, mixed-p010 `0f8d6882…`, mixed-p015 `bdb17b3c…`, int8 `625baaea…` — trùng bit với phép đo trực tiếp. LoRA `natural-skin` đổi ảnh rõ rệt và job không-LoRA chạy sau đó bit-giống job trước (trên int8-convrot và mixed-p015). Thứ tự hai nửa của `img_mlp.gate_up` (`gate_layer` rồi `proj`) đã được chứng minh từ trọng số (cosine 1,007 so với 0,000 chéo), đóng mục R24.
- **Chưa làm (⬜):** bài chấm mù của nhà phát hành chưa lặp lại; chưa so định lượng chất lượng ConvRot với `int8`; chưa tối ưu RAM (đỉnh 22–23 GiB); chưa thử GPU khác RTX 3090 Ti.
