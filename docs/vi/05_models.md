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
| Trên RTX 3090 Ti 24 GB | ✅ đã đo, xem §3: INT8 1024² ≈ 40 s / 9,5 GiB; BF16 1024² chạy được (16,4 GiB) nhưng chậm; cỡ gốc 2048² tràn VRAM | §3 |
| Biến thể AHA open cho chọn | `int8` (máy yếu, ~18 GB đĩa) và `bf16` (bản đầy đủ, ~33 GB đĩa, máy mạnh ≥ 40 GiB VRAM — ngưỡng này là suy luận, chưa đo). `aha-open models install` gợi ý theo GPU và nói lý do | `aha_open/hardware.py` |
| Cỡ ảnh AHA open khai | 14 cỡ: 7 cỡ gốc (`tier: native`) + 7 cỡ ~1 MP (`tier: 1k`, mọi cạnh chia hết cho **32** — pipeline làm tròn xuống bội của 32: yêu cầu 1200×896 ra 1184×896). Chỉ cho `aspect_ratio` ⇒ hạng `1k` | plan R20 |
| LoRA | Người vận hành đặt file `.safetensors` vào `<data_dir>/loras/` (`aha-open loras add FILE`); client chọn theo id + `scale` (≤ 3 cùng lúc, 0 < scale ≤ 2). ✅ chạy thật trên `int8` với hai LoRA thật (xem §3.2); LoRA có lớp lạ bị **từ chối**, không áp dụng một phần; `bf16`: chưa thử, không bị từ chối | plan R23 |
| `mask` | **Không khai** (`mask: false`): README chỉ nói "separate masks", không có tham số nào; chưa có phép chạy nào cho thấy nó hoạt động ra sao | plan R21 |

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
Client như AHA Studio chỉ hiện `commercial_use: not_allowed` từ manifest, không có logic riêng cho Qwen (plan R12, Q6).

Manifest: `"license": {"name": "Qwen RESEARCH LICENSE", "commercial_use": "not_allowed"}`.

## 2. MiniMax H3 (video có âm thanh)

| Mục | Giá trị | Nguồn |
|---|---|---|
| Loại | Model sinh video + âm thanh đa phương thức | README chính thức |
| Thời lượng | 4–15 giây | README |
| Khung hình | 24 fps | README |
| Độ phân giải | Cạnh ngắn mặc định 768 px; 2K qua bước "H3-Regenerate-2K" (chạy lại chính model) | README |
| Tỉ lệ | Nhiều, gồm 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 | README |
| Âm thanh | Stereo 32 kHz; lời thoại ổn định ở 11 ngôn ngữ (không có tiếng Việt trong danh sách) | README |
| Checkpoint FL2VA | Text→video; 1 ảnh (khung đầu hoặc cuối); 2 ảnh (đầu + cuối) | README |
| Checkpoint Ref2VA | Tham chiếu: ≤ 9 ảnh, ≤ 3 video, ≤ 3 audio (mỗi clip 2–15 s, tổng ≤ 15 s), tối đa 12 file | README |
| Độ chính xác | BF16; Omni-Transformer "33B" | README, requirements |
| Phục vụ chính thức (ví dụ) | `sglang serve --num-gpus 4` | README |
| Môi trường | diffusers ghim commit `f53d552…`, `transformers>=4.57`, torch ≥ 2.4 | requirements |
| Mở rộng prompt "H3-Context-IR" | Gọi **API cloud của MiniMax** (cần token), không chạy local. AHA open **không** bật mặc định — gửi prompt ra bên thứ ba trái nguyên tắc dữ liệu | script `full-2k-t2va-h3-context-ir.sh` |
| Encoder | Qwen3-VL-32B (Apache-2.0) | ghi chú LICENSE |
| Trên RTX 3090 Ti 24 GB | **cần đo** — suy diễn: khó chạy nếu không offload/lượng tử mạnh (transformer 33B + encoder 32B) | |

Ánh xạ `/v1`: `minimax-h3-fl2va` (`video.generate`, `first_frame`, `last_frame`), `minimax-h3-ref2va`
(`video.generate`, `max_reference`). Kết quả: mp4 có âm thanh (`has_audio: true`).

### 2.1 Giấy phép — "MiniMax H3 COMMUNITY LICENSE AGREEMENT"

License date: August 2, 2026. Bên cấp phép: Nanonoble Pte. Ltd. Nguồn: `huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE`
(đọc 2026-11-22 qua công cụ web; câu dài bị cắt ở 125 ký tự, đánh dấu "(cắt)" — **đối chiếu lại toàn văn ở P0**).

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
| MiniMax H3 FL2VA | BF16 + offload | 4 s · 768p | cần đo | | | cần đo | **Hoãn** (chủ sản phẩm, 2026-11-23). Từ file index đã đọc: transformer 66,28 GB + encoder 66,71 GB ≈ 133 GB cho một checkpoint |

Đọc bảng: trên GPU 24 GB, **INT8 ở 1024² là lựa chọn thoải mái** (~40 s, 9,5 GiB). BF16 chạy được nhưng không nhanh hơn và
tốn gấp đôi đĩa; bản BF16 phù hợp GPU lớn (≥ 40 GiB) nơi không cần offload — **chưa đo trên GPU như vậy** (không có
phần cứng), nên ngưỡng 40 GiB ở `hardware.py` là suy luận từ kích thước trọng số (33 GB) + hoạt hoá, không phải số đo.

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
án khác** (một bộ nạp LoRA mã nguồn mở độc lập cho Qwen-Image: "first half of the rows ... `gate_layer`, second half `proj`"; tên dự án ghi ở plan R24), không phải từ tài liệu
upstream; bằng chứng độc lập duy nhất ở đây là sau sửa tác dụng bật/tắt tăng từ 1,85 lên 2,85 và log không còn dòng cảnh báo — chưa có phép thử
nào phân biệt thứ tự đúng với thứ tự đảo. `natural-skin` cho **đúng cùng sha256 ảnh** trước và sau sửa (`b699c70c77c4`) ⇒ đường mới không đổi kết quả của file đã chạy đúng.
Cổng chống tái phát: `tests/test_lora_state.py` (4 test, đã sabotage-verify: đảo nửa, bỏ `alpha`, bỏ kiểm lớp lạ ⇒ đúng test đó đỏ).

Hệ quả cho manifest (plan R20): bảng cỡ của Qwen có hai hạng — `1k` (~1 MP cùng tỉ lệ, nhanh; ✅ cả 7 cỡ đã sinh ảnh thật) và `native`
(cỡ gốc ~4 MP, chậm khi VRAM < ~27 GiB). Ảnh Qwen trả về luôn là PNG RGBA (cả khi không yêu cầu trong suốt).
Cloudflare Tunnel (giới hạn thân request, timeout): **⬜ chưa đo** — cần tài khoản Cloudflare thật, xem `docs/06_deployment.md`.

Model không chạy được trên GPU nào trong bảng ⇒ manifest báo `unsupported_hardware` trên máy đó, kèm phần cứng tối
thiểu đã đo được ở nơi khác (nếu có).

## 4. Thêm một model mới

1. Đo theo khuôn bảng §3 trên ít nhất một GPU.
2. Đọc giấy phép, trích nguyên văn các điều về thương mại vào mục mới ở file này.
3. Tạo `workers/<tên>/` với `pyproject` ghim đúng upstream.
4. Khai khả năng trong manifest đúng những gì đã đo; chạy bộ kiểm hợp chuẩn.
