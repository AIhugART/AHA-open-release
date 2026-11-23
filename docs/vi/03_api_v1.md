# Đặc tả API `/v1` (bản nháp)

> Chuẩn này **trung lập**: dịch vụ nào cũng có thể cài đặt nó; AHA open là một cài đặt. Khi có
> `schema/aha_provider_v1.openapi.yaml` (giai đoạn P1), file OpenAPI là nguồn sự thật máy đọc được.
>
> **Bản gốc duy nhất của chuẩn `/v1` nằm ở repo này** (plan R13). Client — kể cả AHA Studio — chỉ trỏ sang đây và
> ghim phiên bản đặc tả mà chúng hỗ trợ; không giữ bản sao riêng.

## 1. Quy ước chung

- JSON UTF-8; thời gian ISO 8601 UTC (`2026-11-22T10:00:00Z`).
- Xác thực: `Authorization: Bearer <client_token>` — bắt buộc ở hồ sơ `lan`/`internet`; ở `local` tuỳ cấu hình (mặc định bật).
- Hồ sơ `internet` thêm hai header của Cloudflare Access: `CF-Access-Client-Id`, `CF-Access-Client-Secret`
  (Cloudflare kiểm ở edge; AHA open không cần đọc chúng).
- Mọi lỗi: HTTP status + `{"error": {"code": "<mã>", "message": "<câu cho người>", "retry_after_s": <số|null>}}`.
- Request mang header `Origin` của trình duyệt bị từ chối `403`, trừ khi người vận hành cho phép tường minh.
- Client phải bỏ qua trường không biết (để thêm trường không phá tương thích).

## 2. Route

| Route | Mục đích |
|---|---|
| `GET /v1/health` | Sống/chết, trạng thái GPU, model đang nạp, độ dài hàng đợi |
| `GET /v1/manifest` | Model, task, khả năng, giấy phép, giới hạn, khả năng tuỳ chọn |
| `POST /v1/uploads` | Tạo upload (khai `bytes`, `sha256`, `media_type`) → `upload_id`, `part_size` |
| `PUT /v1/uploads/{id}/parts/{n}` | Gửi khúc thứ `n` (≤ `part_size`) |
| `POST /v1/uploads/{id}/complete` | Đóng upload; server kiểm sha256 |
| `POST /v1/jobs` | Tạo job → `202` + `job_id` ngay |
| `GET /v1/jobs/{id}` | Trạng thái, tiến độ, metadata kết quả |
| `POST /v1/jobs/{id}/cancel` | Huỷ |
| `GET /v1/jobs/{id}/result?index=N` | Tải kết quả theo luồng, hỗ trợ `Range` |
| `POST /v1/enroll` · `GET /v1/enroll/{id}` | Ghép cặp (tuỳ chọn, khai trong manifest) |

Tải lên theo khúc tồn tại vì input video/ảnh lớn có thể vượt giới hạn thân request của proxy (vd Cloudflare).

## 3. Health

```json
GET /v1/health
{"status": "ok", "api_version": "1.0",
 "gpus": [{"index": 0, "name": "…", "memory_total_mib": 24564, "active_model": "qwen-image-2.1"}],
 "queue": {"queued": 2, "running": 1}}
```

`status`: `ok` · `degraded` (một worker hỏng) · `starting`.

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
- Trường giấy phép tuỳ chọn (cộng thêm): `excluded_territories` (mã vùng, vd `["EU","GB","KR","US"]`),
  `required_ui_label` (chuỗi client **phải** hiện nguyên văn khi cho chọn model), `notes` (một câu cho người dùng).
  Client không hiểu trường nào thì vẫn hiện `name` + `commercial_use`.

Ví dụ MiniMax:

```json
"license": {"name": "MiniMax H3 Community License", "commercial_use": "allowed_with_conditions",
            "excluded_territories": ["EU", "GB", "KR", "US"], "required_ui_label": "MiniMax H3", "url": "https://…"}
```
- Mục `capabilities.sizes[]` có thể mang thêm `tier` (`"1k"` ≈ 1 MP, nhanh; `"native"` ≈ 4 MP, cỡ gốc của model, có thể
  cần nhiều VRAM hơn rất nhiều). Chỉ cho `aspect_ratio` thì server chọn **mục đầu tiên khớp** (hạng `1k`); muốn cỡ gốc thì
  gửi `width` + `height`. Chỉ các cỡ **đã sinh ảnh thật** mới được khai.
- Model hỗ trợ LoRA khai `capabilities.loras` = `{"max": 3, "scale": {"min", "max", "default"}, "format": "safetensors",
  "available": [{"id", "display_name", "default_scale", "bytes", "trigger?", "license?", "notes?"}]}`. `available` là các file
  **người vận hành đặt trên server** (client không tải LoRA lên qua `/v1`) và được đọc lại ở mỗi lần gọi manifest — thêm
  file không cần khởi động lại. Không bao giờ lộ đường dẫn.
- Model có nhiều biến thể (vd Qwen `bf16`/`int8`) khai thêm: `variant` (biến thể đang chạy ở server này) và
  `variants` = `{ "<tên>": {"label", "disk_gib", "vram_gib": {"1k", "native"}, "selected", "recommended"} }`.
  `recommended` là gợi ý theo phần cứng server (chỉ để hiển thị; server không tự đổi biến thể). `vram_gib` là số đo
  thật (`null` = chưa đo).
- Khả năng video: `durations_s` (danh sách hoặc `{min, max}`), `fps`, `short_edge`, `aspect_ratios`,
  `audio` (`{channels, sample_rate}` hoặc `null`), `first_frame`, `last_frame`, `max_reference` (`{image, video, audio, total}`).
- Task định nghĩa trong v1: `image.generate`, `image.edit`, `video.generate`, `audio.generate`, `speech.generate`.

## 5. Tạo job

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
- Ảnh: `params` gồm `width`+`height` (cùng nhau) hoặc `aspect_ratio`, `seed`, `steps`, `num_outputs`, và `transparent`
  (bool; chỉ model khai `rgba: true`; server thêm câu mô tả nền trong suốt vào prompt). `seed` vắng ⇒ server chọn một
  seed ngẫu nhiên và ghi lại trong `results[].seed`; `steps` vắng ⇒ dùng `capabilities.steps.default`.
- Model khai `mask: false` thì từ chối vai trò `mask`.
- `loras`: `[{"id": "<id trong manifest>", "scale": 0.8}]` (≤ `capabilities.loras.max`, `scale` mặc định 1.0, không trùng
  id). Id lạ, `scale` ngoài khoảng, trường thừa (vd `path`) ⇒ `400 invalid_request` trước khi xếp hàng. Job không có
  `loras` cho kết quả **y hệt** như khi server chưa từng nạp LoRA nào.
- Video: `params` gồm `aspect_ratio`, `duration_s`, `short_edge`, `seed`.
- `client_request_id` (tuỳ chọn): gửi lại cùng giá trị trong 10 phút ⇒ trả lại đúng job cũ, không chạy lần hai.
- Tham số ngoài `capabilities` ⇒ `400 invalid_request` **trước khi** xếp hàng; server không âm thầm kéo giãn hay đổi kích thước.

## 6. Vòng đời job

```text
queued → loading_model → running → succeeded
                                 ↘ failed
(mọi trạng thái chưa kết thúc) → cancelled
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

Khi `succeeded`, mỗi phần tử `results[]`:
`{"index": 0, "kind": "image", "media_type": "image/png", "width": 2048, "height": 2048, "bytes": 5123456, "sha256": "…", "seed": 42, "loras": [{"id": "ink", "scale": 0.8, "sha256": "…"}]}`
(server từ chối một kết quả PNG có cỡ khác cỡ đã chấp nhận — job `failed`, không bao giờ resize ngầm; ảnh Qwen là PNG
RGBA nguyên trạng model trả về);
video thêm `duration_s`, `fps`, `has_audio`.

- Job chỉ đọc/huỷ được bởi đúng token đã tạo nó; token khác nhận `404` (không lộ sự tồn tại).
- Huỷ có hiệu lực ở ranh giới bước kế tiếp; sau `cancelled` không bao giờ có kết quả.
- Kết quả giữ ít nhất `limits.result_ttl_s` rồi xoá.

## 7. Ghép cặp (tuỳ chọn)

```text
POST /v1/enroll        {"client_name": "Laptop An", "client_kind": "aha-studio"}
  → 201 {"id": "enr_…", "enroll_code": "K7Q4-2M", "expires_at": "…"}
GET /v1/enroll/enr_…   → {"status": "pending"}
                       | {"status": "approved", "token": "…"}   (token chỉ trả đúng một lần)
                       | {"status": "denied"} | {"status": "expired"}
```

## 8. Mã lỗi

| Mã | HTTP | Ý nghĩa |
|---|---|---|
| `invalid_request` | 400 | Thiếu/sai trường, tham số ngoài `capabilities` |
| `unsupported_task` | 400 | Model không làm task này |
| `upload_incomplete` | 400 | Upload chưa `complete` hoặc sai sha256 |
| `unauthorized` | 401 | Thiếu/sai token |
| `forbidden` | 403 | Không có quyền, `Origin` bị chặn |
| `not_found` | 404 | Không có job/upload (hoặc không thuộc token này) |
| `job_not_ready` | 409 | Tải kết quả của job chưa `succeeded` (đang chạy, thất bại hoặc đã huỷ) |
| `model_not_installed` | 409 | Model chưa cài |
| `unsupported_hardware` | 409 | Phần cứng không chạy được model |
| `busy` | 429 | Hàng đợi của client/server đầy; có `retry_after_s` |
| `model_loading` | 503 | Đang nạp model; có `retry_after_s` |
| `gpu_out_of_memory` | (trong job) | Hết VRAM |
| `user_cancelled` | (trong job) | Đã huỷ |
| `generation_failed` | (trong job) | Lỗi khác của pipeline |

## 9. Hồ sơ mạng (`profile` trong manifest)

| Hồ sơ | Lắng nghe | Giao thức | Xác thực |
|---|---|---|---|
| `local` | `127.0.0.1` | HTTP | token (mặc định bật) |
| `lan` | interface do người vận hành chọn | **HTTPS** | token riêng từng client |
| `internet` | `127.0.0.1` (sau Cloudflare Tunnel) | HTTPS ở edge | Cloudflare Access + token riêng từng client |

Client (vd AHA Studio) chỉ chấp nhận `http://` với `127.0.0.1`/`localhost`.

## 10. Quy tắc phiên bản

Thêm trường/route/task/mã lỗi/hồ sơ = **minor** (`1.x`). Đổi nghĩa hoặc xoá = **major** (`/v2`).
