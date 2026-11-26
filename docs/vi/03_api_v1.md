# Đặc tả API `/v1` (bản nháp)

> Chuẩn này **trung lập**: dịch vụ nào cũng có thể cài đặt nó; AHA open là một cài đặt. Khi có
> `schema/aha_provider_v1.openapi.yaml` (giai đoạn P1), file OpenAPI là nguồn sự thật máy đọc được.
>
> **Bản gốc duy nhất của chuẩn `/v1` nằm ở repo này**. Client — kể cả AHA Studio — chỉ trỏ sang đây và
> ghim phiên bản đặc tả mà chúng hỗ trợ; không giữ bản sao riêng.

## 1. Quy ước chung

- JSON UTF-8; thời gian ISO 8601 UTC (`2026-11-22T10:00:00Z`).
- Xác thực: `Authorization: Bearer <client_token>` — bắt buộc ở hồ sơ `lan`/`internet`; ở `local` tuỳ cấu hình (mặc định bật).
- Hồ sơ `internet` thêm hai header của Cloudflare Access: `CF-Access-Client-Id`, `CF-Access-Client-Secret`
  (Cloudflare kiểm ở edge; AHA open không cần đọc chúng).
- Mọi lỗi: HTTP status + `{"error": {"code": "<mã>", "message": "<câu cho người>", "retry_after_s": <số|null>}}`.
- Request mang header `Origin` của trình duyệt bị từ chối `403`, trừ khi người vận hành cho phép tường minh.
- Client phải bỏ qua trường không biết (để thêm trường không phá tương thích).
- Chỉ `GET /v1/health`, `POST /v1/enroll` và `GET /v1/enroll/{id}` được gọi không cần token. Mọi route khác —
  kể cả `GET /v1/manifest` — trả `401 unauthorized` nếu không có token hợp lệ.
- Khi `retry_after_s` là một số, response còn mang header HTTP `Retry-After` cùng giá trị tính bằng giây nguyên, làm tròn lên.
- Chỉ các route dưới `/v1` thuộc chuẩn này. Một server có thể có đường khác (trang web riêng của bản cài đặt, `/admin/*` chỉ dành cho
  người vận hành); client không bao giờ được phụ thuộc vào chúng, và server có thể trả `404` cho chúng.

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

Quy tắc upload (client vi phạm một quy tắc sẽ nhận `400 invalid_request` hoặc `400 upload_incomplete`):

- Khúc được đánh số **từ 0**. Một upload `bytes` byte có `max(1, ceil(bytes / part_size))` khúc, đánh số `0` tới `parts - 1`; số khác bị từ chối.
- Mỗi khúc phải dài **đúng** `part_size` byte, trừ khúc cuối dài đúng phần còn lại (`bytes - part_size × (parts - 1)`). Khúc ngắn hơn hay dài hơn bị từ chối.
- Một khúc có thể gửi lại trước `complete`; byte mới thay byte cũ. Sau `complete`, gửi khúc bị từ chối.
- `complete` khi thiếu khúc, hoặc với file có kích thước hay sha256 khác giá trị đã khai, trả `400 upload_incomplete`; trường hợp sai sha256 thì server
  huỷ upload, client bắt đầu lại bằng `POST /v1/uploads` mới. `complete` trên một upload đã hoàn tất trả `200` lần nữa.
- Một upload chỉ dùng được bởi token đã tạo nó (token khác nhận `404`). Upload bị xoá sau một khoảng thời gian do server đặt tính từ lúc tạo,
  dù đã có job dùng hay chưa (cài đặt của người vận hành; không có trong manifest).
- Job đã được nhận thì giữ nguyên đầu vào của nó: xoá upload không bao giờ lấy mất đầu vào của job đang xếp hàng hoặc đang chạy.

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
- Model có cung cấp biến thể (trường này là chung; Qwen-Image-2.1 chỉ có `int8`) khai thêm: `variant` (biến thể đang chạy ở server này) và
  `variants` = `{ "<tên>": {"label", "disk_gib", "vram_gib": {"1k", "native?"}, "ram_gib", "seconds": {"1k"}, "loras", "quality_note", "installed", "fit", "selected", "recommended"} }` (đều cộng thêm). Qwen-Image-2.1 có bốn biến thể: `int8`, `int8-convrot`, `mixed-p010`, `mixed-p015`. `vram_gib`, `ram_gib`, `seconds` là số đã đo (`null` = chưa đo; số chưa đo không bao giờ từ chối gì). `installed` = trọng số của biến thể đã đủ trên đĩa server; `fit` ∈ `fits` | `tight` | `too_small` | `unknown`, tính từ đỉnh đo ở 1024² so với GPU của server; `loras` = `true` khi đã chạy LoRA thật trên biến thể đó, `null` = chưa thử; `quality_note` = một câu về chất lượng, trích đánh giá của chính nhà phát hành.
  `recommended` là gợi ý theo phần cứng server (chỉ để hiển thị; server không tự đổi biến thể). `vram_gib` là số đo
  thật (`null` = chưa đo).
- Khả năng video: `durations_s` (danh sách hoặc `{min, max}`), `fps`, `short_edges`, `aspect_ratios`,
  `audio` (`{channels, sample_rate}` hoặc `null`), `first_frame`, `last_frame`, `max_reference` (`{image, video, audio, total}`).
  Một model video còn có thể khai thêm (cộng thêm): `sizes` = `[{"aspect_ratio", "short_edge", "width", "height", "tier"}]`, các kích thước đầu ra
  chính xác (chỉ các cặp đã sinh thật; cặp không có trong danh sách bị từ chối), `frame_grid` = `{"step", "base"}` (model tạo `base + n * step` khung:
  thời lượng được làm tròn **lên** tới số khung kế tiếp như vậy, vd 5 s ở 24 fps của MiniMax H3 là 124 khung = 5,17 s) và `steps`
  (`{min, max, default}` như với ảnh).
  **MiniMax H3** (`minimax-h3-fl2va`) chỉ khai những gì đã chạy trên máy tham chiếu: `durations_s` `{min: 5, max: 5}`, `aspect_ratios` `["16:9"]`,
  kích thước 1344×768 (`native`, `short_edge` 768) và 608×352 (`preview`, `short_edge` 352), `steps` 20, `first_frame: false`, `last_frame: false`
  (request có khung hình bị `400`), âm thanh 2 kênh 32 kHz. `variants.mixed.vram_gib` của nó cho các đỉnh đã đo (`preview` 17,2, `native` 19,6).
  Một clip mất vài phút (khoảng 5 phút ở `preview`, 22 phút ở `native` trên RTX 3090 Ti): luôn tạo job rồi thăm dò. `license.required_ui_label` là
  `"MiniMax H3"` và phải được hiển thị.
  **MiniMax H3 tham chiếu** (`minimax-h3-ref2va`, một model riêng) có cùng độ dài clip, số bước, âm thanh và giấy phép, chỉ cung cấp **cỡ `preview`
  (608×352, `short_edge` 352)** (cỡ native 1344×768 chưa chạy với tham chiếu), và khai `reference_required: true` (request không có ảnh hay video tham chiếu bị
  `400`; dùng `minimax-h3-fl2va` cho trường hợp chỉ có chữ) cùng `max_reference` `{image: 1, video: 0, audio: 0, total: 1}`: mới chạy thật một ảnh
  tham chiếu nên chỉ nhận chừng đó (ảnh thứ hai, video hay audio tham chiếu bị `400`). Nó được cài vào thư mục của `minimax-h3-fl2va` (hai model dùng chung
  bộ mã hoá và các VAE), nên báo `not_installed` cho tới khi có transformer riêng của nó (14 GiB).
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
- Tham chiếu cho video: gửi các input `reference` (ảnh), `reference_video` (video) và `reference_audio` (audio) **theo thứ tự model nên đọc**: thứ tự
  đặt nhãn cho chúng trong prompt (`<Picture 1>`, `<Video 1>`, `<Audio 1>`) và quy định đồng hồ âm thanh/hình, nên đổi thứ tự là một request khác. Upload
  phải có loại media đúng với vai trò (`image/*`, `video/*`, `audio/*`; nếu không là `400`). Tham chiếu audio không được là tham chiếu duy nhất.
  Server hay worker không bao giờ tải gì từ URL: tham chiếu luôn là một upload.
- `params.loras` (nằm trong `params`, như `seed`): `[{"id": "<id trong manifest>", "scale": 0.8}]` (≤ `capabilities.loras.max`, `scale` mặc định 1.0, không trùng
  id). Id lạ, `scale` ngoài khoảng, trường thừa (vd `path`) ⇒ `400 invalid_request` trước khi xếp hàng. Job không có
  `loras` cho kết quả **y hệt** như khi server chưa từng nạp LoRA nào.
- Video: `params` gồm `aspect_ratio`, `duration_s`, `short_edge`, `seed`, và `steps` (chỉ với model khai `capabilities.steps`; vắng nghĩa là mặc
  định của nó). Model có `capabilities.sizes` nhận `width`, `height` và `num_frames` chính xác từ bảng đó; client không bao giờ gửi `width` hay `height`
  tự do cho video (`unknown params`).
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
video thêm `duration_s`, `fps`, `has_audio`, và với model có `capabilities.sizes` thì thêm `width` và `height` (server cho job `failed` nếu worker báo
cỡ khác cỡ đã chấp nhận, hoặc không báo cỡ, đúng như với ảnh).

- Job chỉ đọc/huỷ được bởi đúng token đã tạo nó; token khác nhận `404` (không lộ sự tồn tại).
- `POST /v1/jobs/{id}/cancel` trả `200` kèm job. Server chờ một khoảng ngắn, có giới hạn (vài giây) cho job dừng rồi mới trả lời, nên thân thường là
  `cancelled`; nếu job chưa dừng thì nó được trả ở trạng thái hiện tại, và client tiếp tục thăm dò `GET /v1/jobs/{id}` cho tới khi trạng thái cuối cùng.
  Huỷ một job đã ở trạng thái cuối không đổi gì và trả `200` kèm job đó.
- Huỷ có hiệu lực ở ranh giới bước kế tiếp; sau `cancelled` không bao giờ có kết quả.
- `429 busy` không chỉ là `limits.max_queue_per_client`. Server còn có thể từ chối vì cả hàng đợi đầy, vì client tạo quá nhiều job trong phút vừa
  rồi, hoặc vì quá nhiều upload đang được lưu cho client. Các con số đó là cấu hình của người vận hành và **không** có trong manifest. Client chờ
  `retry_after_s` (và header `Retry-After`) rồi thử lại; không bao giờ thử lại dồn dập.
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
