# Kiến trúc AHA open

> Trạng thái: gateway (P2/P3) đã cài đặt và qua bộ kiểm hợp chuẩn với worker giả (🧪, xem `conformance/`);
> worker Qwen (P4) và giao diện `/ui/` (P5b) xem [05_models.md](05_models.md) §3.
> Gateway ↔ worker dùng `subprocess.Popen` + thread đọc (không dùng asyncio subprocess), plan R15.

## 1. Hai loại tiến trình

```text
                 ┌──────────────────────────── aha-open gateway ────────────────────────────┐
client ──/v1──►  │ HTTP (FastAPI/uvicorn)  ·  auth/token  ·  uploads  ·  job store (SQLite)  │
                 │ hàng đợi theo GPU (công bằng theo client)  ·  quản lý cư trú model         │
                 └──────────────┬───────────────────────────────────────┬────────────────────┘
                                │ stdio JSON-lines + thư mục job          │
                 ┌──────────────▼─────────────┐           ┌──────────────▼─────────────┐
                 │ worker qwen-image           │           │ worker minimax-h3           │
                 │ venv riêng (Py 3.13, torch, │           │ venv riêng (torch, diffusers│
                 │ diffusers @80c7ed2)         │           │ @f53d552)                   │
                 └─────────────────────────────┘           └─────────────────────────────┘
```

| Tiến trình | Có torch? | Vai trò |
|---|---|---|
| **gateway** | **không** | Toàn bộ API `/v1`, xác thực, lưu trữ, hàng đợi, quyết định model nào đang trên GPU. Khởi động nhanh, luôn trả lời `/v1/health`. |
| **worker** (một cho mỗi họ model) | có | Nạp đúng một pipeline, chạy từng job, báo tiến độ, dừng ở ranh giới bước khi được huỷ. |

Vì sao tách: hai model ghim hai phiên bản diffusers/transformers khác nhau (đã đo, xem plan R2) — một môi trường chung
sẽ làm hỏng một trong hai. Worker cũng là đơn vị có thể chết (OOM, lỗi CUDA) mà không kéo gateway theo.

## 2. Giao tiếp gateway ↔ worker

Gateway khởi động worker như tiến trình con và nói chuyện qua **stdin/stdout, mỗi dòng một JSON** (không mở cổng mạng
nào ⇒ không có bề mặt tấn công mạng thêm). Pixel/video **không** đi qua pipe: gateway đặt input vào thư mục job, worker
ghi output vào đó.

```text
gateway → worker: {"op": "load", "model": "qwen-image-2.1", "variant": "int8"}
worker  → gateway: {"event": "loaded", "vram_peak_mib": 12380}
gateway → worker: {"op": "run", "job_id": "job_…", "dir": "<data>/jobs/job_…", "task": "image.generate", "params": {…}}
worker  → gateway: {"event": "progress", "job_id": "job_…", "step": 12, "total_steps": 40}
gateway → worker: {"op": "cancel", "job_id": "job_…"}
worker  → gateway: {"event": "done", "job_id": "job_…", "outputs": [{"file": "out_0.png", "width": 2048, "height": 2048}]}
                   {"event": "failed", "job_id": "job_…", "code": "gpu_out_of_memory", "message": "…"}
gateway → worker: {"op": "unload"}
```

Huỷ: worker kiểm cờ huỷ trong callback từng bước của diffusers (`callback_on_step_end`) và dừng ở bước kế tiếp.
Worker không phản hồi sau một thời gian ⇒ gateway kết thúc tiến trình và đánh dấu job `failed`; một kết quả xuất hiện
sau khi job đã `cancelled` bị bỏ.

## 3. Quản lý cư trú model và hàng đợi

- Mỗi GPU một hàng đợi. Tại một thời điểm, mỗi GPU chỉ có **một** model nằm trong VRAM.
- Chọn job kế tiếp: vòng tròn giữa các client (công bằng), trong mỗi client theo thứ tự gửi. Ưu tiên job dùng model
  đang nạp sẵn khi không làm một client chờ quá `max_wait_for_swap_s` (giảm số lần đổi model tốn hàng chục giây).
- Đổi model: `unload` worker cũ → `load` worker mới; job ở trạng thái `loading_model` trong lúc đó.
- Giới hạn: `max_queue_per_client`, `max_queue_total`; vượt ⇒ `429 busy`.

## 4. Lưu trữ

```text
<data_dir>/
  config.toml           cấu hình (hồ sơ mạng, cổng, model bật, GPU)
  clients.sqlite        client, token (chỉ lưu hash), trạng thái ghép cặp
  jobs.sqlite           job, trạng thái, metadata kết quả
  uploads/<upload_id>/  khúc + file đã ghép, xoá sau khi job dùng xong hoặc hết hạn
  jobs/<job_id>/        input (liên kết tới upload), output, log của job; xoá sau result_ttl_s
  models/               trọng số tải từ Hugging Face (người vận hành tải, đã đồng ý giấy phép)
  loras/                <id>.safetensors [+ <id>.json] do người vận hành đặt (`aha-open loras add`); chỉ safetensors, id a-z0-9_-
  tls/                  chứng chỉ tự ký cho hồ sơ lan
```

Token chỉ lưu dạng hash; giá trị token hiện đúng một lần lúc tạo/duyệt.

## 5. Cấu hình (ví dụ)

```toml
profile = "local"            # local | lan | internet
listen = "127.0.0.1:8890"    # lan: phải chỉ rõ interface, vd "192.168.1.20:8890"

[auth]
require_token = true
enroll = true                # internet: mặc định false

[[models]]
id = "qwen-image-2.1"
worker = "qwen_image"
variant = "int8"             # int8 | bf16
gpu = 0

[[models]]
id = "minimax-h3-fl2va"
worker = "minimax_h3"
gpu = 0
enabled = false              # bật sau khi P0 xác nhận phần cứng
```

## 6. CLI dự kiến

| Lệnh | Việc |
|---|---|
| `aha-open init --profile local` | Tạo `data_dir`, cấu hình mặc định |
| `aha-open models install qwen-image-2.1 --variant int8 --accept-license qwen-research` | Tạo venv worker, tải trọng số; không có `--accept-license` thì từ chối |
| `aha-open models list` | Model, trạng thái, giấy phép |
| `aha-open serve` | Chạy gateway |
| `aha-open clients create --name "Laptop An"` | Tạo token thủ công (in một lần) |
| `aha-open clients approve <code>` / `revoke <id>` | Duyệt ghép cặp / thu hồi |
| `aha-open tls fingerprint` | In fingerprint chứng chỉ (hồ sơ lan) |
| `aha-open doctor` | Kiểm GPU, CUDA, dung lượng, cổng, cấu hình |

## 7. Bố cục mã dự kiến

```text
aha_open/            gateway (không import torch — có test chặn)
  api/  auth/  queue/  storage/  workers/  cli.py
workers/
  qwen_image/        pyproject riêng, adapter gọi diffusers
  minimax_h3/        pyproject riêng
  fake/              worker giả cho test (không torch)
conformance/         bộ kiểm hợp chuẩn /v1 (pytest, gọi HTTP thật)
schema/              OpenAPI
docs/
```

## 8. Bất biến

1. Gateway không import torch (test chạy trong tiến trình con, kiểm `sys.modules`).
2. Không có route nào ngoài `/v1/*`, tệp tĩnh `/ui/*` (giao diện web — chỉ gọi `/v1` như mọi client, plan R14) và `/admin/*` nếu có trang quản trị (chỉ trên loopback).
3. Không có pixel/prompt nào ra khỏi máy ngoài phản hồi cho chính client đã gửi.
4. Kích thước output đúng như `params` đã được chấp nhận; không resize ngầm.
5. Một job đã `cancelled` không bao giờ có kết quả.
