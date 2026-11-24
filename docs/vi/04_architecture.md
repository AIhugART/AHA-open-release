# Kiến trúc AHA open

> Trạng thái: gateway (P2/P3) đã cài đặt và qua bộ kiểm hợp chuẩn với worker giả (🧪, xem `conformance/`);
> worker Qwen (P4) và giao diện `/ui/` (P5b) xem [plan](../../plan/2026-11-22_aha_open_plan.md) §3.
> Gateway ↔ worker dùng `subprocess.Popen` + thread đọc (không dùng asyncio subprocess).

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

Vì sao tách: hai model ghim hai phiên bản diffusers/transformers khác nhau (đã đo) — một môi trường chung
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
  tmp/                  LoRA mà trang web đang nhận; được đổi tên vào loras/ hoặc xoá ngay, không giữ lại
  tls/                  chứng chỉ tự ký cho hồ sơ lan
  logs/                 gateway.log (xoay vòng theo kích thước; cùng các dòng với terminal) và worker-<tên>.log; không bao giờ chứa token hay prompt
  run/gateway.json      do `aha-open serve` ghi: pid, started_at, version, source_digest (xoá khi thoát bình thường; nếu crash thì còn lại, mang một pid đã chết)
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
variant = "int8"             # int8 là biến thể duy nhất Qwen cung cấp
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
| `aha-open models install qwen-image-2.1 --accept-license qwen-research` | Tạo venv worker, tải trọng số; không có `--accept-license` thì từ chối |
| `aha-open models list` | Model, trạng thái, giấy phép |
| `aha-open serve` | Chạy gateway |
| `/ui/` ▸ *Download model* | Cùng việc `models install` ngay trên trang web (chỉ hồ sơ local), xem mục 6.1 |
| `aha-open clients create --name "Laptop An"` | Tạo token thủ công (in một lần) |
| `aha-open clients approve <code>` / `revoke <id>` | Duyệt ghép cặp / thu hồi |
| `aha-open tls fingerprint` | In fingerprint chứng chỉ (hồ sơ lan) |
| `aha-open doctor` | Kiểm GPU, CUDA, dung lượng, cổng, cấu hình |

### 6.1 Cài model từ trang web (`/admin/models/*`)

Dành cho người ngồi tại máy, cùng điều kiện với trang quản lý LoRA (hồ sơ local, người gọi và tên đều là loopback, có token; mọi trường
hợp khác trả `404`). **Không** thuộc `/v1`: cài model là quyết định của người vận hành, không phải của client.

| Yêu cầu | Trả lời |
|---|---|
| `GET /admin/models` | Các model cài được ở đây: `state`, giấy phép cần hiện (`license.key`, `name`, `commercial_use`, `notes`, `url`), `disk_gib`, `blocker` (`{code, message}` hoặc `null`) và trạng thái `install` hiện tại |
| `POST /admin/models/{id}/install` với `{"accept_license": "<license.key>"}` | `202` + trạng thái. Khoá phải đúng từng chữ; sai là `400` và không có gì chạy |
| `GET /admin/models/{id}/install` | `{state: idle \| running \| succeeded \| failed \| cancelled, phase, downloaded_bytes, expected_bytes, error, started_at, finished_at}` |
| `DELETE /admin/models/{id}/install` | `204`, giết cả cây tiến trình; lần sau tải tiếp được. `404` nếu không có gì đang chạy |

Cách làm: gateway chạy **một tiến trình con là chính lệnh `aha-open models install`** (một cách cài duy nhất), đọc đầu ra để biết giai
đoạn và số byte (tiến trình con, chạy với `AHA_OPEN_PROGRESS=lines`, in `AHAOPEN_PROGRESS <đã tải> <tổng>` lấy từ bộ đếm byte của chính trình tải; không
đo thư mục trọng số được vì `hf-xet` giữ dữ liệu trong bộ đệm riêng và chỉ ráp từng file ở cuối — quan sát trên lần tải 18 GB thật), và không chờ nó trong request. Khi tiến trình con thoát sạch, gateway đọc lại `config.toml` và
thay `cfg.models` tại chỗ; trạng thái model tính từ đĩa ở mỗi request nên manifest đổi mà không cần khởi động lại. Tắt gateway thì cài dở bị dừng.

Bị từ chối **trước khi chạy** (`409`), mỗi nguyên nhân một mã và một câu riêng: `unsupported_hardware`, `uv_missing`,
`download_support_missing`, `not_enough_disk`, `install_running`, `already_installed`. Sau khi chạy: `network_unreachable` hoặc `install_failed`
(dòng `error:` cuối, đường dẫn thư mục dữ liệu được thay bằng `<data folder>`). `uv` được tìm ở `AHA_OPEN_UV` (trình cài Windows đặt biến này), rồi `PATH`.

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
docs/  plan/
```

## 8. Bất biến

1. Gateway không import torch (test chạy trong tiến trình con, kiểm `sys.modules`).
2. Không có route nào ngoài `/v1/*`, tệp tĩnh `/ui/*` (giao diện web — chỉ gọi `/v1` và `/admin/*`), `/admin/loras*` (quản lý LoRA) và `/admin/models*` (cài model, mục 6.1) — `/admin/*` chỉ ở hồ sơ `local`, chỉ khi người gọi là loopback và hỏi bằng tên loopback, và vẫn cần token; mọi trường hợp khác trả `404`.
3. Không có pixel/prompt nào ra khỏi máy ngoài phản hồi cho chính client đã gửi.
4. Kích thước output đúng như `params` đã được chấp nhận; không resize ngầm.
5. Một job đã `cancelled` không bao giờ có kết quả.
