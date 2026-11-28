# Kiến trúc AHA open

> Trạng thái: gateway (P2/P3) đã cài đặt và qua bộ kiểm hợp chuẩn với worker giả (🧪, xem `conformance/`);
> worker Qwen (P4) và giao diện `/ui/` (P5b) xem [05_models.md](05_models.md) §3.
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
  uploads/<upload_id>/  khúc + file đã ghép, xoá khi hết hạn (upload_ttl_s tính từ lúc tạo)
  jobs/<job_id>/        hard link riêng của job tới đầu vào (chép file nếu ổ đĩa không hỗ trợ; gỡ khi job kết thúc), output;
                        xoá sau result_ttl_s
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
| `aha-open models install qwen-image-2.1 --variant int8-convrot ...` | Chọn biến thể (mặc định: biến thể tốt nhất vừa máy). Biến thể thứ hai chỉ được thêm cạnh biến thể đầu và không bao giờ chuyển biến thể đang dùng |
| `aha-open models list` | Model, trạng thái, giấy phép, và với mỗi model mọi biến thể: đã cài / đang dùng / chưa cài, đĩa, độ vừa GPU |
| `aha-open models use qwen-image-2.1 mixed-p015` | Chọn một biến thể đã cài (khởi động lại server để nạp) |
| `aha-open models remove qwen-image-2.1 mixed-p015 [--include-shared] [--yes]` | Xoá file của một biến thể; không có `--yes` chỉ báo sẽ xoá gì; phần dùng chung còn nguyên khi biến thể khác còn cần |
| `aha-open serve` | Chạy gateway |
| `/ui/` ▸ *Models* | Mọi biến thể của model đã cài: kích thước, VRAM và thời gian đã đo, độ vừa máy, ghi chú chất lượng; **Use this**, **Download**, **Delete** (hỏi trước và nói sẽ giải phóng bao nhiêu; phần dùng chung được đề nghị xoá riêng khi biến thể cuối đi) và **Cancel download** |
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
| `GET /admin/models` | Các model cài được ở đây: `state`, giấy phép cần hiện (`license.key`, `name`, `commercial_use`, `notes`, `url`, `excluded_territories`), `requires_territory_confirmation`, `disk_gib` (số GB **còn phải tải**), `blocker` (`{code, message}` hoặc `null`) và trạng thái `install` hiện tại. `engine_missing: true` nghĩa là có trọng số nhưng chưa có môi trường chạy: trang hiện nút *Set up engine*, tức lệnh `install` bên dưới với một biến thể đã cài (không tải lại file đã có; file được kiểm trước, pha `verifying`). |
| `POST /admin/models/{id}/install` với `{"accept_license": "<license.key>"}` (kèm `"confirm_territory": true` khi danh sách ghi `requires_territory_confirmation`) | `202` + trạng thái. Khoá phải đúng từng chữ, và `confirm_territory` phải là boolean `true` (chuỗi hay số không đủ); sai là `400` và không có gì chạy |
| `GET /admin/models/{id}/install` | `{state: idle \| running \| succeeded \| failed \| cancelled, phase, downloaded_bytes, expected_bytes, error, started_at, finished_at}` |
| `POST /admin/models/{id}/use` với `{"variant": "<tên>"}` | Chọn một biến thể đã cài; worker đang giữ biến thể cũ được nhả và job kế tiếp nạp biến thể mới. `409 variant_not_installed`, `409 model_busy` (một job của model đang chờ, nạp hoặc chạy), `409 install_running`; tên lạ là `400`. Phản hồi là mục của model |
| `DELETE /admin/models/{id}/variants/{variant}[?include_shared=true]` | Xoá file của một biến thể (mục của model + `message` nói đã giải phóng và giữ lại gì, và `shared_offer: true` khi biến thể cuối đã xoá mà phần dùng chung vẫn còn). Chỉ xoá trong thư mục models; phần biến thể khác còn cần được giữ, phần dùng chung chỉ đi cùng biến thể cuối **và** `include_shared=true`. `409 model_busy`/`install_running` như trên, `404` khi chưa cài, `400` khi tên lạ |
| `DELETE /admin/models/{id}/install` | `204`, giết cả cây tiến trình; lần sau tải tiếp được. `404` nếu không có gì đang chạy |

Model được đưa ra: Qwen-Image-2.1, MiniMax H3 (chữ/khung hình sang video) và MiniMax H3 (tham chiếu sang video). Trang vẽ mỗi model một thẻ. Giấy phép
loại trừ một số vùng (MiniMax H3: EU, GB, KR, US) có **một ô tick riêng thứ hai** — "I confirm that I am not in …" — và nút chỉ bật khi tick cả hai;
máy chủ cũng ép đúng luật đó, nên một script bỏ qua trang cũng bị từ chối. `--confirm-territory` của dòng lệnh là phần tương đương. `disk_gib` chỉ đếm các
tệp chưa có trên đĩa: model tham chiếu dùng chung thư mục với model chữ (bộ mã hoá và các VAE là cùng một tệp), nên khi đã cài model chữ nó chỉ đòi khoảng
15 GB, không phải 61. Biến thể và câu "GPU này quá nhỏ" lấy từ luật riêng của model (`recommend_variant_for`, cùng luật dòng lệnh dùng): card 12 GiB cài
được Qwen nhưng không cài được MiniMax H3 (cần 18 GiB).

Cách làm: gateway chạy **một tiến trình con là chính lệnh `aha-open models install`** (một cách cài duy nhất; với model có giới hạn vùng thì tiến trình
con nhận thêm `--confirm-territory`), đọc đầu ra để biết giai
đoạn và số byte (tiến trình con, chạy với `AHA_OPEN_PROGRESS=lines`, in `AHAOPEN_PROGRESS <đã tải> <tổng>` lấy từ bộ đếm byte của chính trình tải; không
đo thư mục trọng số được vì `hf-xet` giữ dữ liệu trong bộ đệm riêng và chỉ ráp từng file ở cuối — quan sát trên lần tải 18 GB thật), và không chờ nó trong request. Khi tiến trình con thoát sạch, gateway đọc lại `config.toml` và
thay `cfg.models` tại chỗ; trạng thái model tính từ đĩa ở mỗi request nên manifest đổi mà không cần khởi động lại. Tắt gateway thì cài dở bị dừng.

Bị từ chối **trước khi chạy** (`409`), mỗi nguyên nhân một mã và một câu riêng: `unsupported_hardware`, `uv_missing`,
`download_support_missing`, `not_enough_disk`, `install_running`, `already_installed`. Một lần tải bị cắt ngang (huỷ, khởi động lại, mất điện) **không** phải
`ready`: `model_index.json` nằm trong số các tệp đến đầu tiên, nên trạng thái còn tìm các tệp vẫn bị `huggingface_hub` đánh dấu `.incomplete` — trừ khi
`install.json`, chỉ được ghi sau khi một lần tải trả về thành công, nói rằng đã xong (một lần thử bị cắt trước đó để lại một `.incomplete` lạc mà không gì
xoá) — và bấm *Download* lần nữa sẽ tiếp tục (`install_model` tải lại mỗi khi lần tải chưa xong — trước đây nó bỏ qua việc tải ngay khi
`model_index.json` tồn tại). Sau khi chạy: `network_unreachable` (đầu ra cho thấy không với tới mạng) hoặc `install_failed`
(dòng `error:` cuối, đường dẫn thư mục dữ liệu được thay bằng `<data folder>`). `uv` được tìm ở `AHA_OPEN_UV` (trình cài Windows đặt biến này), rồi `PATH`.

### 6.2 Worker MiniMax H3 (đã xây và chạy thật: `minimax-h3-fl2va` và `minimax-h3-ref2va`, [05_models.md](05_models.md) §2.3, §2.4)

Các tệp MiniMax H3 duy nhất AHA open dùng là các bản lượng tử hoá lại của cộng đồng, ở định dạng một-tệp của ComfyUI (`convrot` INT4/INT8,
[05_models.md](05_models.md) §2.0). diffusers có model H3, các VAE, scheduler và các khối pipeline (Apache-2.0) nhưng không đọc được các tệp này, còn ComfyUI,
đọc được, là GPL-3.0 và là cả một ứng dụng. Vì vậy worker mang một bộ nạp nhỏ của riêng nó (`workers/minimax_h3/`), viết từ phần đầu của các tệp, từ diffusers và
từ thư viện kernel Apache-2.0 `comfy-kitchen`:

| Tệp | Làm gì |
|---|---|
| `lenient_safetensors.py` | đọc tensor theo offset trong phần đầu tệp; các tệp cộng đồng mang một đuôi 61–83 byte mà gói `safetensors` từ chối |
| `convrot.py` | các lớp tuyến tính trên trọng số INT8 / INT4 đóng gói đã xoay theo mỗi 256 đầu vào; trọng số được giải mã sang bf16 ngay trước mỗi phép nhân (nhanh hơn phép nhân INT8 trên GPU không có cuBLASLt, và gần model chưa lượng tử hơn); và trên trọng số NVFP4 (FP4 hai giá trị mỗi byte, scale FP8 mỗi 16 đầu vào theo layout khối của cuBLAS, một scale cho cả tensor) nên chạy được trên GPU không phải Blackwell |
| `h3_models.py` | dựng transformer (ánh xạ tên khoá, bảng thời gian "pruned" đã nướng sẵn, feed-forward chia lát), bộ mã hoá văn bản 50 lớp mà các lớp được đọc từ đĩa khi gọi (nó còn nạp tháp thị giác, chỉ cho job có tham chiếu; attention của nó lặp các head key/value trước, vì đường grouped-query của PyTorch không có kernel tiết kiệm bộ nhớ trên Windows và sẽ dựng ma trận điểm 10 GiB), và cả hai VAE; mỗi hàm dựng từ chối một tệp thừa hay thiếu tensor |
| `h3_references.py` | đổi `inputs` có thứ tự của request (`reference`, `reference_video`, `reference_audio`) thành các đối tượng tham chiếu của diffusers, đúng thứ tự đã gửi; từ chối URL, vai trò lạ và một tham chiếu audio đứng một mình |
| `h3_runner.py` | chạy pipeline diffusers theo ba giai đoạn (mã hoá, khử nhiễu, giải mã) để mỗi lúc chỉ một phần lớn nằm trên GPU (job có tham chiếu thêm một giai đoạn đổi chúng thành latent qua cả hai VAE trước khi transformer tới); kiểm tra huỷ trước mỗi bước lấy mẫu; giới hạn PyTorch ở lượng bộ nhớ GPU còn trống thật |

Một worker phục vụ hai model: `minimax-h3-fl2va` (chữ sang video) và `minimax-h3-ref2va` (một ảnh tham chiếu, transformer `transformer_ref` 14 GiB riêng của nó);
hai model chung một thư mục cài vì bộ mã hoá, các VAE và các cấu hình nhỏ là cùng một tệp.

Bất biến riêng của worker này: môi trường của nó không bao giờ chứa `comfyui` (một test đọc tệp khoá); mọi tệp tải về được kiểm với sha256 đã ghim trước khi `install.json` bảo
đảm cho nó; vượt bộ nhớ vật lý của GPU là một lỗi chứ không phải một sự chậm đi âm thầm ([06_deployment.md](06_deployment.md) §6.1).

## 7. Bố cục mã

```text
aha_open/            gateway (không bao giờ import torch — có test ép buộc)
  app.py  scheduler.py  store.py  auth.py  catalog.py  install.py  model_install.py  launcher.py  loras.py  cli.py ...
  workers/           mã worker (ánh xạ từ thư mục workers/ ở gốc khi đóng gói wheel)
  ui/                trang web đi kèm
workers/
  qwen_image/        pyproject + uv.lock riêng, adapter gọi diffusers
  minimax_h3/        pyproject + uv.lock riêng (không ComfyUI); bộ nạp cho các tệp ConvRot của cộng đồng + pipeline H3 của diffusers
  fake/              worker giả cho test (không torch)
windows_installer/   script Inno Setup + build_installer.py (docs/product/06_deployment.md mục 0.1)
conformance/         bộ kiểm hợp chuẩn /v1 (pytest, gọi HTTP thật)
schema/              OpenAPI
docs/
```

## 8. Bất biến

1. Gateway không import torch (test chạy trong tiến trình con, kiểm `sys.modules`).
2. Không có route nào ngoài `/v1/*`, tệp tĩnh `/ui/*` (giao diện web — chỉ gọi `/v1` và `/admin/*`), `/admin/loras*` (quản lý LoRA) và `/admin/models*` (cài model, mục 6.1) — `/admin/*` chỉ ở hồ sơ `local`, chỉ khi người gọi là loopback và hỏi bằng tên loopback, và vẫn cần token; mọi trường hợp khác trả `404`.
3. Không có pixel/prompt nào ra khỏi máy ngoài phản hồi cho chính client đã gửi.
4. Kích thước output đúng như `params` đã được chấp nhận; không resize ngầm.
5. Một job đã `cancelled` không bao giờ có kết quả.
