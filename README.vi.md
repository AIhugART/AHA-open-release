# AHA open

> Bản tiếng Việt (bản dịch). **Bản chính là tiếng Anh: [README.md](README.md).** Bản này có thể chậm hơn bản chính.

**AHA open** là một dịch vụ mã nguồn mở viết bằng Python, chạy các model AI mở trên GPU của chính bạn và cung cấp
chúng qua một API HTTP chuẩn (`/v1`). Ứng dụng client đầu tiên là **AHA Studio**: AHA Studio gửi yêu cầu tạo ảnh hoặc
video tới AHA open và nhận kết quả về kho của nó, về Photoshop (AHA Paint), Premiere (AHA Clip) hoặc AHA Workflow.

> **Đừng dùng trình cài `0.1.0.dev2`:** bản đó thiếu mã worker nên Qwen-Image-2.1 ConvRot và MiniMax H3 không chạy được. Dùng
> `0.1.0.dev3` trở lên; cài đè lên dev2 vẫn giữ model và cấu hình.
>
> **Trạng thái (2026-11-28):** bản pre-release `0.1.0.dev3`, có trình cài Windows. Đã chạy thật trên một máy (RTX 3090 Ti
> 24 GiB, Windows): ảnh với Qwen-Image-2.1 và video với MiniMax H3 qua gateway. **Chưa thử:** hồ sơ `lan` từ máy thứ hai và
> hồ sơ `internet` (cần tài khoản Cloudflare thật). Mức bằng chứng từng phần: [docs/product/05_models.md](docs/vi/product/05_models.md) §3.

## Bắt đầu nhanh trên Windows (trình cài)

1. Từ [trang Releases](https://github.com/AIhugART/AHA-open-release/releases) tải `AHA-Open-Setup-<phiên bản>.exe` và `SHA256SUMS` (hiện là
   bản pre-release), so SHA-256 của file với `SHA256SUMS`. Trình cài **chưa ký số** nên Windows SmartScreen có thể cảnh báo lần đầu.
2. Chạy file (khoảng 16 MiB, không cần quyền admin; lần đầu cần internet để tải Python). Rồi Start menu ▸ **AHA open**: gateway chạy nền và
   trang web mở ra.
3. Trên trang, tick ô đồng ý giấy phép rồi bấm **Download model** (Qwen-Image-2.1, biến thể hợp GPU của bạn, khoảng 17 GB; chỉ dùng nghiên cứu/đánh giá — đọc giấy phép
   trang hiển thị). Cần GPU NVIDIA từ khoảng 10 GiB VRAM và khoảng 25 GB ổ đĩa trống.

Đã thử thật trên một máy (RTX 3090 Ti, Windows 11): cài, tải model qua trang, sinh ảnh chuẩn (cùng byte với lần đo trước), gỡ cài. **Chưa thử:**
hộp thoại cài tương tác, cảnh báo SmartScreen, máy chưa có Python. Chưa hỗ trợ macOS ở v1. Chi tiết: [docs/product/06_deployment.md](docs/product/06_deployment.md) mục 0.1.

## Bắt đầu nhanh (từ mã nguồn, cần `uv`)

```bash
uv sync --extra dev --extra download
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research   # biến thể hợp GPU của bạn (~17 GB); xem `models list`
aha-open serve                                                          # nghe 127.0.0.1:8890
aha-open open                                                           # mở trang web, không cần dán token
```

Qwen-Image-2.1 có **bốn biến thể lượng tử hoá** để chọn theo máy — `int8`, `int8-convrot`, `mixed-p010`, `mixed-p015` (15–18 GB đĩa, 8–10 GiB VRAM ở 1024², hợp GPU 12–24 GB; `aha-open models list` cho biết biến thể nào vừa máy và đã cài, `models use` / `models remove` để chuyển và xoá); bản BF16 đầy đủ không được hỗ trợ.

**LoRA:** `aha-open loras add my-style.safetensors --name "My style" --trigger mystyle` — file được kiểm (chỉ safetensors) rồi
hiện trong manifest/`/ui/`; client chọn bằng `params.loras`.

Trên máy chạy AHA open, trang web cũng thêm/gỡ file LoRA được (**Manage LoRA files**), và nút **Refresh** đọc lại danh sách sau khi bạn tự chép file vào; không cần khởi động lại.

## Mục tiêu v1

| Model | Việc | Ghi chú |
|---|---|---|
| Qwen-Image-2.1 | Tạo ảnh từ chữ, sửa ảnh (≤10 ảnh tham chiếu), ảnh trong suốt RGBA, LoRA | Giấy phép phi thương mại (Qwen Research License) |
| MiniMax H3 | Tạo video có âm thanh stereo | **Hoãn**; chỉ dự kiến dùng bản lượng tử hoá cộng đồng (không dùng bản đầy đủ ~133 GB), chưa thử; xem [docs/product/05_models.md](docs/vi/product/05_models.md) §2.0 |

## Ba cách chạy

| Hồ sơ | Ở đâu | Bảo mật |
|---|---|---|
| `local` | Cùng máy với AHA Studio | Chỉ nghe `127.0.0.1` |
| `lan` | Một máy GPU trong mạng nội bộ | HTTPS bắt buộc + token riêng từng máy |
| `internet` | Qua Cloudflare Tunnel | Cloudflare Access + token riêng từng máy |

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/00_index.md](docs/vi/00_index.md) | Bản đồ tài liệu |
| [docs/product/01_overview.md](docs/vi/product/01_overview.md) | AHA open là gì, ranh giới, nguyên tắc |
| [docs/product/03_api_v1.md](docs/vi/product/03_api_v1.md) | Đặc tả API `/v1` |
| [docs/product/04_architecture.md](docs/vi/product/04_architecture.md) | Gateway, worker, hàng đợi, quản lý VRAM |
| [docs/product/05_models.md](docs/vi/product/05_models.md) | Qwen-Image-2.1, MiniMax H3: khả năng, phần cứng, giấy phép |
| [docs/product/06_deployment.md](docs/vi/product/06_deployment.md) | Cài đặt cho `local`, `lan`, `internet` (Cloudflare Tunnel) |
| [docs/product/07_security.md](docs/vi/product/07_security.md) | Mô hình đe doạ và luật bảo mật |

## Giấy phép

Mã của AHA open: **Apache-2.0** (xem `LICENSE`, `NOTICE`; danh sách giấy phép phụ thuộc: `THIRD_PARTY_LICENSES.md`). Dùng miễn phí. **Trọng số model không nằm trong repo này**; mỗi model có giấy phép
riêng, người vận hành tự tải về và tự chịu trách nhiệm tuân thủ. Xem [docs/product/05_models.md](docs/vi/product/05_models.md).
