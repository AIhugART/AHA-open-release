# AHA open

> Bản tiếng Việt (bản dịch). **Bản chính là tiếng Anh: [README.md](README.md).** Bản này có thể chậm hơn bản chính.

**AHA open** là một dịch vụ mã nguồn mở viết bằng Python, chạy các model AI mở trên GPU của chính bạn và cung cấp
chúng qua một API HTTP chuẩn (`/v1`). Ứng dụng client đầu tiên là **AHA Studio**: AHA Studio gửi yêu cầu tạo ảnh hoặc
video tới AHA open và nhận kết quả về kho của nó, về Photoshop (AHA Paint), Premiere (AHA Clip) hoặc AHA Workflow.

> **Trạng thái (2026-11-23):** chưa có bản phát hành. Gateway `/v1` + bộ kiểm hợp chuẩn đã có (qua test với worker giả);
> worker Qwen-Image-2.1 và giao diện web `/ui/` đã viết; MiniMax H3 hoãn. Mức bằng chứng từng phần: xem bảng trạng thái
> trong [docs/vi/05_models.md](docs/vi/05_models.md) §3 và bản tiếng Anh [README.md](README.md).

## Bắt đầu nhanh (từ mã nguồn, cần `uv`)

```bash
uv sync --extra dev --extra download
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research   # bản INT8, ~18 GB
aha-open serve                                                          # nghe 127.0.0.1:8890
aha-open open                                                           # mở trang web, không cần dán token
```

Qwen-Image-2.1 chỉ cung cấp **bản INT8** (~18 GB đĩa, ~9,5 GiB VRAM ở 1024², hợp GPU 12–24 GB); bản BF16 đầy đủ không được hỗ trợ.

**LoRA:** `aha-open loras add my-style.safetensors --name "My style" --trigger mystyle` — file được kiểm (chỉ safetensors) rồi
hiện trong manifest/`/ui/`; client chọn bằng `params.loras`.

Trên máy chạy AHA open, trang web cũng thêm/gỡ file LoRA được (**Manage LoRA files**), và nút **Refresh** đọc lại danh sách sau khi bạn tự chép file vào; không cần khởi động lại.

## Mục tiêu v1

| Model | Việc | Ghi chú |
|---|---|---|
| Qwen-Image-2.1 | Tạo ảnh từ chữ, sửa ảnh (≤10 ảnh tham chiếu), ảnh trong suốt RGBA, LoRA | Giấy phép phi thương mại (Qwen Research License) |
| MiniMax H3 | Tạo video có âm thanh stereo | **Hoãn**; chỉ dự kiến dùng bản lượng tử hoá cộng đồng (không dùng bản đầy đủ ~133 GB), chưa thử; xem [docs/05_models.md](docs/vi/05_models.md) §2.0 |

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
| [docs/01_overview.md](docs/vi/01_overview.md) | AHA open là gì, ranh giới, nguyên tắc |
| [docs/03_api_v1.md](docs/vi/03_api_v1.md) | Đặc tả API `/v1` |
| [docs/04_architecture.md](docs/vi/04_architecture.md) | Gateway, worker, hàng đợi, quản lý VRAM |
| [docs/05_models.md](docs/vi/05_models.md) | Qwen-Image-2.1, MiniMax H3: khả năng, phần cứng, giấy phép |
| [docs/06_deployment.md](docs/vi/06_deployment.md) | Cài đặt cho `local`, `lan`, `internet` (Cloudflare Tunnel) |
| [docs/07_security.md](docs/vi/07_security.md) | Mô hình đe doạ và luật bảo mật |

## Giấy phép

Mã của AHA open: **Apache-2.0** (xem `LICENSE`, `NOTICE`; danh sách giấy phép phụ thuộc: `THIRD_PARTY_LICENSES.md`). Dùng miễn phí. **Trọng số model không nằm trong repo này**; mỗi model có giấy phép
riêng, người vận hành tự tải về và tự chịu trách nhiệm tuân thủ. Xem [docs/05_models.md](docs/vi/05_models.md).
