# AHA open — bản đồ tài liệu

| File | Vai trò | Đọc khi |
|---|---|---|
| [01_overview.md](01_overview.md) | Mục tiêu, ranh giới, nguyên tắc | Lần đầu tìm hiểu |
| [03_api_v1.md](03_api_v1.md) | Đặc tả API `/v1` (nguồn sự thật cho tới khi có OpenAPI) | Viết client hoặc một provider `/v1` khác |
| [04_architecture.md](04_architecture.md) | Cấu trúc bên trong AHA open | Phát triển AHA open |
| [05_models.md](05_models.md) | Model, phần cứng, giấy phép | Chọn model, triển khai |
| [06_deployment.md](06_deployment.md) | Ba hồ sơ triển khai | Cài đặt |
| [07_security.md](07_security.md) | Bảo mật | Mở ra LAN/internet |
| [08_auto_test_and_debug.md](08_auto_test_and_debug.md) | `aha-open-mcp`: kiểm thử đầu-cuối và chẩn đoán do agent điều khiển | Kiểm một server thật, gỡ lỗi một job hỏng |

## Quy ước

- **Đã đo** và **suy diễn** luôn tách bạch; số chưa đo ghi "cần đo".
- Đổi API ⇒ sửa `03_api_v1.md` (và OpenAPI khi có) trong cùng commit.
- Thêm model ⇒ thêm một mục trong `05_models.md` với số đo phần cứng và trích dẫn giấy phép nguyên văn.
- Repo này tự đủ: không liên kết tới mã nguồn riêng của AHA Studio.

## Phiên bản

| Thành phần | Phiên bản |
|---|---|
| Đặc tả API | `v1` (bản nháp) |
| AHA open | chưa phát hành |
