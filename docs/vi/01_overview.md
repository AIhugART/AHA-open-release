# AHA open — tổng quan

## 1. Vấn đề

Model mở như Qwen-Image-2.1 và MiniMax H3 cho chất lượng tốt nhưng cần PyTorch, CUDA, hàng chục GB trọng số và
phiên bản thư viện ghim riêng cho từng model. Người dùng ứng dụng sáng tạo không nên phải lo những thứ đó, và ứng dụng
sáng tạo cũng không nên chứa chúng.

## 2. Giải pháp

AHA open là một **dịch vụ chạy riêng**:

- chạy trên máy có GPU (máy của bạn, server trong công ty, hay máy ở nhà truy cập qua internet);
- nạp model mở, nhận yêu cầu qua API HTTP `/v1`, trả kết quả ảnh/video;
- phục vụ nhiều client cùng lúc (hàng đợi, công bằng theo client).

## 3. Ranh giới

| Thuộc AHA open | Không thuộc AHA open |
|---|---|
| Chạy model, quản lý VRAM, hàng đợi | Tài khoản, thanh toán, quyền dùng của AHA Studio |
| Token theo client, ghép cặp máy | Kho ảnh/video của người dùng (AHA Studio giữ) |
| Manifest khả năng + giấy phép từng model | Quyết định giấy phép thay người vận hành |
| Hướng dẫn triển khai LAN/internet | Trọng số model (người vận hành tự tải) |

**AHA Studio không phụ thuộc AHA open.** AHA Studio chỉ biết chuẩn `/v1`; dịch vụ nào cài đúng `/v1` cũng dùng được.
Ngược lại, AHA open không phụ thuộc AHA Studio: client nào nói `/v1` cũng dùng được.

**AHA open là dự án độc lập**, mã nguồn mở (Apache-2.0), ở repo riêng, có phiên bản và lịch phát hành riêng. AHA
Studio là một client của nó; mã của hai dự án không phụ thuộc nhau.

**AHA open có giao diện web riêng, đơn giản** (`/ui/`) để tạo ảnh và tạo video mà không cần AHA Studio: chọn model, gõ
prompt, chọn kích thước/thời lượng, đính kèm ảnh tham chiếu hoặc khung đầu/cuối, theo dõi tiến độ, huỷ, tải kết quả.
Nó **không** phải kho ảnh hay trình chỉnh sửa — chỉ giữ các kết quả gần đây trong thời hạn lưu job. Giao diện này là
một client `/v1` như mọi client khác, không có đường đi riêng tới model.

## 4. Nguyên tắc

1. **Dữ liệu ở lại nơi người dùng chọn.** Ảnh, prompt, video đi từ client tới máy chạy AHA open, không đi đâu khác.
   AHA open không gửi telemetry ra ngoài. Ngoại lệ do người vận hành tự chọn: Cloudflare Tunnel (xem
   [06_deployment.md](06_deployment.md)).
2. **Khả năng phải đo được.** Manifest chỉ khai những gì đã chạy được trên phần cứng đó.
3. **Giấy phép theo từng model và hiển thị rõ.** Không đóng gói trọng số.
4. **Mở rộng mạng là việc tường minh.** Mặc định chỉ nghe `127.0.0.1`.
5. **Mỗi model một môi trường.** Nâng một model không làm hỏng model khác.
6. **Huỷ là thật.** Huỷ dừng ở bước kế tiếp của pipeline; không có kết quả nào trả về sau khi đã huỷ.

## 5. Ai dùng

| Người dùng | Kịch bản |
|---|---|
| Cá nhân có GPU | AHA Studio và AHA open cùng một máy |
| Studio/công ty | Một server GPU phục vụ các máy nhân viên trong LAN |
| Cá nhân/nhóm nhỏ | Máy GPU ở nhà, dùng từ laptop nơi khác qua Cloudflare Tunnel |
| Nhà phát triển | Dùng `/v1` cho ứng dụng của mình |
