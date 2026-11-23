# Triển khai AHA open — ba hồ sơ

> Các lệnh `aha-open` dưới đây đã có (🧪 qua test; hồ sơ `lan` có chạy thật trên loopback, chưa thử từ máy thứ hai;
> `internet` chưa thử với tài khoản Cloudflare thật). Ví dụ dùng địa chỉ/tên miền giả.

## 0. Yêu cầu chung

- GPU NVIDIA + driver CUDA (macOS/Apple Silicon: chưa nằm trong v1, cần đo).
- Dung lượng: ~19 GB cho Qwen INT8. MiniMax H3 (chỉ bản lượng tử hoá cộng đồng, chưa cài được) cần khoảng 15–20 GB cho model cộng ~5 GB VAE và một bộ mã hoá văn bản — xem [05_models.md](05_models.md) §2.0.
- Python + `uv` (AHA open tự tạo venv riêng cho từng worker).

```bash
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research            # bản INT8, ~18 GB
# MiniMax (nếu P0 cho phép): đòi thêm xác nhận không ở lãnh thổ bị loại trừ
# aha-open models install minimax-h3-fl2va --accept-license minimax-h3-community --confirm-territory
aha-open doctor
```

## 1. Hồ sơ `local` — cùng máy với AHA Studio

```bash
aha-open serve            # nghe http://127.0.0.1:8890
aha-open clients create --name "AHA Studio trên máy này"   # in token một lần
```

Hồ sơ `local` tự ghi `local_client.token` vào thư mục dữ liệu (chỉ người dùng hiện tại đọc được). Trong AHA Studio:
Settings ▸ AHA open ▸ **Máy này** ▸ Kết nối — AHA Studio tự đọc token, không phải dán. Lệnh `clients create` ở trên chỉ
cần khi AHA Studio chạy dưới tài khoản hệ điều hành khác.

Lưu ý: AHA Studio và AHA open dùng chung GPU với ứng dụng đồ hoạ (Photoshop/Premiere). Khi VRAM căng, đặt
`variant = "int8"` và tránh chạy model video trên cùng máy đang dựng phim.

## 2. Hồ sơ `lan` — một máy GPU phục vụ cả mạng nội bộ

```toml
profile = "lan"
listen = "192.168.1.20:8890"   # chỉ rõ interface; không bao giờ tự nghe 0.0.0.0
[auth]
require_token = true
enroll = true
```

```bash
aha-open tls init           # tạo chứng chỉ tự ký (hoặc trỏ tới chứng chỉ của CA nội bộ công ty)
aha-open tls fingerprint    # SHA-256 để nhân viên đối chiếu
aha-open serve
```

Máy nhân viên: AHA Studio ▸ thêm server `https://192.168.1.20:8890` ▸ Ghép cặp ▸ đối chiếu fingerprint ▸ quản trị viên
`aha-open clients approve <code>`. Mỗi máy một token, thu hồi riêng bằng `aha-open clients revoke <id>`.

Tường lửa: chỉ mở cổng cho dải mạng nội bộ.

## 3. Hồ sơ `internet` — qua Cloudflare Tunnel

Mục tiêu: dùng máy GPU ở nhà/văn phòng từ bất kỳ đâu, **không mở cổng router**, không lộ IP.

```text
AHA Studio (ở xa) ──HTTPS──► Cloudflare edge (Access kiểm service token) ──tunnel──► cloudflared ──► 127.0.0.1:8890 (AHA open)
```

Các bước:

1. Có một tên miền trên Cloudflare (vd `aha-open.example.com`).
2. Cấu hình AHA open:

   ```toml
   profile = "internet"
   listen = "127.0.0.1:8890"      # chỉ loopback; cloudflared là thứ duy nhất nói chuyện với nó
   hosts = ["aha-open.example.com"]   # tên công khai mà tunnel phục vụ; Host khác bị từ chối
   access_confirmed = true            # chỉ đặt sau khi xong bước 4 (Cloudflare Access)
   max_queue_per_client = 2
   [auth]
   require_token = true
   enroll = false                # mặc định tắt ở internet
   ```

3. Cài `cloudflared`, tạo tunnel trỏ `aha-open.example.com` → `http://127.0.0.1:8890`.
4. Cloudflare Zero Trust ▸ Access ▸ tạo application cho `aha-open.example.com` với policy **Service Auth**, tạo một
   **service token** cho mỗi người dùng/máy.
5. `aha-open clients create --name "Laptop An"` → token AHA open.
6. Trong AHA Studio: URL `https://aha-open.example.com`, dán token AHA open, `CF-Access-Client-Id`, `CF-Access-Client-Secret`.

`aha-open tunnel --hostname aha-open.example.com` in sẵn `config.yml` của cloudflared, các dòng cần thêm vào
`config.toml` và các bước Access (bước 3–4 cần đăng nhập tài khoản Cloudflare của người vận hành nên AHA open **không**
tự làm thay). Sau khi xong, kiểm từ một mạng khác:

```bash
aha-open check https://aha-open.example.com --cf-id <id> --cf-secret <secret> --token <token AHA open>
```

`aha-open check` nói rõ lớp nào hỏng: server tắt, tên không phân giải, chứng chỉ không tin cậy, Cloudflare Access chặn
(sai service token), hay token AHA open sai/đã thu hồi.

**Cần biết trước khi chọn hồ sơ này:**

| Điều | Hệ quả |
|---|---|
| TLS kết thúc ở edge của Cloudflare | Cloudflare có thể thấy ảnh/prompt dạng rõ. Dữ liệu nhạy cảm ⇒ dùng `lan` hoặc VPN |
| Giới hạn thân request (theo hiểu biết 100 MB ở gói Free — **cần đo**) | Vì vậy input đi qua `POST /v1/uploads` theo khúc |
| Timeout phản hồi của proxy (**cần đo**) | API là job bất đồng bộ, không có request nào chờ cả lượt tạo |
| Giấy phép model | Qwen (`not_allowed`) chỉ cho nghiên cứu/đánh giá. MiniMax H3 **tắt mặc định** ở hồ sơ này: phục vụ người khác qua API là "Hosted Service" theo giấy phép MiniMax, người vận hành phải tự có điều khoản người dùng, biện pháp an toàn và kênh báo vi phạm rồi mới đặt `hosted_service_obligations = "accepted"`. MiniMax không dùng được ở EU/Anh/Hàn/Mỹ |
| Chi phí GPU/điện | Giới hạn hàng đợi theo client để một người không chiếm hết |

## 4. Chạy như dịch vụ hệ thống

- Windows: dịch vụ (dự kiến `aha-open service install`) hoặc Task Scheduler lúc đăng nhập.
- Linux: unit `systemd` (mẫu sẽ có trong repo).
- Cập nhật: `aha-open update` — dừng nhận job mới, chờ job đang chạy xong, rồi khởi động lại.

## 5. Kiểm tra sau khi cài

```bash
aha-open doctor                      # GPU, cấu hình, model
aha-open open                        # mở trang web (/ui/); hồ sơ local không phải dán token
curl http://127.0.0.1:8890/v1/health
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8890/v1/manifest
```

Từ máy khác (lan): `curl --cacert <chứng chỉ> https://192.168.1.20:8890/v1/health`.
Qua internet: `curl -H "CF-Access-Client-Id: …" -H "CF-Access-Client-Secret: …" https://aha-open.example.com/v1/health`.
