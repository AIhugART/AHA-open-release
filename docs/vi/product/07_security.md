# Bảo mật AHA open

## 1. Tài sản cần bảo vệ

| Tài sản | Vì sao |
|---|---|
| Ảnh, prompt, video của người dùng | Dữ liệu riêng, có thể là tài sản khách hàng của một công ty |
| GPU | Tài nguyên đắt; bị chiếm = mất dịch vụ, tốn điện |
| Token client | Ai có token là dùng được GPU và đọc được job của chủ token |
| Tên/địa chỉ server nội bộ | Lộ cấu trúc mạng công ty |

## 2. Mối đe doạ theo hồ sơ

| Đe doạ | `local` | `lan` | `internet` |
|---|---|---|---|
| Trang web trong trình duyệt gọi tới `127.0.0.1`/IP nội bộ | có | có | có |
| Máy khác trong mạng gọi vào | — | có | — |
| Nghe lén đường truyền | — | có | (Cloudflare thấy dữ liệu rõ ở edge) |
| Bot quét toàn internet | — | — | có |
| Token lộ | thấp | trung bình | cao |

## 3. Luật bắt buộc

1. **Mặc định chỉ nghe `127.0.0.1`.** Hồ sơ `lan` phải chỉ rõ interface; cấm tự nghe `0.0.0.0`.
2. **`lan` bắt buộc HTTPS.** Chứng chỉ tự ký + fingerprint, hoặc chứng chỉ của CA nội bộ.
3. **`internet` chỉ qua tunnel:** gateway vẫn nghe loopback; bắt buộc Cloudflare Access (service token) **và** token
   AHA open — hai lớp độc lập. Ghép cặp tắt mặc định.
4. **Token riêng từng client**, chỉ lưu hash, thu hồi riêng, hiện đúng một lần.
5. **Quyền sở hữu job:** chỉ token tạo job mới đọc/huỷ/tải kết quả; token khác nhận `404`.
6. **Chặn trình duyệt:** request có `Origin` lạ ⇒ `403`; kiểm `Host` thuộc danh sách tên server đã cấu hình
   (chống DNS rebinding).
7. **Giới hạn:** hàng đợi theo client, kích thước upload, số khúc, tốc độ tạo job; `429` khi vượt.
8. **Không telemetry ra ngoài.** AHA open không gửi gì tới AHA hay bên thứ ba.
9. **Log không chứa prompt hay token.** Chỉ ghi id job, model, kích thước, thời gian, mã lỗi.
10. **Mã chạy trong worker là mã đã ghim.** Không `trust_remote_code` với mã chưa kiểm; tải trọng số `safetensors`,
    không nạp pickle tuỳ ý. (Ghi chú: pipeline MiniMax dùng lớp model đã có trong diffusers ghim commit — kiểm ở P5
    rằng không cần `trust_remote_code` từ Hub.)
11. **Ghép cặp phải có người duyệt** trên máy server; mã ghép cặp hết hạn sau vài phút, dùng một lần.
12. **`local_client.token`** (hồ sơ `local`) chỉ ghi khi hồ sơ là `local`, quyền đọc chỉ cho người dùng hiện tại; xoá
    khi đổi sang hồ sơ khác.
13. **Model có nghĩa vụ "hosted service"** (MiniMax H3) tắt mặc định ở hồ sơ `internet`.
14. **`/admin` dành cho người ngồi tại máy** (quản lý LoRA và cài, chuyển và xoá biến thể model — tải tới 18 GB, xoá file và dựng môi trường Python, nên một trang ở site khác hay máy khác không bao giờ được phép bắt đầu việc đó). Chỉ ở hồ sơ `local`, chỉ khi người gọi là loopback và hỏi bằng tên loopback, và vẫn cần token; mọi trường hợp khác trả `404`. Chỉ cần người gọi là loopback là chưa đủ: ở hồ sơ `internet`, tunnel kết nối từ `127.0.0.1`.

## 4. Test âm tính bắt buộc (bộ kiểm hợp chuẩn)

| Test | Kỳ vọng |
|---|---|
| Không token / token sai | `401` |
| Token A đọc/huỷ/tải job của token B | `404` |
| Header `Origin: https://evil.example` | `403` |
| `Host` lạ | `403` |
| Hồ sơ `lan` khởi động với `listen = 0.0.0.0` | từ chối khởi động |
| Hồ sơ `lan` không có TLS | từ chối khởi động |
| Upload sai sha256 | `400 upload_incomplete` |
| Vượt hàng đợi | `429 busy` |
| Mã ghép cặp đã dùng/hết hạn | không cấp token |
| Token đã thu hồi | `401` ngay lần gọi kế tiếp |
| Hồ sơ `lan`/`internet` | không có `local_client.token` trên đĩa |
| Hồ sơ `internet`, MiniMax chưa chấp nhận nghĩa vụ | model `state` không phải `ready`, job bị từ chối |
| `/admin/loras` ở hồ sơ `lan`/`internet`, kể cả có token hợp lệ và `Host: 127.0.0.1` | `404`, không ghi gì |
| `/admin/loras` không token / sai token | `401`, không ghi gì |
| `/admin/loras` với `Host` hoặc `Origin` lạ | `403`, không ghi gì |
| `/admin/models` (liệt kê, cài, trạng thái, huỷ) ở hồ sơ `internet`, kể cả qua đường của chính tunnel (`Host: 127.0.0.1` hay tên công khai) | `404`, không tiến trình nào chạy |
| `/admin/models` không token / sai token | `401`, không tiến trình nào chạy |
| `/admin/models/{id}/install` với `Host` hoặc `Origin` lạ | `403`, không tiến trình nào chạy |
| Yêu cầu cài không có đúng khoá giấy phép (thiếu, rỗng, sai hoa/thường, `true`) hoặc model không cài được | `400` / `404`, không tiến trình nào chạy |
| Cài khi không có GPU, GPU nhỏ, thiếu `uv`, thiếu thư viện tải hoặc thiếu ổ đĩa | `409` kèm mã riêng của nguyên nhân, không tiến trình nào chạy |
| Huỷ cài, hoặc tắt gateway khi đang cài | cả cây tiến trình biến mất, không nhận kết quả nào |
| Lỗi cài hiện trên trang | không bao giờ chứa đường dẫn thư mục dữ liệu |
| Tải lên qua trang quản trị một file không phải LoRA, id sai hoặc thông tin sai | `400`, không để lại gì trong `loras/` hay `tmp/` |
| Tìm kiếm phủ định: prompt/token mẫu trong log và file trạng thái | 0 lần khớp |

## 5. Trách nhiệm của người vận hành

- Tuân thủ giấy phép từng model (xem [05_models.md](05_models.md)).
- Quyết định có chấp nhận Cloudflare thấy dữ liệu (hồ sơ `internet`) hay không.
- Bảo vệ máy chạy AHA open: một tiến trình khác trên cùng máy với quyền người dùng đọc được `data_dir`.

## 6. Báo lỗi bảo mật

Sẽ có `SECURITY.md` trước khi repo chuyển sang công khai (kênh báo riêng, không qua issue công khai).
