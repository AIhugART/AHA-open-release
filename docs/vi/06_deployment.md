# Triển khai AHA open — ba hồ sơ

> Các lệnh `aha-open` dưới đây đã có (🧪 qua test; hồ sơ `lan` có chạy thật trên loopback, chưa thử từ máy thứ hai;
> `internet` chưa thử với tài khoản Cloudflare thật). Ví dụ dùng địa chỉ/tên miền giả.

## 0. Yêu cầu chung

- GPU NVIDIA + driver CUDA (macOS/Apple Silicon: chưa nằm trong v1, cần đo).
- Dung lượng: 15–19 GB cho một biến thể Qwen (text encoder và VAE dùng chung chỉ tải một lần; mỗi biến thể ConvRot thêm 4,6–7,3 GB). MiniMax H3 (bản "mixed" INT4/INT8 của cộng đồng, cài được từ dòng lệnh hoặc từ trang web) cần
  **~46 GB**: transformer 14,8 GiB + bộ mã hoá văn bản 25,3 GiB + các VAE 5,4 GiB, từ ba repo được ghim, cộng môi trường PyTorch riêng
  của nó — xem [05_models.md](05_models.md) §2.0 và §2.3.
- GPU cho MiniMax H3: **ít nhất 16 GB** (thẻ 8 và 12 GB không hỗ trợ); model ảnh tham chiếu cần 18 GB trở lên. Bản và cỡ nào vừa:
  [05_models.md](05_models.md) §2.5. Dung lượng tải một bản khoảng 43–50 GB.
- Dùng chung GPU với chương trình khác: `[gpu]` `memory_limit_gib` trong `config.toml` giới hạn bộ nhớ server được dùng.
- Python + `uv` (AHA open tự tạo venv riêng cho từng worker).

```bash
aha-open init --profile local
aha-open models install qwen-image-2.1 --accept-license qwen-research            # biến thể hợp GPU của bạn (~17 GB); `--variant` để chọn biến thể khác, xem `models list`
# MiniMax H3 (~46 GB): đòi thêm xác nhận không ở lãnh thổ bị loại trừ
# aha-open models install minimax-h3-fl2va --accept-license minimax-h3 --confirm-territory
# MiniMax H3 với ảnh tham chiếu (model thứ hai): vào cùng thư mục và chỉ tải transformer 14 GiB của riêng nó khi dòng trên đã chạy
# aha-open models install minimax-h3-ref2va --accept-license minimax-h3 --confirm-territory
aha-open doctor
```

## 0.1 Trình cài Windows (Windows 10/11, 64-bit)

`AHA-Open-Setup-<phiên bản>.exe` là trình cài nhỏ (khoảng 16 MiB: `uv` và wheel của AHA open; **không** có PyTorch, **không** có trọng số
model). Cài theo người dùng hiện tại (không cần quyền admin) vào `%LOCALAPPDATA%\Programs\AHA open`, dựng môi trường Python cho gateway bằng
`uv` (lần đầu phải tải Python nên cần internet), ghim mọi phụ thuộc theo `uv.lock`, và ghi cấu hình `local` mặc định. Cài đặt, model và ảnh
nằm riêng ở `%LOCALAPPDATA%\AHA-open`.

- **Cài:** chạy file `.exe`. Chưa ký số nên Windows SmartScreen có thể cảnh báo lần đầu; so SHA-256 của file với `SHA256SUMS` trên trang release.
- **Chạy lần đầu:** Start menu ▸ *AHA open*. Gateway chạy nền (không có cửa sổ console) rồi mở trang web.
- **Model:** trình cài **không** tải model. Trang web hiện *Download a Model* kèm giấy phép; tick ô đồng ý rồi bấm nút (khoảng 18 GB, thêm
  môi trường Python riêng của model). MiniMax H3 nằm trên cùng panel (khoảng 46 GB; giấy phép của nó loại trừ EU, Anh, Hàn Quốc và Mỹ, nên một
  ô tick thứ hai yêu cầu người dùng xác nhận mình không ở các vùng đó; model tham chiếu khi đó chỉ cần thêm ~15 GB). Xem
  [05_models.md](05_models.md) và `docs/04_architecture.md` mục 6.1. Việc tải bị từ chối trước khi bắt đầu nếu không có GPU NVIDIA (hoặc dưới
  ~10 GiB VRAM), hoặc còn dưới ~20 GB đĩa trống.
- **Dừng:** Start menu ▸ *Stop AHA open* (dừng luôn lượt tải đang chạy).
- **Gỡ cài:** Settings ▸ Apps. Dừng gateway trước, **giữ model và cài đặt của bạn** trừ khi bạn trả lời *Yes* cho câu hỏi xoá dữ liệu.

Cài lỗi sẽ nói rõ vì sao (không có internet / hết ổ đĩa / Windows không cho ghi) và không để lại gì (thoát mã khác 0, không có mục trong
*Add or remove programs*). Chưa hỗ trợ macOS ở v1.

Mức bằng chứng: ✅ trình cài đã build được cài vào một thư mục tạm với một thư mục dữ liệu tạm; các gói đã cài khớp với phiên bản ghim và không chứa
PyTorch; nó được gỡ (tệp chương trình mất, dữ liệu còn); lần cài đè thứ hai nâng cấp bản đầu chỉ với một mục *Add or remove programs*; hai lỗi mạng thật
(ngoại tuyến, proxy chết) thoát với mã khác 0, không đăng ký gì và tự giải thích. 🧪 các chốt chặn trong `tests/test_windows_installer.py`. ⬜ Chưa thấy:
hộp thoại cài tương tác (mới chạy đường im lặng), cảnh báo SmartScreen, máy hoàn toàn chưa có Python (máy thử đã có Python 3.12); chưa có biểu tượng.
✅ Đầu-cuối trên một máy thật: đã cài, khởi động bằng lệnh của lối tắt, model tải qua trang (bị cắt một lần rồi tiếp tục), ảnh tham chiếu được sinh
(sha256 giống hệt số đo P0, [05_models.md](05_models.md)), rồi gỡ cài với chương trình bị xoá và dữ liệu còn.

**Đính chính (2026-11-24).** Các lần chạy smoke đầu khởi động gateway bằng `python -m aha_open.cli` từ bản checkout mã nguồn; `-m` đặt *thư mục hiện tại*
lên đầu `sys.path`, nên chúng kiểm bản checkout chứ không phải gói đã cài (người dùng không bị ảnh hưởng: lối tắt khởi động trong thư mục cài). Smoke nay chạy mọi
thứ từ một thư mục trung lập, kiểm `aha_open` được import từ môi trường đã cài và `source_digest` của gateway đang chạy (`run/gateway.json`) bằng digest của
các tệp đã cài; `aha-open start` và việc cài model từ trang web cũng khởi động các tiến trình con trong thư mục dữ liệu vì cùng lý do. Chỉ bằng chứng từ bản smoke đã sửa mới tính.

### Build trình cài (dành cho người bảo trì)

```powershell
winget install JRSoftware.InnoSetup        # Inno Setup 6; ISCC.exe KHÔNG được đưa vào PATH, script build tự tìm
python windows_installer/build_installer.py --smoke     # test -> wheel -> trình cài -> cài vào thư mục tạm, chạy, gỡ
python windows_installer/build_installer.py --check-tag v0.1.0-dev1    # tag phải khớp phiên bản duy nhất trong pyproject.toml
```

Build từ chối: thay đổi chưa commit trong các đường dẫn được đóng gói (build đóng gói cây làm việc chứ không phải một commit; `--allow-dirty` để thử),
test đỏ (`AHA_OPEN_BUILD_SKIP_QUALITY_GATE=1` bỏ qua, và nói to), wheel cần PyTorch hoặc mang trọng số model, thiếu ghim phụ thuộc, và kích thước ngoài khoảng
10–80 MiB. `--smoke-offline` cài khi internet bị làm cho không với tới được và kiểm rằng nó thất bại gọn gàng. Build ghi `dist/AHA-Open-Setup-<phiên bản>.exe`,
wheel nằm trong nó và `dist/SHA256SUMS` (đúng hai tệp đó).

### Phát hành (dành cho người bảo trì)

Mọi bước công khai (đẩy snapshot, GitHub Release) mỗi lần đều là quyết định của chủ dự án.

1. **Một phiên bản, không bao giờ dùng lại.** `pyproject.toml` và `aha_open/__init__.py` mang cùng một số (và đã chạy `uv lock`). Số đã công khai không
   bao giờ dùng lại cho các byte khác: một bản phát hành có wheel khác wheel bên trong trình cài của nó là một bản build khác và nhận số mới.
2. `python windows_installer/build_installer.py --smoke` (cây sạch, test, rồi trình cài thật được cài, chạy và gỡ).
3. `python windows_installer/build_installer.py --check-tag v<phiên bản>`; `gh auth status`.
4. *(công khai)* `python tools/release/snapshot.py OUT_DIR` ghi cây công khai từ `HEAD` (chỉ byte đã commit; nó lỗi nếu một thư mục cấp cao nhất chưa được
   phân loại trong `tools/release/`); đẩy nó lên repo phát hành công khai.
5. *(công khai)* **Tải lên trước, rồi mới nói ở nơi khác:** `gh release create v<phiên bản> --repo <repo công khai> --prerelease --notes-file NOTES
   dist/AHA-Open-Setup-<phiên bản>.exe dist/aha_open-<phiên bản>-py3-none-any.whl dist/SHA256SUMS`.
6. Kiểm từ bên ngoài, ẩn danh: tải ba tệp từ trang release và chạy `sha256sum -c SHA256SUMS` (PowerShell: `Get-FileHash`). "Tải lên xong" không phải là phép kiểm.
7. Chỉ khi đó mới cập nhật bất cứ thứ gì trỏ tới bản phát hành (một trang tải) bằng các băm **đọc từ bản phát hành**, không gõ theo trí nhớ.

| Nếu bạn đã làm… | Lấy lại bằng… | Lưu ý |
|---|---|---|
| Tải lên nhầm tệp | `gh release upload --clobber` với tệp đúng | Đừng xoá một tệp mà ai đó có thể đã liên kết tới |
| Công bố nhầm phiên bản | `gh release edit --draft` để ẩn nó, rồi công bố một số **mới** | Không bao giờ dùng lại số đã công khai |
| Trỏ một trang tới nhầm tệp | Sửa trang và công bố lại | Nhanh nhất: không đổi tệp nhị phân |

### Các vấn đề gặp khi build (triệu chứng → nguyên nhân → cách xử lý)

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| "ISCC not found" dù Inno Setup đã cài | `winget` cài theo người dùng và không thêm vào `PATH` | Script build tìm trong `%LOCALAPPDATA%\Programs\Inno Setup 6`; kiểm bằng `winget list --id JRSoftware.InnoSetup` |
| Gỡ cài im lặng chạy từ Git Bash bị treo ở một cửa sổ | MSYS đổi `/VERYSILENT` thành `C:/Program Files/Git/VERYSILENT` | Dùng PowerShell (`Start-Process … -ArgumentList "/VERYSILENT"`) hoặc `MSYS_NO_PATHCONV=1` |
| Một lần cài lỗi thoát mã 0 và còn trong *Add or remove programs* | Môi trường được dựng sau khi các tệp đã đăng ký; Inno chỉ ghi lại ngoại lệ phát sinh ở đó | Nó được dựng trong `PrepareToInstall`; `--smoke-offline` canh giữ điều này |
| Smoke xanh nhưng đã kiểm bản mã nguồn | `python -m aha_open.cli` đặt thư mục hiện tại lên đầu `sys.path` | Smoke chạy từ thư mục trung lập và so `source_digest`; `start` và việc cài model khởi động tiến trình con trong thư mục dữ liệu |
| "weights found in the installer" cho vài byte | `.pth` cũng là tệp đường dẫn của Python (`_virtualenv.pth`) | Một tệp chỉ tính là trọng số theo đuôi **và** kích thước (≥ 10 MiB) |
| Lỗi ngoại tuyến chỉ nói "could not be prepared" | Danh sách dấu hiệu không chứa câu chữ thật của uv | Dấu hiệu lấy từ log thật lưu trong `tests/fixtures/installer/`; thêm log mới khi có câu chữ mới |
| Thanh tiến độ của trang đứng ở 0% hàng phút | `hf-xet` giữ byte trong bộ đệm riêng và ráp từng tệp ở cuối | Thanh dùng bộ đếm byte của chính trình tải (`AHAOPEN_PROGRESS`) |
| Một lần tải bị cắt ngang được báo là *ready* và *Download* không tiếp tục | `model_index.json` nằm trong số các tệp đến đầu tiên | Chưa xong = có tệp `.incomplete`, hoặc hf đã bắt đầu mà không có `install.json` kết thúc nó |

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

## 6. Khi tiến trình worker chết

Một job có thể thất bại với thông điệp như `while loading the model: the worker crashed in native code (access violation) (exit
code 3221225477 = 0xC0000005). The request is not the cause; try again, and if it repeats report it with that log.
worker-qwen_image.log in the server's logs folder shows where it stopped.` `code` của lỗi vẫn là `generation_failed`.

- Thông điệp nêu nguyên nhân mà hệ điều hành báo (một crash, hoặc hết bộ nhớ) và khuyên khác nhau cho từng loại: lỗi bộ nhớ thì bảo giải phóng bộ nhớ; crash thì
  nói rằng request không phải nguyên nhân. Client chỉ được cho biết **tên tệp** log, không bao giờ là đường dẫn trên server.
- Log là `<data dir>/logs/worker-<họ>.log`. Trong lúc nạp model, worker ghi các dòng như
  `[worker +  41.2s rss=14.3GiB ram_available=9.8GiB] load text_encoder: weights read` (số giây từ lúc worker khởi động, bộ nhớ của worker, và RAM còn trống), nên một lần
  crash cho thấy **nạp tới đâu và còn bao nhiêu RAM**. Nếu tiến trình chết ở mức native, `faulthandler` còn ghi stack Python của mọi luồng
  (`Windows fatal exception: access violation` / `Fatal Python error: Segmentation fault`), cho biết dòng nào đang chạy.
- Bộ nhớ eo hẹp trên máy 32 GiB: nạp bản Qwen INT8 làm worker giữ khoảng **17,6 GiB** RAM (đo trên máy RTX 3090 Ti, RAM 31,7 GiB), để lại 1,8–4,5 GiB trống khi
  các chương trình khác dùng phần còn lại. Với ít RAM trống hơn, việc nạp vẫn xong, chỉ chậm hơn (132 s thay vì 87 s trong một test với 5,8 GiB trống lúc đầu), vì Windows
  chuyển bộ nhớ của worker sang tệp trang; hãy để trống chỗ trên ổ chứa tệp trang.
- Một lần crash native khi nạp đã thấy một lần (mã thoát 3221225477, lần nạp đầu khi nguội) và không tái hiện được sau đó, nên nguyên nhân chưa biết;
  các dòng trên là thứ sẽ cho thấy nếu nó xảy ra lần nữa.
- Không có gì được tự chạy lại: chạy lại tốn phút GPU, nên client quyết định. Job kế tiếp khởi động một worker mới.

### 6.1 GPU đầy trên Windows: chậm đi thay vì báo lỗi (MiniMax H3)

Trên Windows, driver NVIDIA có thể chuyển bộ nhớ GPU sang RAM hệ thống khi card đầy, thay vì báo "out of memory". Không có gì thất bại; công việc chỉ chậm đi
nhiều lần. Đo trên RTX 3090 Ti với MiniMax H3 ở 1344×768: một khối transformer mất **3,95 s khi card đầy** và **1,19 s khi còn chỗ** (cùng mã), biến một clip 20 phút
thành 40 phút. Vì vậy worker MiniMax giới hạn PyTorch ở lượng bộ nhớ còn trống thật khi một job bắt đầu (trừ 0,5 GiB) — vượt quá khi đó là lỗi hết bộ nhớ với
thông điệp rõ ràng — và tính các lớp feed-forward theo từng lát để một clip 1344×768 nằm dưới giới hạn đó. Hãy đóng các chương trình khác dùng GPU (game, công cụ AI khác)
trước một clip dài. Thiết lập *CUDA – Sysmem Fallback Policy: Prefer No Sysmem Fallback* trong NVIDIA Control Panel khiến chính driver từ chối tràn bộ nhớ; nó là tuỳ chọn.
