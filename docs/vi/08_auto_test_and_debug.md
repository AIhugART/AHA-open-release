# Tự động kiểm thử và gỡ lỗi: `aha-open-mcp`

> Bản tiếng Anh ([../08_auto_test_and_debug.md](../08_auto_test_and_debug.md)) là bản chính; bản này có thể chậm hơn.
> **Trạng thái: đã làm phần lớn.** ✅ Đã chạy thật với tiến trình `aha-open serve` thật và giao thức MCP thật: công cụ quan sát,
> sandbox dùng một lần, job, kịch bản, khẳng định, sổ chạy, chẩn đoán job hỏng, đích `live`, sandbox model thật (các cổng đã
> kiểm), và model Qwen INT8 thật đã chạy qua một kịch bản hai lần cho kết quả giống hệt từng byte (xem playbook).
> Mã: [`aha_open_mcp/`](../../aha_open_mcp/). Lý do từng lựa chọn nằm trong kế hoạch nội bộ của dự án.

## 1. Nó là gì

`aha-open-mcp` là một máy chủ [MCP](https://modelcontextprotocol.io) cho phép một agent lập trình **điều khiển, quan sát và
phán xét** một server AHA open đang chạy, từ đầu đến cuối:

- **điều khiển** — gửi upload và job, huỷ job, qua API `/v1` công khai như mọi client;
- **quan sát** — đọc log gateway và worker, trạng thái tiến trình, file kết quả; luôn kèm đường dẫn và tuổi file;
- **phán xét** — chạy kịch bản khai báo sẵn, nói ĐẠT/KHÔNG ĐẠT kèm lý do, ghi các lượt chạy và hiện hồi quy thành bản so sánh;
- **chẩn đoán** — với job hỏng: nêu nguyên nhân và việc cần làm, kèm các dòng log đúng cửa sổ thời gian của job.

Nó bổ sung chứ không thay bộ kiểm hợp chuẩn `conformance/`:

| | `conformance/` (pytest) | `aha-open-mcp` |
|---|---|---|
| Câu hỏi | Gateway có theo đặc tả `/v1` không? | Server NÀY, với model THẬT, có ra kết quả đúng không — nếu không thì vì sao? |
| Worker | giả (không torch) | giả trong sandbox, hoặc model thật |
| Chạy khi | trong CI, mỗi thay đổi | khi agent hoặc người phát triển yêu cầu |
| Mức bằng chứng | 🧪 | 🧪 với worker giả, ✅ với model thật |

## 2. Ranh giới

- Nó là một **client**: chỉ nói chuyện qua `/v1` và đọc những file người vận hành đọc được trong thư mục dữ liệu. Không thêm route nào vào server.
- Không bao giờ in token. Kết quả chỉ chép một **danh sách cho phép** các trường.
- Không bao giờ tự chạy lại job đã hỏng. Không bao giờ in lại prompt (sổ chạy chỉ giữ 12 ký tự đầu của sha256).
- Không gửi gì ra ngoài máy.

## 3. Hai đích

| Đích | Là gì | Khi nào |
|---|---|---|
| `sandbox` (mặc định) | Một gateway dùng một lần do chính công cụ khởi động: thư mục dữ liệu tạm, cổng loopback ngẫu nhiên, worker giả (hoặc Qwen thật khi chỉ định). Dừng khi công cụ thoát | Phát triển; cố ý phá mã để chứng minh một cổng bắt được lỗi |
| `live` | Server đang chạy của người vận hành (đọc `AHA_OPEN_DATA_DIR` hoặc thư mục mặc định) | Chỉ khi gọi tường minh; model thật còn cần `allow_gpu=true` |

## 4. Hợp đồng tool — tiền tố là lời hứa

| Tiền tố | Lời hứa |
|---|---|
| `check_*`, `read_*`, `probe_*`, `list_*`, `get_*`, `diagnose_*` | Chỉ đọc. Không đổi trạng thái |
| `assert_*`, `diff_*` | Biến "chạy được" thành "đạt"; sổ chạy chỉ ghi vào thư mục `runs/` của chính công cụ |
| `run_*`, `wait_*` | Hành động. Mỗi `run_*` in khối `SIDE EFFECTS:` **trước** khi làm |

Mọi kết quả mở đầu bằng một dòng đầu: máy, đích, loại worker, mức bằng chứng, và công cụ (`harness`) hay server (`gateway`)
có đang chạy mã cũ hơn file trên đĩa không (`STALE`).

## 5. Kịch bản là dữ liệu

Một kịch bản là một file YAML trong `aha_open_mcp/src/aha_open_mcp/scenarios/`; thêm một mục kiểm = thêm một file, không sửa Python.
Các khoá: `id` (trùng tên file), `title`, `covers` (trỏ tới một test hoặc một tiêu đề tài liệu CÓ THẬT; con trỏ chết thì file bị
đánh dấu `INVALID`), `target`, `requires_worker`, `preconditions`, `side_effects` (danh sách cho phép), `vars`, `steps`
(`run_upload` | `run_job` | `wait_job` | `run_cancel` | `probe_v1` | `check_result_image`), `assert` (đường dẫn trên ngữ cảnh do các
bước điền: `submit`, `job_id`, `upload_id`, `job`, `probe`, `image`). File hỏng được liệt kê `INVALID` kèm lý do, không bao giờ bị bỏ qua im lặng.

## 6. Phía server mà nó dựa vào

Hai file do `aha-open serve` ghi (không đổi API `/v1`): `<data_dir>/logs/gateway.log` (log gateway, xoay vòng theo kích thước) và
`<data_dir>/run/gateway.json` (`pid`, `started_at`, `version`, mã băm nguồn gateway) — để người quan sát thụ động phân biệt "sản phẩm
sai" với "tiến trình đang chạy mã cũ". Không file nào chứa token hay prompt; một test âm tính tìm giá trị mẫu và đòi 0 khớp.

## 7. Đăng ký

```bash
claude mcp add aha-open-mcp -- uv run --project <đường-dẫn>/aha_open_mcp aha-open-mcp
```

Danh sách tool được chốt lúc phiên agent khởi động: khởi động lại phiên sau khi đăng ký hoặc cập nhật. Môi trường của nó riêng
(`aha_open_mcp/pyproject.toml`, `uv.lock`, SDK MCP `>=2,<3`); gateway không bao giờ import nó (một test quét import của gateway).

## 8. Hiện có gì

19 tool: `check_harness`, `check_health`, `probe_v1`, `read_gateway_log`, `read_worker_log`, `full_diagnostics`, `run_sandbox_start`,
`run_sandbox_stop`, `run_ui_url`, `run_upload`, `run_job`, `wait_job`, `run_cancel`, `assert_result_image`, `list_scenarios`,
`get_scenario`, `run_scenario`, `diff_runs`, `diagnose_job`. Chi tiết từng tool: bản tiếng Anh, mục 8.

Điểm an toàn đáng nhớ: `real_model=true` cần `allow_gpu=true` và ít nhất **20 GiB RAM trống** (đỉnh đo được là 17,6 GiB; một lần nạp
khi chỉ còn vài GiB trống từng chết bằng crash native), nếu không công cụ từ chối và nói con số; `allow_low_ram=true` là việc con người
tự chấp nhận rủi ro. Model đã nạp thì không kiểm lại. `run_ui_url` chỉ dành cho sandbox (token chết cùng sandbox); token của `live` không bao giờ được in.
