# Nét Vẽ: chuyển ảnh thành tranh vẽ

Bài tập lớn môn Xử lý ảnh (INT13146), **Đề tài 4**. Web nhỏ chuyển ảnh y tế, ảnh tự nhiên và ảnh công nghiệp thành tranh vẽ, kết hợp **phát hiện biên**, **chuyển đổi mức xám** và **lọc làm mịn**, có thêm các bộ lọc bảo toàn biên (Bilateral, Kuwahara, Perona-Malik).

> Toàn bộ thuật toán xử lý ảnh được tự cài đặt bằng **NumPy**, **không dùng OpenCV** (theo quy định của đề).
> Pillow chỉ dùng để đọc/ghi file và đổi kích thước ảnh. SciPy chỉ dùng trong test để đối chiếu kết quả.

## Tính năng

- 6 hiệu ứng: Phác thảo chì (color dodge), Chì + biên + vân giấy, Nét mực (line art), Hoạt hình, Chì màu, Sơn dầu.
- Toán tử biên: Roberts, Prewitt, Sobel, Laplacian, LoG (Marr-Hildreth), Canny (đủ 4 bước), XDoG.
- Làm mịn: Gaussian, trung vị, Bilateral, Kuwahara, Perona-Malik. Phân ngưỡng: Otsu, đẳng liệu, tam giác, thích nghi.
- Preset tham số theo loại ảnh (tự nhiên / y tế / công nghiệp). Đọc được ảnh xám 16-bit (X-quang, MRI dạng PNG/TIFF).
- Thanh trượt so sánh trước/sau, xem ảnh trung gian từng bước của pipeline, tab so sánh các thuật toán đặt cạnh nhau, histogram mức xám.
- Tải kết quả về dạng PNG/JPG ở độ phân giải gốc (tối đa 1600px).
- Tự cắt viền letterbox (viền đen/dải chuyển màu của ảnh chụp màn hình video) khi upload. Ảnh xem trước được xử lý theo độ phân giải thật của màn hình (480–1400px). Ảnh xám (nét chì, nét mực) gửi về dạng PNG không mất dữ liệu.
- Thanh "Độ nét" (unsharp mask) và "Đổ bóng" cho các hiệu ứng chì, "Tô mảng tối" (gạch chéo) cho nét mực, lượng tử hoá theo độ sáng cho hoạt hình.
- **Lịch sử và preset lưu trong SQLite**, riêng cho từng trình duyệt (cookie ẩn danh), tự xoá sau 7 ngày.

## Cấu trúc

```
core/      thuật toán xử lý ảnh (NumPy): filters, edges, threshold, morphology, color, point_ops, effects
server/    Flask API, SQLite (db.py), lưu file (image_store.py), rate limit, dọn dẹp
db/        schema.sql
web/       frontend TypeScript (Vite, không framework) -> build ra static/
tests/     pytest: thuật toán (gồm các bài tập trong slide), API, DB
samples/   ảnh mẫu từ Wikimedia Commons (nguồn và giấy phép ghi trong samples/SOURCES.md)
```

## Chạy trên máy (Windows/macOS/Linux)

Cần Python 3.12 và Node 20 trở lên.

```bash
pip install -r requirements-dev.txt
cd web && npm install && npm run build && cd ..   # build frontend ra static/
python app.py                                      # http://localhost:5000
```

Khi sửa frontend thì chạy thêm `cd web && npm run dev` (http://localhost:5173, tự proxy `/api` sang Flask).

Chạy test:

```bash
python -m pytest -q
```

Chạy giống môi trường production bằng Docker:

```bash
docker compose up --build    # http://localhost:8000, dữ liệu nằm trong ./storage-docker
```

## Deploy lên Railway

1. Đưa thư mục `sketch_app/` lên một repo GitHub (`.gitignore` đã loại `storage/`, `static/`, `node_modules/`).
2. Trên [railway.com](https://railway.com), chọn **New Project → Deploy from GitHub repo** và chọn repo vừa tạo. Railway tự đọc `railway.json` và build bằng `Dockerfile`. Image build gồm 2 stage: Node build frontend, sau đó Python chạy gunicorn.
3. **Tạo Volume** (bắt buộc, vì SQLite và ảnh cần ổ đĩa lưu bền): trong service chọn **Volume → New Volume**, đặt mount path là **`/app/storage`**.
   Nếu thiếu volume, mỗi lần deploy hoặc restart sẽ mất hết lịch sử.
4. Tab **Variables**, thêm:

   | Biến | Giá trị | Ghi chú |
   |---|---|---|
   | `SECRET_KEY` | chuỗi ngẫu nhiên dài | ví dụ `python -c "import secrets; print(secrets.token_hex(32))"` |
   | `RETENTION_DAYS` | `7` | số ngày giữ ảnh/lịch sử |
   | `MAX_STORAGE_MB` | `3000` | nhỏ hơn dung lượng volume |
   | `WEB_CONCURRENCY` | `2` | số worker gunicorn |

   `PORT`, `TRUST_PROXY=true` và `STORAGE_DIR=/app/storage` đã đặt sẵn trong Dockerfile, Railway tự truyền `PORT`.
5. Tab **Settings → Networking → Generate Domain** để có địa chỉ `https://<tên>.up.railway.app`.
6. Kiểm tra: mở `https://<tên>.up.railway.app/healthz`, phải trả về `{"ok": true}`.

Lưu ý:
- Chỉ chạy **1 replica**: một volume chỉ gắn được vào một instance, và SQLite là file cục bộ. Muốn chịu tải nhiều hơn thì tăng `WEB_CONCURRENCY` hoặc RAM/CPU của service.
- Volume được mount với quyền root. `docker-entrypoint.sh` sẽ `chown` thư mục storage rồi chạy app dưới user thường `app`.
- Schema DB được tạo lúc app khởi động (volume không có ở bước build).

Deploy bằng CLI (nếu đã cài `railway` và đăng nhập):

```bash
railway link        # chọn project/service
railway up          # build và deploy từ thư mục hiện tại
```

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `STORAGE_DIR` | `./storage` | chứa `app.db`, `uploads/`, `results/`, `thumbs/` |
| `SECRET_KEY` | ngẫu nhiên | khoá ký của Flask |
| `RETENTION_DAYS` | `7` | xoá ảnh/kết quả cũ hơn số ngày này |
| `MAX_UPLOAD_MB` | `15` | giới hạn kích thước file tải lên |
| `MAX_STORAGE_MB` | `3000` | vượt quá thì xoá dần dữ liệu cũ nhất |
| `WORK_SIDE` / `PREVIEW_SIDE` | `1600` / `800` | độ phân giải xử lý khi tải về / khi xem trước |
| `RATE_LIMIT` / `HEAVY_RATE_LIMIT` | `120/minute` / `20/minute` | giới hạn request cho mỗi trình duyệt |
| `MAX_CONCURRENT_JOBS` | `2` | số phép xử lý nặng chạy đồng thời trong mỗi worker |
| `TRUST_PROXY` | `false` | `true` khi chạy sau reverse proxy (Railway) |
| `SECURE_COOKIES` | `false` | ép cookie `Secure` (tự bật khi request đi qua HTTPS) |

## API

| Method | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/effects` | danh sách hiệu ứng, schema tham số, preset theo loại ảnh |
| POST | `/api/upload` | multipart `file`, `category`, trả về `image` |
| POST | `/api/process` | `{image_id, effect, params, steps}`, trả ảnh preview, các bước và histogram |
| POST | `/api/compare` | `{image_id, kind: filters\|edges}` |
| POST | `/api/download` | `{image_id, effect, params, format}`, trả file PNG/JPG độ phân giải cao |
| GET/POST/DELETE | `/api/results[/<id>]` | lịch sử (SQLite) |
| GET/POST/DELETE | `/api/presets[/<id>]` | preset của người dùng |
| GET | `/api/samples`, POST `/api/samples/use` | ảnh mẫu |
| GET | `/healthz` | health check |
