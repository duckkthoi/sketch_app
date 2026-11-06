-- Schema SQLite cho Sketch App. Chạy lại được nhiều lần (IF NOT EXISTS).
-- Đường dẫn file ảnh lưu TƯƠNG ĐỐI so với STORAGE_DIR.

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Ảnh gốc người dùng tải lên (đã thu nhỏ về tối đa WORK_SIDE px).
CREATE TABLE IF NOT EXISTS images (
    id          TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL,
    filename    TEXT,
    category    TEXT NOT NULL DEFAULT 'nature',
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    path        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_client ON images (client_id, created_at);
CREATE INDEX IF NOT EXISTS idx_images_created ON images (created_at);

-- Kết quả người dùng lưu vào lịch sử.
CREATE TABLE IF NOT EXISTS results (
    id          TEXT PRIMARY KEY,
    image_id    TEXT REFERENCES images (id) ON DELETE SET NULL,
    client_id   TEXT NOT NULL,
    effect      TEXT NOT NULL,
    params      TEXT NOT NULL,          -- JSON
    category    TEXT NOT NULL DEFAULT 'nature',
    filename    TEXT,
    path        TEXT NOT NULL,          -- PNG độ phân giải làm việc
    thumb_path  TEXT NOT NULL,          -- WEBP 320px
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_client ON results (client_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_results_created ON results (created_at);

-- Bộ tham số người dùng tự đặt tên.
CREATE TABLE IF NOT EXISTS presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    effect      TEXT NOT NULL,
    params      TEXT NOT NULL,          -- JSON
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_presets_client ON presets (client_id, created_at);
