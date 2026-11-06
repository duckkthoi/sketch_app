"""Dọn dẹp storage: xoá ảnh/kết quả quá hạn, giữ tổng dung lượng dưới giới hạn, xoá file mồ côi."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .db import connect

log = logging.getLogger(__name__)


def _unlink(root: Path, *rels: str | None) -> int:
    freed = 0
    for rel in rels:
        if not rel:
            continue
        p = root / rel
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    return freed


def _storage_bytes(root: Path) -> int:
    return sum(p.stat().st_size for sub in ("uploads", "results", "thumbs") for p in (root / sub).glob("*") if p.is_file())


def purge(storage: Path, db_path: Path, retention_days: float, max_storage_mb: int) -> dict:
    stats = {"results": 0, "images": 0, "orphans": 0}
    conn = connect(db_path)
    try:
        cutoff = time.time() - retention_days * 86400
        for r in conn.execute("SELECT id, path, thumb_path FROM results WHERE created_at < ?", (cutoff,)).fetchall():
            _unlink(storage, r["path"], r["thumb_path"])
            conn.execute("DELETE FROM results WHERE id = ?", (r["id"],))
            stats["results"] += 1
        for r in conn.execute("SELECT id, path FROM images WHERE created_at < ?", (cutoff,)).fetchall():
            _unlink(storage, r["path"])
            conn.execute("DELETE FROM images WHERE id = ?", (r["id"],))
            stats["images"] += 1
        conn.commit()

        # Vượt dung lượng: xoá dần từ cũ nhất (ảnh gốc trước, rồi tới kết quả).
        limit = max_storage_mb * 1024 * 1024
        total = _storage_bytes(storage)
        if total > limit:
            for table, cols in (("images", "path"), ("results", "path, thumb_path")):
                for r in conn.execute(f"SELECT id, {cols} FROM {table} ORDER BY created_at").fetchall():
                    if total <= limit:
                        break
                    total -= _unlink(storage, *[r[c.strip()] for c in cols.split(",")])
                    conn.execute(f"DELETE FROM {table} WHERE id = ?", (r["id"],))
                    stats[table] += 1
                conn.commit()

        # File mồ côi (không có dòng DB tương ứng) và cũ hơn 1 giờ.
        known = {r[0] for r in conn.execute("SELECT path FROM images UNION SELECT path FROM results UNION SELECT thumb_path FROM results")}
        old = time.time() - 3600
        for sub in ("uploads", "results", "thumbs"):
            for p in (storage / sub).glob("*"):
                rel = f"{sub}/{p.name}"
                if rel not in known and p.stat().st_mtime < old:
                    _unlink(storage, rel)
                    stats["orphans"] += 1
    finally:
        conn.close()
    return stats


def start_background(cfg) -> threading.Thread | None:
    if cfg.cleanup_interval_s <= 0:
        return None

    def loop():
        while True:
            try:
                st = purge(cfg.storage_dir, cfg.db_path, cfg.retention_days, cfg.max_storage_mb)
                if any(st.values()):
                    log.info("cleanup: %s", st)
            except Exception:  # noqa: BLE001 - không để thread chết
                log.exception("cleanup failed")
            time.sleep(cfg.cleanup_interval_s)

    t = threading.Thread(target=loop, name="storage-cleanup", daemon=True)
    t.start()
    return t
