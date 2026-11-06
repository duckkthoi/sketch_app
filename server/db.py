"""Truy cập SQLite bằng module sqlite3 chuẩn (không ORM)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from flask import current_app, g

SCHEMA_VERSION = 1


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=15, detect_types=0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def init_db(path: Path, schema_path: Path) -> None:
    """Tạo bảng nếu chưa có. Chạy lúc khởi động app (volume Railway chỉ có khi container chạy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # đọc và ghi đồng thời giữa các worker
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect(current_app.config["APP"].db_path)
    return g.db


def close_db(_exc: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


# --------------------------------------------------------------------------- images
def insert_image(db, *, id: str, client_id: str, filename: str, category: str, width: int, height: int, path: str) -> None:
    db.execute(
        "INSERT INTO images (id, client_id, filename, category, width, height, path, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (id, client_id, filename, category, width, height, path, time.time()),
    )
    db.commit()


def get_image(db, image_id: str, client_id: str) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM images WHERE id = ? AND client_id = ?", (image_id, client_id)).fetchone()


# --------------------------------------------------------------------------- results
def _result_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["params"] = json.loads(d["params"])
    return d


def insert_result(db, **kw: Any) -> None:
    kw = {**kw, "params": json.dumps(kw["params"], ensure_ascii=False), "created_at": time.time()}
    cols = ", ".join(kw)
    db.execute(f"INSERT INTO results ({cols}) VALUES ({', '.join('?' for _ in kw)})", tuple(kw.values()))
    db.commit()


def list_results(db, client_id: str, page: int, per_page: int) -> tuple[list[dict], int]:
    total = db.execute("SELECT COUNT(*) FROM results WHERE client_id = ?", (client_id,)).fetchone()[0]
    rows = db.execute(
        "SELECT r.*, (i.id IS NOT NULL) AS image_available FROM results r LEFT JOIN images i ON i.id = r.image_id "
        "WHERE r.client_id = ? ORDER BY r.created_at DESC LIMIT ? OFFSET ?",
        (client_id, per_page, (page - 1) * per_page),
    ).fetchall()
    return [_result_dict(r) for r in rows], total


def get_result(db, result_id: str, client_id: str) -> dict | None:
    row = db.execute(
        "SELECT r.*, (i.id IS NOT NULL) AS image_available FROM results r LEFT JOIN images i ON i.id = r.image_id "
        "WHERE r.id = ? AND r.client_id = ?",
        (result_id, client_id),
    ).fetchone()
    return _result_dict(row) if row else None


def delete_result(db, result_id: str, client_id: str) -> dict | None:
    row = get_result(db, result_id, client_id)
    if row:
        db.execute("DELETE FROM results WHERE id = ? AND client_id = ?", (result_id, client_id))
        db.commit()
    return row


# --------------------------------------------------------------------------- presets
def list_presets(db, client_id: str) -> list[dict]:
    rows = db.execute("SELECT * FROM presets WHERE client_id = ? ORDER BY created_at DESC", (client_id,)).fetchall()
    return [_result_dict(r) for r in rows]


def count_presets(db, client_id: str) -> int:
    return db.execute("SELECT COUNT(*) FROM presets WHERE client_id = ?", (client_id,)).fetchone()[0]


def insert_preset(db, client_id: str, name: str, effect: str, params: dict) -> dict:
    cur = db.execute(
        "INSERT INTO presets (client_id, name, effect, params, created_at) VALUES (?,?,?,?,?)",
        (client_id, name, effect, json.dumps(params, ensure_ascii=False), time.time()),
    )
    db.commit()
    return _result_dict(db.execute("SELECT * FROM presets WHERE id = ?", (cur.lastrowid,)).fetchone())


def delete_preset(db, preset_id: int, client_id: str) -> bool:
    cur = db.execute("DELETE FROM presets WHERE id = ? AND client_id = ?", (preset_id, client_id))
    db.commit()
    return cur.rowcount > 0
