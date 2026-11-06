"""Cấu hình đọc từ biến môi trường (để deploy Railway/Docker không phải sửa code)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _rate(v: str) -> tuple[int, float]:
    """'120/minute' -> (120, 60.0)."""
    n, _, per = v.partition("/")
    seconds = {"second": 1, "s": 1, "minute": 60, "min": 60, "m": 60, "hour": 3600, "h": 3600}.get(per.strip() or "minute", 60)
    return int(n), float(seconds)


@dataclass
class Config:
    storage_dir: Path = field(default_factory=lambda: Path(_env("STORAGE_DIR", str(ROOT / "storage"))))
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY", "") or secrets.token_hex(32))
    retention_days: float = field(default_factory=lambda: float(_env("RETENTION_DAYS", "7")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "15")))
    max_storage_mb: int = field(default_factory=lambda: int(_env("MAX_STORAGE_MB", "3000")))
    work_side: int = field(default_factory=lambda: int(_env("WORK_SIDE", "1600")))
    preview_side: int = field(default_factory=lambda: int(_env("PREVIEW_SIDE", "800")))
    rate_limit: tuple[int, float] = field(default_factory=lambda: _rate(_env("RATE_LIMIT", "120/minute")))
    heavy_rate_limit: tuple[int, float] = field(default_factory=lambda: _rate(_env("HEAVY_RATE_LIMIT", "20/minute")))
    max_concurrent_jobs: int = field(default_factory=lambda: int(_env("MAX_CONCURRENT_JOBS", "2")))
    secure_cookies: bool = field(default_factory=lambda: _bool(_env("SECURE_COOKIES", "false")))
    cleanup_interval_s: float = field(default_factory=lambda: float(_env("CLEANUP_INTERVAL_S", "3600")))
    trust_proxy: bool = field(default_factory=lambda: _bool(_env("TRUST_PROXY", "false")))
    static_dir: Path = ROOT / "static"
    samples_dir: Path = ROOT / "samples"
    schema_path: Path = ROOT / "db" / "schema.sql"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "app.db"
