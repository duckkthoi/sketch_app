"""Validate dữ liệu request. Mọi tham số hiệu ứng đi qua Effect.normalize() (ép kiểu + kẹp miền)."""
from __future__ import annotations

import re
from typing import Any

from core.effects import CATEGORIES, EFFECTS

ID_RE = re.compile(r"^[0-9a-f]{32}$")
FORMATS = {"png": ("PNG", "image/png"), "jpg": ("JPEG", "image/jpeg"), "jpeg": ("JPEG", "image/jpeg")}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def body(req) -> dict:
    data = req.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError(400, "Body phải là JSON object")
    return data


def image_id(value: Any) -> str:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise ApiError(400, "image_id không hợp lệ")
    return value


def effect_and_params(data: dict) -> tuple[str, dict]:
    eid = data.get("effect")
    if eid not in EFFECTS:
        raise ApiError(400, "Hiệu ứng không tồn tại")
    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ApiError(400, "params phải là object")
    return eid, EFFECTS[eid].normalize(params)


def category(value: Any) -> str:
    return value if value in CATEGORIES else "nature"


def out_format(value: Any) -> tuple[str, str, str]:
    key = str(value or "png").lower()
    if key not in FORMATS:
        raise ApiError(400, "Định dạng phải là png hoặc jpg")
    fmt, mime = FORMATS[key]
    return fmt, mime, "jpg" if fmt == "JPEG" else "png"


def preset_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, "Tên preset không được trống")
    return value.strip()[:60]


def safe_filename(name: str | None) -> str:
    base = re.sub(r"[^\w\-. ]+", "_", (name or "image").rsplit("/", 1)[-1].rsplit("\\", 1)[-1], flags=re.UNICODE)
    return (base.strip(" .") or "image")[:80]
