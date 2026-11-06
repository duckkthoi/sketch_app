"""Lưu ảnh trên đĩa (volume) + cache LRU trong RAM của mỗi worker."""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from core.io_utils import encode, load_image, resize_max


def new_id() -> str:
    return uuid.uuid4().hex


class ImageStore:
    def __init__(self, root: Path, preview_side: int, cache_size: int = 12):
        self.root = root
        self.preview_side = preview_side
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._sized: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        for sub in ("uploads", "results", "thumbs"):
            (root / sub).mkdir(parents=True, exist_ok=True)

    def abspath(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("Đường dẫn không hợp lệ")
        return p

    # ------------------------------------------------------------------ ảnh gốc
    def save_upload(self, img: np.ndarray) -> tuple[str, str]:
        image_id = new_id()
        rel = f"uploads/{image_id}.png"
        self.abspath(rel).write_bytes(encode(img, "PNG"))
        self._put(image_id, img)
        return image_id, rel

    def load(self, image_id: str, rel: str) -> tuple[np.ndarray, np.ndarray]:
        """Trả về (ảnh làm việc độ phân giải cao, ảnh preview)."""
        with self._lock:
            if image_id in self._cache:
                self._cache.move_to_end(image_id)
                return self._cache[image_id]
        img = load_image(self.abspath(rel))
        return self._put(image_id, img)

    def _put(self, image_id: str, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        entry = (img, resize_max(img, self.preview_side))
        with self._lock:
            self._cache[image_id] = entry
            self._cache.move_to_end(image_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return entry

    def preview_at(self, image_id: str, rel: str, side: int) -> np.ndarray:
        """Ảnh preview ở cạnh dài `side` px (cache theo từng kích thước, không phóng to)."""
        work, default = self.load(image_id, rel)
        if side == self.preview_side:
            return default
        key = f"{image_id}@{side}"
        with self._lock:
            if key in self._sized:
                self._sized.move_to_end(key)
                return self._sized[key]
        img = resize_max(work, side)
        with self._lock:
            self._sized[key] = img
            while len(self._sized) > self.cache_size * 2:
                self._sized.popitem(last=False)
        return img

    def forget(self, image_id: str) -> None:
        with self._lock:
            self._cache.pop(image_id, None)

    # ------------------------------------------------------------------ kết quả
    def save_result(self, img: np.ndarray) -> tuple[str, str, str]:
        rid = new_id()
        rel = f"results/{rid}.png"
        thumb = f"thumbs/{rid}.webp"
        self.abspath(rel).write_bytes(encode(img, "PNG"))
        self.abspath(thumb).write_bytes(encode(resize_max(img, 320), "WEBP", 82))
        return rid, rel, thumb

    def delete(self, *rels: str | None) -> None:
        for rel in rels:
            if rel:
                try:
                    self.abspath(rel).unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
