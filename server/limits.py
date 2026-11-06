"""Giới hạn tần suất (rate limit) và số job xử lý ảnh chạy đồng thời trong một worker."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from .schemas import ApiError


class RateLimiter:
    """Cửa sổ trượt: tối đa `n` request mỗi `per` giây cho mỗi khoá (client_id/IP)."""

    def __init__(self, n: int, per: float):
        self.n, self.per = n, per
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.per:
                q.popleft()
            if len(q) >= self.n:
                raise ApiError(429, "Bạn thao tác quá nhanh, thử lại sau ít giây")
            q.append(now)
            if len(self._hits) > 10_000:  # dọn khoá cũ để không phình bộ nhớ
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.per]:
                    del self._hits[k]


class JobGate:
    """Semaphore giới hạn số phép xử lý nặng chạy cùng lúc (tránh quá tải CPU)."""

    def __init__(self, n: int, wait_s: float = 30.0):
        self._sem = threading.BoundedSemaphore(max(1, n))
        self.wait_s = wait_s

    @contextmanager
    def slot(self):
        if not self._sem.acquire(timeout=self.wait_s):
            raise ApiError(503, "Máy chủ đang bận, thử lại sau")
        try:
            yield
        finally:
            self._sem.release()
