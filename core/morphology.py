"""Xử lý ảnh hình thái (Chương 4): giãn, co, mở, đóng với phần tử cấu trúc (SE)."""
from __future__ import annotations

import numpy as np

from .filters import box_filter


def se_square(r: int) -> np.ndarray:
    return np.ones((2 * r + 1, 2 * r + 1), dtype=bool)


def se_cross(r: int) -> np.ndarray:
    se = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    se[r, :] = True
    se[:, r] = True
    return se


def se_disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r * 0.5


def _shift_reduce(img: np.ndarray, se: np.ndarray, op, fill) -> np.ndarray:
    se = np.asarray(se, dtype=bool)
    ry, rx = se.shape[0] // 2, se.shape[1] // 2
    h, w = img.shape[:2]
    p = np.pad(img, ((ry, ry), (rx, rx)) + ((0, 0),) * (img.ndim - 2), mode="constant", constant_values=fill)
    out = None
    for i, j in zip(*np.nonzero(se)):
        sh = p[i:i + h, j:j + w]
        out = sh.copy() if out is None else op(out, sh)
    return out


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Giãn: (A ⊕ B)(x) = max_{b∈B} A(x+b). Với ảnh nhị phân là phép OR các bản dịch."""
    if img.dtype == bool:
        return _shift_reduce(img, se, np.logical_or, False)
    return _shift_reduce(img, se, np.maximum, 0.0)


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Co: (A ⊖ B)(x) = min_{b∈B} A(x+b). Với ảnh nhị phân là phép AND các bản dịch."""
    if img.dtype == bool:
        return _shift_reduce(img, se, np.logical_and, True)
    return _shift_reduce(img, se, np.minimum, 1.0)


def opening(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Mở = co rồi giãn: xoá chi tiết nhỏ hơn SE."""
    return dilate(erode(img, se), se)


def closing(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Đóng = giãn rồi co: lấp lỗ/khe nhỏ, nối nét đứt."""
    return erode(dilate(img, se), se)


def remove_isolated(mask: np.ndarray, min_neighbors: int = 1) -> np.ndarray:
    """Xoá điểm biên cô lập: giữ pixel có ít nhất `min_neighbors` lân cận 8 cũng là biên."""
    cnt = box_filter(mask.astype(np.float32), 3) * 9.0 - mask
    return mask & (cnt >= min_neighbors - 1e-3)


def thicken(mask: np.ndarray, radius: float) -> np.ndarray:
    """Làm dày nét (giãn với đĩa bán kính radius). radius < 0.5 thì giữ nguyên."""
    r = int(round(radius))
    if r <= 0:
        return mask
    return dilate(mask, se_disk(r))
