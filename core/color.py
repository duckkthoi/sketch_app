"""Chuyển đổi mức xám và không gian màu (Chương 3 - Xử lý ảnh màu)."""
from __future__ import annotations

import numpy as np

# Trọng số độ sáng theo chuẩn ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh RGB sang ảnh mức xám. Ảnh đã xám thì trả về nguyên."""
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    return (img[..., :3] @ LUMA_WEIGHTS).astype(np.float32)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=-1) if gray.ndim == 2 else gray


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    return gray_to_rgb(img).astype(np.float32, copy=False)


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> YCbCr [0,1] (Cb, Cr có tâm 0.5)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (b - y) * 0.564
    cr = 0.5 + (r - y) * 0.713
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    y, cb, cr = img[..., 0], img[..., 1] - 0.5, img[..., 2] - 0.5
    r = y + 1.403 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.773 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1).astype(np.float32)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> HSV với H, S, V đều trong [0,1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = img.max(axis=-1)
    minc = img.min(axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-12), 0.0)
    safe = np.maximum(delta, 1e-12)
    rc = (maxc - r) / safe
    gc = (maxc - g) / safe
    bc = (maxc - b) / safe
    h = np.where(r == maxc, bc - gc, np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(delta > 0, (h / 6.0) % 1.0, 0.0)
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    h, s, v = img[..., 0], img[..., 1], img[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int32) % 6
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)
