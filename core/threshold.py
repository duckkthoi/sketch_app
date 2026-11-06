"""Phân ngưỡng (Chương 5, Slide5 tr.67-90): Otsu, đẳng liệu (iso-data), tam giác, ngưỡng thích nghi."""
from __future__ import annotations

import numpy as np

from .filters import box_filter
from .point_ops import histogram


def otsu_from_hist(hist: np.ndarray) -> tuple[int, float]:
    """Thuật toán Otsu trên histogram n_i (i = 0..L-1). Trả về (k*, σ_B²(k*)).

    1. p_i = n_i / MN
    2. P1(k) = sum_{i<=k} p_i
    3. m(k)  = sum_{i<=k} i p_i
    4. m_G   = sum_i i p_i
    5. σ_B²(k) = (m_G P1(k) - m(k))² / (P1(k) (1 - P1(k)))
    6. k* = argmax σ_B²; nếu nhiều cực đại thì lấy trung bình các k.
    """
    hist = np.asarray(hist, dtype=np.float64)
    p = hist / hist.sum()
    levels = np.arange(len(p), dtype=np.float64)
    P1 = np.cumsum(p)
    m = np.cumsum(levels * p)
    mG = m[-1]
    den = P1 * (1.0 - P1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = np.where(den > 1e-12, (mG * P1 - m) ** 2 / den, 0.0)
    best = sb.max()
    ks = np.flatnonzero(np.isclose(sb, best, rtol=1e-9, atol=1e-12))
    return int(round(ks.mean())), float(best)


def otsu_threshold(gray: np.ndarray, levels: int = 256) -> float:
    """Ngưỡng Otsu cho ảnh [0,1]; pixel > ngưỡng thuộc lớp đối tượng."""
    k, _ = otsu_from_hist(histogram(gray, levels))
    return k / (levels - 1)


def isodata_from_hist(p: np.ndarray, t0: float | None = None, delta: float = 1e-3, max_iter: int = 100) -> float:
    """Thuật toán đẳng liệu / lặp (slide tr.83-86) trên histogram chuẩn hoá p(z).

    G1 = {z <= t}, G2 = {z > t};  t_{k+1} = (μ1(t_k) + μ2(t_k)) / 2;  dừng khi |t_{k+1} - t_k| < ΔT.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    z = np.arange(len(p), dtype=np.float64)
    t = float((z * p).sum()) if t0 is None else float(t0)
    for _ in range(max_iter):
        g1 = z <= t
        w1, w2 = p[g1].sum(), p[~g1].sum()
        mu1 = (z[g1] * p[g1]).sum() / w1 if w1 > 0 else t
        mu2 = (z[~g1] * p[~g1]).sum() / w2 if w2 > 0 else t
        t_new = 0.5 * (mu1 + mu2)
        if abs(t_new - t) < delta:
            return t_new
        t = t_new
    return t


def isodata_threshold(gray: np.ndarray, levels: int = 256) -> float:
    return isodata_from_hist(histogram(gray, levels)) / (levels - 1)


def triangle_from_hist(hist: np.ndarray) -> int:
    """Thuật toán tam giác (Zack, slide tr.90): nối đỉnh histogram với đầu mút xa nhất,
    ngưỡng là mức b có khoảng cách từ h[b] tới đường thẳng đó lớn nhất."""
    h = np.asarray(hist, dtype=np.float64)
    nz = np.flatnonzero(h)
    if len(nz) < 2:
        return int(nz[0]) if len(nz) else 0
    bmax = int(np.argmax(h))
    lo, hi = int(nz[0]), int(nz[-1])
    end = lo if (bmax - lo) > (hi - bmax) else hi  # phía đuôi dài hơn
    a, b = sorted((bmax, end))
    xs = np.arange(a, b + 1, dtype=np.float64)
    x1, y1, x2, y2 = bmax, h[bmax], end, h[end]
    # khoảng cách (chưa chuẩn hoá) từ điểm (x, h[x]) tới đường thẳng qua (x1,y1)-(x2,y2)
    d = np.abs((y2 - y1) * xs - (x2 - x1) * h[a:b + 1] + x2 * y1 - y2 * x1)
    return int(xs[np.argmax(d)])


def adaptive_mean_threshold(gray: np.ndarray, block: int = 15, c: float = 0.02) -> np.ndarray:
    """Ngưỡng thích nghi: T(x,y) = trung bình cửa sổ block x block - c.
    Trả về mặt nạ True ở pixel tối hơn lân cận (thường là nét vẽ)."""
    return gray < (box_filter(gray, block) - c)


def binarize(values: np.ndarray, method: str = "otsu", t: float | None = None) -> np.ndarray:
    """Nhị phân hoá ảnh [0,1]: g = 1 nếu f > T. method: 'otsu', 'isodata', 'triangle', 'fixed'."""
    if method == "fixed":
        thr = 0.5 if t is None else t
    elif method == "isodata":
        thr = isodata_threshold(values)
    elif method == "triangle":
        thr = triangle_from_hist(histogram(values)) / 255.0
    else:
        thr = otsu_threshold(values)
    return values > thr
