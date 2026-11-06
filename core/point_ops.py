"""Toán tử điểm ảnh và histogram (Chương 2 - Nâng cao chất lượng ảnh)."""
from __future__ import annotations

import numpy as np


def negative(img: np.ndarray) -> np.ndarray:
    """Biến đổi âm bản: s = (L-1) - r, với ảnh [0,1] là s = 1 - r."""
    return (1.0 - img).astype(np.float32)


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    """Biến đổi lũy thừa s = r^gamma. gamma > 1 làm tối, gamma < 1 làm sáng."""
    return np.power(np.clip(img, 0, 1), gamma).astype(np.float32)


def contrast_stretch(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Giãn tương phản tuyến tính giữa hai phân vị low_pct và high_pct."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi - lo < 1e-6:
        return img.astype(np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram n_k của ảnh xám [0,1] lượng tử về `bins` mức."""
    idx = np.clip((gray * (bins - 1) + 0.5).astype(np.int64), 0, bins - 1)
    return np.bincount(idx.ravel(), minlength=bins)


def equalize_hist(gray: np.ndarray, levels: int = 256) -> np.ndarray:
    """Cân bằng histogram: s_k = (L-1) * sum_{j<=k} p_j  (rồi chuẩn hoá về [0,1])."""
    h = histogram(gray, levels).astype(np.float64)
    cdf = np.cumsum(h) / h.sum()
    idx = np.clip((gray * (levels - 1) + 0.5).astype(np.int64), 0, levels - 1)
    return cdf[idx].astype(np.float32)


def quantize_uniform(img: np.ndarray, levels: int) -> np.ndarray:
    """Lượng tử hoá đều: giảm số mức xám/màu xuống `levels` mức (gom cụm mức xám)."""
    levels = max(2, int(levels))
    return (np.round(np.clip(img, 0, 1) * (levels - 1)) / (levels - 1)).astype(np.float32)


def kmeans_colors(img: np.ndarray, k: int, iters: int = 8, sample: int = 20000, seed: int = 0) -> np.ndarray:
    """Lượng tử hoá màu bằng phân cụm k-means (Chương 5 - phân cụm).

    Học các tâm cụm trên một mẫu pixel ngẫu nhiên rồi gán mọi pixel về tâm gần nhất.
    """
    k = max(2, int(k))
    flat = img.reshape(-1, img.shape[-1] if img.ndim == 3 else 1).astype(np.float32)
    rng = np.random.default_rng(seed)
    pts = flat[rng.choice(len(flat), size=min(sample, len(flat)), replace=False)]
    # Khởi tạo kiểu k-means++ đơn giản để tâm cụm trải đều.
    centers = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d = np.min(((pts[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        prob = d / d.sum() if d.sum() > 0 else None
        centers.append(pts[rng.choice(len(pts), p=prob)])
    centers = np.array(centers, dtype=np.float32)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                centers[j] = pts[m].mean(axis=0)
    # Gán toàn ảnh theo từng khối để tiết kiệm bộ nhớ.
    out = np.empty_like(flat)
    step = 200_000
    for s in range(0, len(flat), step):
        blk = flat[s:s + step]
        lab = np.argmin(((blk[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        out[s:s + step] = centers[lab]
    return out.reshape(img.shape).astype(np.float32)


def enhance(gray: np.ndarray, mode: str) -> np.ndarray:
    """Bước tiền xử lý tương phản dùng chung cho các hiệu ứng."""
    if mode == "stretch":
        return contrast_stretch(gray)
    if mode == "equalize":
        return equalize_hist(gray)
    return gray
