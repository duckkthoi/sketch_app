"""Phát hiện biên (Chương 5 - Phân vùng ảnh, Slide5 tr.13-58).

Quy ước: gx là đạo hàm theo chiều ngang (cột), gy theo chiều dọc (hàng, hướng xuống).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .filters import correlate2d, gaussian_blur, pad

# Mặt nạ 3x3 dạng tổng quát (slide tr.18-19): hàng/cột ngoài có trọng số 1, hàng giữa có c.
# Prewitt: c = 1, Sobel: c = 2.
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = PREWITT_X.T.copy()
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()
LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
LAPLACIAN_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)


# --------------------------------------------------------------------------- gradient bậc 1
def roberts(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Toán tử Roberts (mặt nạ chéo 2x2): gx = f(y,x) - f(y+1,x+1), gy = f(y,x+1) - f(y+1,x)."""
    p = np.pad(gray.astype(np.float32), ((0, 1), (0, 1)), mode="edge")
    gx = p[:-1, :-1] - p[1:, 1:]
    gy = p[:-1, 1:] - p[1:, :-1]
    return gx, gy


def prewitt(gray: np.ndarray, mode: str = "reflect") -> tuple[np.ndarray, np.ndarray]:
    return correlate2d(gray, PREWITT_X, mode), correlate2d(gray, PREWITT_Y, mode)


def sobel(gray: np.ndarray, mode: str = "reflect") -> tuple[np.ndarray, np.ndarray]:
    return correlate2d(gray, SOBEL_X, mode), correlate2d(gray, SOBEL_Y, mode)


def gradient(gray: np.ndarray, operator: str = "sobel") -> tuple[np.ndarray, np.ndarray]:
    ops = {"sobel": sobel, "prewitt": prewitt, "roberts": roberts}
    if operator not in ops:
        raise ValueError(f"Toán tử gradient không hợp lệ: {operator}")
    return ops[operator](gray)


def magnitude(gx: np.ndarray, gy: np.ndarray, l1: bool = False) -> np.ndarray:
    """Độ lớn gradient |G| = sqrt(gx^2 + gy^2), hoặc xấp xỉ |gx| + |gy| (l1=True)."""
    return (np.abs(gx) + np.abs(gy)) if l1 else np.hypot(gx, gy)


def direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Hướng gradient θ = atan2(gy, gx), đơn vị độ trong (-180, 180]."""
    return np.degrees(np.arctan2(gy, gx))


# --------------------------------------------------------------------------- đạo hàm bậc 2
def laplacian(gray: np.ndarray, variant: str = "4") -> np.ndarray:
    """Toán tử Laplacian ∇²f (slide tr.30-35). variant '4' (lân cận 4) hoặc '8' (lân cận 8)."""
    return correlate2d(gray, LAPLACIAN_8 if variant == "8" else LAPLACIAN_4)


def log_kernel(sigma: float) -> np.ndarray:
    """Kernel Laplacian of Gaussian, kích thước ~6σ (slide tr.39-42), chuẩn hoá tổng = 0."""
    r = max(1, int(np.ceil(3 * sigma)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    rr = x * x + y * y
    k = (rr - 2 * sigma**2) / sigma**4 * np.exp(-rr / (2 * sigma**2))
    return (k - k.mean()).astype(np.float32)


def log_filter(gray: np.ndarray, sigma: float) -> np.ndarray:
    """LoG = lọc Gaussian rồi lấy Laplacian (tương đương nhân chập với log_kernel nhưng nhanh hơn)."""
    return laplacian(gaussian_blur(gray, sigma)) * np.float32(sigma * sigma)


def zero_crossings(resp: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Tìm điểm giao không: trong cửa sổ 3x3, có cặp lân cận đối diện (trái/phải, trên/dưới,
    2 đường chéo) trái dấu và chênh lệch tuyệt đối > threshold (slide tr.42)."""
    p = pad(resp, 1, 1, "edge")
    h, w = resp.shape
    out = np.zeros((h, w), dtype=bool)
    pairs = [((1, 0), (1, 2)), ((0, 1), (2, 1)), ((0, 0), (2, 2)), ((0, 2), (2, 0))]
    for (y1, x1), (y2, x2) in pairs:
        a = p[y1:y1 + h, x1:x1 + w]
        b = p[y2:y2 + h, x2:x2 + w]
        out |= (a * b < 0) & (np.abs(a - b) > threshold)
    return out


def zero_crossing_strength(resp: np.ndarray) -> np.ndarray:
    """Độ mạnh giao không: max |a-b| trên các cặp lân cận đối diện trái dấu (0 nếu không có giao không)."""
    p = pad(resp, 1, 1, "edge")
    h, w = resp.shape
    out = np.zeros((h, w), dtype=np.float32)
    for (y1, x1), (y2, x2) in [((1, 0), (1, 2)), ((0, 1), (2, 1)), ((0, 0), (2, 2)), ((0, 2), (2, 0))]:
        a = p[y1:y1 + h, x1:x1 + w]
        b = p[y2:y2 + h, x2:x2 + w]
        out = np.maximum(out, np.where(a * b < 0, np.abs(a - b), 0.0))
    return out


def marr_hildreth(gray: np.ndarray, sigma: float = 2.0, threshold_ratio: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Phương pháp Marr-Hildreth: zero-crossing của LoG, ngưỡng = threshold_ratio * max|LoG|.
    Trả về (bản đồ biên nhị phân, đáp ứng LoG)."""
    resp = log_filter(gray, sigma)
    thr = threshold_ratio * float(np.abs(resp).max() + 1e-12)
    return zero_crossings(resp, thr), resp


def dog(gray: np.ndarray, sigma: float = 1.0, k: float = 1.6) -> np.ndarray:
    """Difference of Gaussians: G_σ * f - G_kσ * f, xấp xỉ LoG (k = 1.6)."""
    return gaussian_blur(gray, sigma) - gaussian_blur(gray, k * sigma)


def xdog(gray: np.ndarray, sigma: float = 1.0, k: float = 1.6, p: float = 20.0, eps: float = 0.35, phi: float = 10.0) -> np.ndarray:
    """eXtended DoG (Winnemöller 2012) - phiên bản DoG cho nét vẽ mực.

    D = (1+p) G_σ - p G_kσ = G_σ + p·DoG; T(D) = 1 nếu D >= ε, ngược lại 1 + tanh(φ(D - ε)).
    Trả về ảnh [0,1]: 1 = giấy trắng, gần 0 = nét mực.
    """
    g1 = gaussian_blur(gray, sigma)
    g2 = gaussian_blur(gray, k * sigma)
    d = (1.0 + p) * g1 - p * g2
    return np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps))).astype(np.float32)


def xdog_lines(gray: np.ndarray, sigma: float = 1.0, k: float = 1.6, eps: float = 0.25, phi: float = 8.0) -> np.ndarray:
    """Biến thể XDoG chỉ giữ nét (bỏ thành phần cường độ G_σ để vùng tối không bị tô đen).

    DoG được chuẩn hoá theo phân vị 99.5 của |DoG| nên ngưỡng ε ổn định giữa các ảnh;
    nét xuất hiện ở phía tối của biên, nơi DoG âm: T = 1 nếu DoG_n >= -ε, ngược lại 1 + tanh(φ(DoG_n + ε)).
    """
    d = dog(gray, sigma, k)
    d = d / (float(np.percentile(np.abs(d), 99.5)) + 1e-6)
    return np.where(d >= -eps, 1.0, 1.0 + np.tanh(phi * (d + eps))).astype(np.float32)


# --------------------------------------------------------------------------- Canny
def non_maximum_suppression(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bước 3 của Canny: giữ pixel chỉ khi độ lớn là cực đại địa phương theo hướng gradient.

    Hướng được lượng tử hoá về 4 hướng cơ bản: 0°, 45°, 90°, 135° (slide tr.49-52).
    """
    h, w = mag.shape
    ang = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    p = np.pad(mag, 1, mode="constant")

    def nb(dy: int, dx: int) -> np.ndarray:
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    # (dy, dx) của lân cận theo hướng gradient; trục y hướng xuống.
    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    d135 = (ang >= 112.5) & (ang < 157.5)
    n1 = np.select([d0, d45, d90, d135], [nb(0, 1), nb(1, 1), nb(1, 0), nb(1, -1)])
    n2 = np.select([d0, d45, d90, d135], [nb(0, -1), nb(-1, -1), nb(-1, 0), nb(-1, 1)])
    keep = (mag >= n1) & (mag > n2)
    return np.where(keep, mag, 0.0).astype(np.float32)


def hysteresis(nms: np.ndarray, low: float, high: float) -> np.ndarray:
    """Bước 4 của Canny: phân ngưỡng kép (slide tr.53-54).

    - > high: điểm biên mạnh, giữ ngay.
    - < low : loại.
    - ở giữa: chỉ giữ nếu nối (8-lân cận) được tới một điểm biên mạnh -> duyệt BFS từ các điểm mạnh.
    """
    h, w = nms.shape
    weak = nms >= low
    strong = nms >= high
    out = strong.copy()
    # Làm việc trên mảng phẳng có viền 1 pixel để không phải kiểm tra biên.
    W = w + 2
    weak_f = np.pad(weak & ~strong, 1).ravel()
    out_f = np.pad(out, 1).ravel()
    offsets = (-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1)
    queue = deque(np.flatnonzero(out_f).tolist())
    weak_l = weak_f.tolist()  # list Python truy cập nhanh hơn mảng numpy từng phần tử
    while queue:
        i = queue.popleft()
        for o in offsets:
            j = i + o
            if weak_l[j]:
                weak_l[j] = False
                out_f[j] = True
                queue.append(j)
    return out_f.reshape(h + 2, W)[1:-1, 1:-1]


def canny(gray: np.ndarray, sigma: float = 1.4, low_ratio: float = 0.5, high_ratio: float = 0.2,
          return_steps: bool = False, high_quantile: float | None = None):
    """Phương pháp Canny đầy đủ 4 bước (slide tr.44-55).

    1. Làm mịn Gaussian(σ)  2. Gradient Sobel  3. Non-maximum suppression  4. Ngưỡng kép.
    high = high_ratio * max(độ lớn sau NMS); low = low_ratio * high (slide gợi ý T1 = 2*T2 => 0.5).
    Nếu cho `high_quantile` (0..1) thì high là phân vị tương ứng của các pixel còn lại sau NMS:
    cách này ổn định hơn khi ảnh có vài biên cực mạnh (ví dụ chữ trên phim X-quang).
    """
    smoothed = gaussian_blur(gray, sigma)
    gx, gy = sobel(smoothed)
    mag = magnitude(gx, gy)
    nms = non_maximum_suppression(mag, gx, gy)
    mx = float(nms.max())
    if mx <= 1e-9:
        edges = np.zeros_like(gray, dtype=bool)
    else:
        if high_quantile is not None:
            high = float(np.quantile(nms[nms > 0], float(np.clip(high_quantile, 0, 1))))
        else:
            high = high_ratio * mx
        edges = hysteresis(nms, low_ratio * high, high)
    if return_steps:
        return edges, {"smoothed": smoothed, "magnitude": mag, "nms": nms}
    return edges


def normalize(img: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Chuẩn hoá đáp ứng biên về [0,1] theo phân vị (bền với vài pixel cực lớn)."""
    hi = float(np.percentile(img, pct)) if img.size else 0.0
    if hi <= 1e-9:
        hi = float(img.max()) or 1.0
    return np.clip(img / hi, 0, 1).astype(np.float32)
