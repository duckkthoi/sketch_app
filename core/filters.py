"""Bộ lọc không gian: tích chập, làm mịn và lọc bảo toàn biên (Chương 2 + mở rộng).

Mọi phép lọc đều cài đặt bằng NumPy theo kiểu "dịch ảnh": thay vì lặp từng pixel,
ta lặp qua từng phần tử của cửa sổ lọc và cộng dồn cả ảnh đã dịch. Số phép toán vẫn
là H*W*k^2 nhưng mỗi phép chạy trên toàn mảng nên nhanh hơn vòng lặp Python hàng trăm lần.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .color import rgb_to_gray


# --------------------------------------------------------------------------- padding
def pad(img: np.ndarray, ry: int, rx: int | None = None, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """Đệm biên ảnh. mode: 'reflect' (đối xứng không lặp pixel biên), 'edge', 'constant' (zero padding)."""
    rx = ry if rx is None else rx
    widths = ((ry, ry), (rx, rx)) + ((0, 0),) * (img.ndim - 2)
    if mode == "constant":
        return np.pad(img, widths, mode="constant", constant_values=cval)
    if mode == "reflect" and (img.shape[0] <= ry or img.shape[1] <= rx):
        mode = "symmetric"  # ảnh quá nhỏ so với cửa sổ
    return np.pad(img, widths, mode=mode)


# --------------------------------------------------------------------------- tích chập
def correlate2d(img: np.ndarray, kernel: np.ndarray, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """Tương quan 2D: g(x,y) = sum_{s,t} w(s,t) f(x+s, y+t). Kernel phải có kích thước lẻ.

    Ảnh màu (H,W,C) được lọc độc lập trên từng kênh.
    """
    kernel = np.asarray(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel phải có kích thước lẻ")
    ry, rx = kh // 2, kw // 2
    h, w = img.shape[:2]
    p = pad(img.astype(np.float32, copy=False), ry, rx, mode, cval)
    out = np.zeros(img.shape, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            k = kernel[i, j]
            if k != 0.0:
                out += k * p[i:i + h, j:j + w]
    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """Tích chập 2D = tương quan với kernel đã lật 180 độ."""
    return correlate2d(img, np.asarray(kernel)[::-1, ::-1], mode, cval)


def correlate1d(img: np.ndarray, kernel: np.ndarray, axis: int, mode: str = "reflect") -> np.ndarray:
    """Tương quan 1D theo một trục (dùng cho bộ lọc tách được)."""
    kernel = np.asarray(kernel, dtype=np.float32)
    r = len(kernel) // 2
    n = img.shape[axis]
    p = pad(img.astype(np.float32, copy=False), r if axis == 0 else 0, r if axis == 1 else 0, mode)
    out = np.zeros(img.shape, dtype=np.float32)
    for i, k in enumerate(kernel):
        if k != 0.0:
            out += k * (p[i:i + n] if axis == 0 else p[:, i:i + n])
    return out


# --------------------------------------------------------------------------- Gaussian
def gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Kernel Gaussian 1D chuẩn hoá, bán kính ceil(3*sigma) (cửa sổ ~6σ như slide)."""
    r = max(1, int(truncate * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def gaussian_kernel2d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    k = gaussian_kernel1d(sigma, truncate)
    return np.outer(k, k)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Lọc thông thấp Gaussian. Vì G(x,y)=G(x)G(y) nên tách thành 2 lần lọc 1D: O(k) thay vì O(k^2)."""
    if sigma <= 0.05:
        return img.astype(np.float32, copy=True)
    if sigma >= 6.0:
        return _gaussian_by_boxes(img, sigma)
    k = gaussian_kernel1d(sigma)
    return correlate1d(correlate1d(img, k, axis=0), k, axis=1)


def _gaussian_by_boxes(img: np.ndarray, sigma: float, n: int = 3) -> np.ndarray:
    """Xấp xỉ Gaussian bằng n lần lọc hộp (định lý giới hạn trung tâm).

    Mỗi lần lọc hộp dùng ảnh tích phân nên chi phí không phụ thuộc sigma; dùng cho sigma lớn
    (ví dụ bước làm mờ của hiệu ứng phác thảo chì). Chọn độ rộng hộp theo Kovesi (2010)
    để tổng phương sai đúng bằng sigma^2.
    """
    w_ideal = math.sqrt(12.0 * sigma * sigma / n + 1.0)
    wl = int(math.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    m = round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4))
    out = img.astype(np.float32, copy=False)
    for i in range(n):
        out = box_filter(out, wl if i < m else wu)
    return out


# --------------------------------------------------------------------------- ảnh tích phân
class _Integral:
    """Ảnh tích phân (summed-area table) để tính tổng/trung bình trên hình chữ nhật trong O(1)/pixel."""

    def __init__(self, img: np.ndarray, r: int):
        self.h, self.w = img.shape[:2]
        self.r = r
        p = pad(img.astype(np.float64), r, r, "reflect")
        s = np.zeros((p.shape[0] + 1, p.shape[1] + 1) + p.shape[2:], dtype=np.float64)
        s[1:, 1:] = p.cumsum(0).cumsum(1)
        self.s = s

    def mean(self, dy0: int, dy1: int, dx0: int, dx1: int) -> np.ndarray:
        """Trung bình trên vùng hàng [y+dy0, y+dy1], cột [x+dx0, x+dx1] (bao gồm 2 đầu) cho mọi (y,x)."""
        r, h, w, s = self.r, self.h, self.w, self.s
        y0, y1, x0, x1 = r + dy0, r + dy1 + 1, r + dx0, r + dx1 + 1
        tot = s[y1:y1 + h, x1:x1 + w] - s[y0:y0 + h, x1:x1 + w] - s[y1:y1 + h, x0:x0 + w] + s[y0:y0 + h, x0:x0 + w]
        return (tot / ((dy1 - dy0 + 1) * (dx1 - dx0 + 1))).astype(np.float32)


def box_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Lọc trung bình cửa sổ size x size (dùng ảnh tích phân nên tốc độ không phụ thuộc size)."""
    r = max(0, int(size) // 2)
    if r == 0:
        return img.astype(np.float32, copy=True)
    return _Integral(img, r).mean(-r, r, -r, r)


# --------------------------------------------------------------------------- trung vị
def median_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Lọc trung vị: thay pixel bằng trung vị cửa sổ. Khử nhiễu muối tiêu mà ít làm nhoè biên."""
    size = int(size) | 1  # ép về số lẻ
    if size < 3:
        return img.astype(np.float32, copy=True)
    if img.ndim == 3:
        return np.stack([median_filter(img[..., c], size) for c in range(img.shape[2])], axis=-1)
    r = size // 2
    h, w = img.shape
    win = sliding_window_view(pad(img.astype(np.float32), r), (size, size))
    out = np.empty((h, w), dtype=np.float32)
    rows = max(1, 3_000_000 // (w * size * size))  # xử lý theo khối hàng để giới hạn bộ nhớ
    for y in range(0, h, rows):
        blk = win[y:y + rows].reshape(-1, w, size * size)
        out[y:y + rows] = np.median(blk, axis=-1)
    return out


# --------------------------------------------------------------------------- Bilateral
def bilateral_filter(
    img: np.ndarray,
    sigma_space: float = 3.0,
    sigma_color: float = 0.1,
    radius: int | None = None,
    step: int | None = None,
) -> np.ndarray:
    """Lọc song phương (bilateral filter) - bộ lọc làm mịn bảo toàn biên.

    BF[I]_p = 1/W_p * sum_q G_s(||p-q||) * G_r(|I_p - I_q|) * I_q

    - G_s: Gaussian theo khoảng cách không gian (sigma_space, đơn vị pixel).
    - G_r: Gaussian theo chênh lệch cường độ (sigma_color, đơn vị [0,1]).
    Pixel bên kia biên có cường độ khác xa nên G_r ~ 0 => biên không bị nhoè.
    Với ảnh màu, trọng số G_r tính trên độ sáng (joint bilateral theo kênh Y) để giảm chi phí.
    `step` > 1 lấy mẫu thưa cửa sổ. Mặc định step = ceil(radius / 6) để số điểm mẫu trong cửa sổ
    luôn <= ~113, giữ thời gian chạy ổn định kể cả khi bán kính lớn hoặc ảnh độ phân giải cao.
    """
    sigma_space = max(0.5, float(sigma_space))
    sigma_color = max(1e-3, float(sigma_color))
    if radius is None:
        radius = max(1, int(round(2.0 * sigma_space)))
    step = max(1, math.ceil(radius / 6)) if step is None else max(1, int(step))
    img = img.astype(np.float32, copy=False)
    guide = rgb_to_gray(img)
    h, w = guide.shape
    # Layout kênh-trước (C,H,W) để nhân broadcast trọng số (H,W) liên tục trong bộ nhớ.
    chw = np.ascontiguousarray(np.moveaxis(img, -1, 0)) if img.ndim == 3 else img[None]
    p_img = np.pad(chw, ((0, 0), (radius, radius), (radius, radius)), mode="reflect" if min(h, w) > radius else "symmetric")
    p_guide = pad(guide, radius)
    acc = np.zeros_like(chw)
    buf = np.empty_like(chw)
    wsum = np.zeros((h, w), dtype=np.float32)
    inv_c = np.float32(-0.5 / (sigma_color * sigma_color))
    tmp = np.empty((h, w), dtype=np.float32)
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            d2 = dy * dy + dx * dx
            if d2 > radius * radius:
                continue  # cửa sổ hình tròn
            ys, xs = slice(radius + dy, radius + dy + h), slice(radius + dx, radius + dx + w)
            # tmp = exp(-(I_q - I_p)^2 / 2σc^2 - d^2 / 2σs^2)
            np.subtract(p_guide[ys, xs], guide, out=tmp)
            np.square(tmp, out=tmp)
            tmp *= inv_c
            tmp += np.float32(-d2 / (2.0 * sigma_space * sigma_space))
            np.exp(tmp, out=tmp)
            wsum += tmp
            np.multiply(p_img[:, ys, xs], tmp, out=buf)
            acc += buf
    acc /= wsum
    return np.moveaxis(acc, 0, -1) if img.ndim == 3 else acc[0]


# --------------------------------------------------------------------------- Kuwahara
def kuwahara_filter(img: np.ndarray, radius: int = 4) -> np.ndarray:
    """Lọc Kuwahara - bảo toàn biên, cho hiệu ứng tranh sơn dầu.

    Chia lân cận (2r+1)x(2r+1) thành 4 góc phần tư chồng nhau kích thước (r+1)x(r+1).
    Pixel nhận giá trị trung bình của góc phần tư có phương sai (độ sáng) nhỏ nhất,
    tức là vùng đồng nhất nhất => không lấy trung bình băng qua biên.
    """
    r = max(1, int(radius))
    img = img.astype(np.float32, copy=False)
    gray = rgb_to_gray(img)
    ig = _Integral(gray, r)
    ig2 = _Integral(gray * gray, r)
    ic = _Integral(img, r) if img.ndim == 3 else ig
    quads = [(-r, 0, -r, 0), (-r, 0, 0, r), (0, r, -r, 0), (0, r, 0, r)]
    variances, means = [], []
    for q in quads:
        m = ig.mean(*q)
        variances.append(ig2.mean(*q) - m * m)
        means.append(ic.mean(*q))
    best = np.argmin(np.stack(variances, axis=0), axis=0)
    stacked = np.stack(means, axis=0)
    if img.ndim == 3:
        return np.take_along_axis(stacked, best[None, ..., None].repeat(img.shape[2], -1), axis=0)[0]
    return np.take_along_axis(stacked, best[None], axis=0)[0]


# --------------------------------------------------------------------------- Perona-Malik
def anisotropic_diffusion(img: np.ndarray, iterations: int = 10, kappa: float = 0.08, lam: float = 0.2) -> np.ndarray:
    """Khuếch tán bất đẳng hướng Perona-Malik: làm mịn trong vùng, dừng khuếch tán tại biên.

    I_{t+1} = I_t + λ * sum_{4 hướng} g(∇I) ∇I, với g(x) = exp(-(x/κ)^2).
    Hệ số dẫn g tính trên độ sáng để mọi kênh màu dùng chung một bản đồ biên.
    """
    out = img.astype(np.float32, copy=True)
    for _ in range(int(iterations)):
        p = pad(out, 1, 1, "edge")
        g = rgb_to_gray(p)
        c = g[1:-1, 1:-1]
        diffs = (p[:-2, 1:-1] - p[1:-1, 1:-1], p[2:, 1:-1] - p[1:-1, 1:-1],
                 p[1:-1, :-2] - p[1:-1, 1:-1], p[1:-1, 2:] - p[1:-1, 1:-1])
        gd = (g[:-2, 1:-1] - c, g[2:, 1:-1] - c, g[1:-1, :-2] - c, g[1:-1, 2:] - c)
        upd = np.zeros_like(out)
        for d, gdi in zip(diffs, gd):
            cond = np.exp(-(gdi / kappa) ** 2)
            upd += (cond[..., None] if out.ndim == 3 else cond) * d
        out += lam * upd
    return out


# --------------------------------------------------------------------------- tiện ích
def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Kernel nhoè chuyển động: đoạn thẳng dài `length` theo góc `angle_deg` (dùng tạo nét gạch bút chì)."""
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    a = math.radians(angle_deg)
    for t in np.linspace(-c, c, length * 3):
        x = int(round(c + t * math.cos(a)))
        y = int(round(c - t * math.sin(a)))
        k[y, x] = 1.0
    return k / k.sum()


def unsharp_mask(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Làm nét: g = f + k·(f − G_σ * f). Phần (f − G_σ * f) là chi tiết tần số cao (gần −∇²f)."""
    if amount <= 0:
        return img
    return np.clip(img + amount * (img - gaussian_blur(img, sigma)), 0, 1).astype(np.float32)


def smooth(img: np.ndarray, method: str, strength: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """Chọn bộ làm mịn theo tên. `strength` ~ sigma (pixel ở độ phân giải preview), `scale` để nhân theo kích thước ảnh."""
    s = max(0.0, strength) * scale
    if method == "none" or s <= 0:
        return img.astype(np.float32, copy=True)
    if method == "gaussian":
        return gaussian_blur(img, s)
    if method == "median":
        return median_filter(img, 2 * min(4, max(1, int(round(s)))) + 1)  # tối đa 9x9 để giới hạn thời gian
    if method == "bilateral":
        return bilateral_filter(img, sigma_space=s, sigma_color=0.12, radius=max(1, int(round(2 * s))))
    if method == "kuwahara":
        return kuwahara_filter(img, max(1, int(round(s * 1.5))))
    if method == "anisotropic":
        return anisotropic_diffusion(img, iterations=max(1, int(round(4 * strength))), kappa=0.06)
    raise ValueError(f"Bộ làm mịn không hợp lệ: {method}")
