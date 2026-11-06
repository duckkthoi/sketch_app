"""Đọc / ghi ảnh. Pillow chỉ được dùng ở đây cho việc giải mã, mã hoá file và đổi kích thước."""
from __future__ import annotations

import base64
import io
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

# Chặn "decompression bomb": ảnh có quá nhiều pixel sẽ bị từ chối.
Image.MAX_IMAGE_PIXELS = 60_000_000


class InvalidImageError(ValueError):
    """File tải lên không phải ảnh hợp lệ."""


def _pil_to_array(im: Image.Image) -> np.ndarray:
    """Chuyển ảnh Pillow về float32 [0,1]; giữ ảnh xám là 2D, ảnh màu là RGB."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # Ảnh y tế/khoa học 16-bit hoặc số thực: chuẩn hoá min-max về [0,1].
        a = np.asarray(im, dtype=np.float64)
        lo, hi = np.percentile(a, 0.1), np.percentile(a, 99.9)
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        return np.clip(a, 0, 1).astype(np.float32)
    if mode in ("1", "L"):
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    if mode in ("LA", "RGBA", "PA") or (mode == "P" and "transparency" in im.info):
        # Ghép nền trắng cho ảnh có kênh alpha.
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    # Ảnh màu nhưng thực chất là xám (R=G=B) thì trả về 2D cho gọn.
    if np.allclose(rgb[..., 0], rgb[..., 1], atol=1e-3) and np.allclose(rgb[..., 1], rgb[..., 2], atol=1e-3):
        return rgb[..., 0].copy()
    return rgb


def load_image(src: str | bytes | BinaryIO) -> np.ndarray:
    """Đọc ảnh từ đường dẫn, bytes hoặc file-like. Ném InvalidImageError nếu không phải ảnh."""
    try:
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(src)
        im = Image.open(src)
        im.load()
        im = ImageOps.exif_transpose(im)  # xoay đúng chiều theo EXIF của điện thoại
    except Image.DecompressionBombError as e:
        raise InvalidImageError("Ảnh quá lớn") from e
    except Exception as e:  # noqa: BLE001 - Pillow ném nhiều loại lỗi khác nhau
        raise InvalidImageError("File không phải ảnh hợp lệ") from e
    return _pil_to_array(im)


def trim_borders(img: np.ndarray, tol: float = 0.006, drift: float = 0.02, min_frac: float = 0.02) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Tự cắt viền phẳng (letterbox đen/trắng hoặc dải chuyển màu của ảnh chụp màn hình, video).

    Đi từ mép vào trong: một hàng/cột là viền nếu gần như phẳng (độ lệch chuẩn < tol) và trung bình
    chỉ thay đổi ít so với hàng/cột trước (< drift), nên bắt được cả viền gradient.
    Chỉ cắt khi viền dày >= min_frac kích thước và phần còn lại >= 40%.
    Trả về (ảnh đã cắt, (top, bottom, left, right)) hoặc (ảnh gốc, None).
    """
    g = img.mean(axis=-1) if img.ndim == 3 else img
    h, w = g.shape

    def run(lines: np.ndarray) -> int:
        std, mean = lines.std(axis=1), lines.mean(axis=1)
        n = 0
        while n < len(lines) and std[n] < tol and (n == 0 or abs(mean[n] - mean[n - 1]) < drift):
            n += 1
        return n

    top, bottom = run(g), run(g[::-1])
    left, right = run(g.T), run(g.T[::-1])
    top, bottom = (top if top >= min_frac * h else 0), (bottom if bottom >= min_frac * h else 0)
    left, right = (left if left >= min_frac * w else 0), (right if right >= min_frac * w else 0)
    if not (top or bottom or left or right):
        return img, None
    if h - top - bottom < 0.4 * h or w - left - right < 0.4 * w:
        return img, None
    return img[top:h - bottom, left:w - right].copy(), (top, bottom, left, right)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_pil(img: np.ndarray) -> Image.Image:
    a = to_uint8(img)
    return Image.fromarray(a, mode="L" if a.ndim == 2 else "RGB")


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """Thu nhỏ để cạnh dài nhất <= max_side (không phóng to)."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / s
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if img.ndim == 2:
        im = Image.fromarray(img.astype(np.float32), mode="F").resize(size, Image.Resampling.LANCZOS)
        return np.clip(np.asarray(im, dtype=np.float32), 0, 1)
    chans = [
        np.asarray(Image.fromarray(img[..., c].astype(np.float32), mode="F").resize(size, Image.Resampling.LANCZOS))
        for c in range(img.shape[2])
    ]
    return np.clip(np.stack(chans, axis=-1), 0, 1).astype(np.float32)


def encode(img: np.ndarray, fmt: str = "PNG", quality: int = 90) -> bytes:
    """Mã hoá ảnh numpy thành bytes (PNG / JPEG / WEBP)."""
    fmt = fmt.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    buf = io.BytesIO()
    im = to_pil(img)
    if fmt == "JPEG":
        im.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=True)
    elif fmt == "WEBP":
        im.save(buf, format="WEBP", quality=quality, method=4)
    else:
        im.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def to_data_url(img: np.ndarray, fmt: str = "WEBP", quality: int = 88) -> str:
    mime = {"WEBP": "image/webp", "PNG": "image/png", "JPEG": "image/jpeg"}[fmt.upper()]
    return f"data:{mime};base64," + base64.b64encode(encode(img, fmt, quality)).decode("ascii")
