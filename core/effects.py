"""Các pipeline hiệu ứng "tranh vẽ" ghép từ các thuật toán cơ bản.

Mỗi hiệu ứng khai báo schema tham số (để web tự sinh thanh trượt) và trả về
ảnh kết quả cùng các ảnh trung gian để người dùng thấy từng bước xử lý.

Các tham số không gian (sigma, bán kính, độ dày nét) được định nghĩa ở độ phân giải
tham chiếu REF_SIDE pixel và nhân với `scale = max(H, W) / REF_SIDE`, nên ảnh preview
và ảnh tải về độ phân giải cao cho kết quả trông giống nhau.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np

from . import edges as E
from . import filters as F
from . import morphology as M
from . import point_ops as P
from . import threshold as T
from .color import ensure_rgb, hsv_to_rgb, rgb_to_gray, rgb_to_hsv, rgb_to_ycbcr, ycbcr_to_rgb

REF_SIDE = 800


def scale_for(img: np.ndarray) -> float:
    return float(np.clip(max(img.shape[:2]) / REF_SIDE, 0.25, 4.0))


# =========================================================================== schema
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # 'float' | 'int' | 'choice' | 'bool'
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[tuple[str, str], ...] = ()
    help: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["options"] = [{"value": v, "label": lbl} for v, lbl in self.options]
        return d

    def coerce(self, value: Any) -> Any:
        """Ép kiểu + kẹp giá trị về miền hợp lệ; giá trị lỗi thì dùng mặc định."""
        try:
            if self.kind == "bool":
                if isinstance(value, str):
                    return value.lower() in ("1", "true", "yes", "on")
                return bool(value)
            if self.kind == "choice":
                return value if value in {v for v, _ in self.options} else self.default
            v = float(value)
            if not np.isfinite(v):
                return self.default
            if self.min is not None:
                v = max(self.min, v)
            if self.max is not None:
                v = min(self.max, v)
            return int(round(v)) if self.kind == "int" else v
        except (TypeError, ValueError):
            return self.default


@dataclass
class EffectResult:
    image: np.ndarray
    steps: list[tuple[str, np.ndarray]] = field(default_factory=list)


@dataclass(frozen=True)
class Effect:
    id: str
    name: str
    description: str
    params: tuple[Param, ...]
    fn: Callable[[np.ndarray, dict, float], EffectResult]

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def normalize(self, raw: dict | None) -> dict:
        raw = raw or {}
        return {p.key: p.coerce(raw.get(p.key, p.default)) for p in self.params}

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description,
                "params": [p.to_json() for p in self.params]}


SMOOTH_OPTS = (("none", "Không"), ("gaussian", "Gaussian"), ("median", "Trung vị"),
               ("bilateral", "Bilateral"), ("kuwahara", "Kuwahara"), ("anisotropic", "Perona-Malik"))
EDGE_OPTS = (("sobel", "Sobel"), ("prewitt", "Prewitt"), ("roberts", "Roberts"), ("laplacian", "Laplacian"),
             ("log", "LoG (Marr-Hildreth)"), ("canny", "Canny"), ("xdog", "XDoG"))
ENHANCE_OPTS = (("none", "Không"), ("stretch", "Giãn tương phản"), ("equalize", "Cân bằng histogram"))

P_ENHANCE = Param("enhance", "Tăng tương phản", "choice", "none", options=ENHANCE_OPTS,
                  help="Tiền xử lý mức xám, hữu ích với ảnh y tế/công nghiệp tương phản thấp")
P_SMOOTH = Param("smoothing", "Bộ làm mịn", "choice", "bilateral", options=SMOOTH_OPTS,
                 help="Khử nhiễu trước khi tìm biên; Bilateral/Kuwahara/Perona-Malik giữ được biên")
P_SMOOTH_S = Param("smooth_strength", "Độ mịn", "float", 2.0, 0.0, 6.0, 0.25, help="≈ sigma của bộ làm mịn (pixel)")
P_EDGE = Param("edge_op", "Toán tử biên", "choice", "sobel", options=EDGE_OPTS)
P_DETAIL = Param("detail", "Mức chi tiết biên", "float", 0.5, 0.0, 1.0, 0.05, help="Cao hơn → nhiều nét hơn")
P_SHADE = Param("shading", "Đổ bóng", "float", 0.4, 0.0, 1.0, 0.05,
                help="Tô xám vùng tối theo độ sáng gốc, tránh vùng tối bị trắng loang")
P_SHARP = Param("sharpness", "Độ nét", "float", 0.6, 0.0, 2.0, 0.05,
                help="Unsharp mask: g = f + k·(f − Gaussian(f)), làm rõ chi tiết")


# =========================================================================== khối dùng chung
def _gray(img: np.ndarray, p: dict) -> np.ndarray:
    return P.enhance(rgb_to_gray(img), p.get("enhance", "none"))


def pencil_dodge(gray: np.ndarray, blur_sigma: float, darkness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phác thảo chì bằng color dodge: dodge = gray / (1 - blur(1 - gray)).

    Vùng phẳng: blur(âm bản) ≈ âm bản => tỉ số ≈ 1 (trắng). Gần biên, âm bản bị làm mờ
    khác ảnh gốc => xuất hiện nét tối. Đây bản chất là một bộ phát hiện biên dạng tỉ số.
    """
    inv = P.negative(gray)
    blur = F.gaussian_blur(inv, blur_sigma)
    dodge = np.clip(gray / np.maximum(1.0 - blur, 1e-3), 0.0, 1.0)
    return P.gamma_correct(dodge, darkness), inv, blur


def tone_layer(gray: np.ndarray, amount: float, scale: float) -> np.ndarray:
    """Lớp đổ bóng: vùng tối được tô xám theo độ sáng đã làm mờ (như người vẽ tô bóng bằng chì).

    Color dodge biến vùng tối phẳng thành trắng (tỉ số ~0/0); nhân thêm lớp này để giữ khối sáng tối.
    T = 1 - amount * (1 - G_σ(gray)),  σ = 2 px ở độ phân giải tham chiếu.
    """
    if amount <= 0:
        return np.ones_like(gray)
    return (1.0 - amount * (1.0 - F.gaussian_blur(gray, 2.0 * scale))).astype(np.float32)


def edge_maps(gray: np.ndarray, op: str, detail: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tìm biên bằng toán tử `op`. Trả về (cường độ biên mềm [0,1], mặt nạ nhị phân, ảnh đáp ứng để hiển thị)."""
    detail = float(np.clip(detail, 0, 1))
    if op in ("sobel", "prewitt", "roberts", "laplacian"):
        if op == "laplacian":
            resp = np.abs(E.laplacian(F.gaussian_blur(gray, 0.8 * scale)))
        else:
            gx, gy = E.gradient(gray, op)
            resp = E.magnitude(gx, gy)
        norm = E.normalize(resp)
        # Ngưỡng Otsu trên độ lớn gradient, dịch lên/xuống theo mức chi tiết.
        thr = T.otsu_threshold(norm) * (1.6 - 1.0 * detail)
        mask = norm > thr
        soft = np.clip((norm - 0.5 * thr) / max(1e-3, 1.0 - 0.5 * thr), 0, 1) ** (1.6 - detail)
        return soft.astype(np.float32), mask, norm
    if op == "log":
        resp = E.log_filter(gray, 1.6 * scale)
        zc = E.zero_crossing_strength(resp)
        nz = zc[zc > 0]
        thr = float(np.quantile(nz, 0.95 - 0.4 * detail)) if nz.size else 1.0
        mask = zc > thr
        return mask.astype(np.float32), mask, E.normalize(np.abs(resp))
    if op == "canny":
        mask, st = E.canny(gray, sigma=1.0 * scale, low_ratio=0.5, high_quantile=0.95 - 0.35 * detail, return_steps=True)
        return mask.astype(np.float32), mask, np.sqrt(E.normalize(st["nms"]))  # căn bậc 2 để nét mảnh dễ nhìn
    if op == "xdog":
        x = E.xdog_lines(gray, sigma=1.0 * scale, k=1.6, eps=0.4 - 0.32 * detail, phi=8.0)
        soft = 1.0 - x
        return soft, soft > 0.5, x
    raise ValueError(f"Toán tử biên không hợp lệ: {op}")


def hatch_texture(shape: tuple[int, int], angle: float, length: float, seed: int = 7) -> np.ndarray:
    """Vân nét gạch bút chì: nhiễu ngẫu nhiên nhoè theo một hướng (motion blur), chuẩn hoá về [0,1]."""
    rng = np.random.default_rng(seed)
    noise = rng.random(shape, dtype=np.float32)
    tex = F.correlate2d(noise, F.motion_kernel(int(length), angle))
    return P.contrast_stretch(tex, 1, 99)


def apply_pencil_texture(shade: np.ndarray, amount: float, angle: float, scale: float) -> np.ndarray:
    """Điều biến vùng tối bằng nét gạch; vùng rất tối thêm lớp gạch chéo (cross-hatching)."""
    if amount <= 0:
        return shade
    L = max(5, int(round(17 * scale)))
    t1 = hatch_texture(shade.shape, angle, L, 7)
    t2 = hatch_texture(shade.shape, angle - 90.0, L, 11)
    dark = 1.0 - shade
    f = 1.0 + amount * (2.0 * t1 - 1.0)
    f *= 1.0 + amount * (2.0 * t2 - 1.0) * np.clip(dark * 2.0 - 0.6, 0, 1)
    grain = F.gaussian_blur(np.random.default_rng(3).random(shade.shape, dtype=np.float32), 0.7 * scale)
    out = 1.0 - dark * f
    return np.clip(out - amount * 0.06 * (grain - 0.5), 0, 1).astype(np.float32)


def ink_hatching(gray: np.ndarray, amount: float, scale: float) -> np.ndarray:
    """Tô mảng tối bằng nét gạch như tranh vẽ mực: tối vừa -> gạch chéo một chiều, rất tối -> gạch chéo hai chiều,
    tối nhất -> mực đặc. Ngưỡng lấy từ độ sáng đã làm mịn; amount = 0 thì tắt."""
    if amount <= 0:
        return np.ones_like(gray)
    g = F.gaussian_blur(gray, 1.2 * scale)
    h, w = g.shape
    y, x = np.mgrid[0:h, 0:w]
    period = max(3.0, 5.0 * scale)
    width = max(1.0, 1.4 * scale)
    d1 = np.mod(x + y, period) < width          # nét gạch 45°
    d2 = np.mod(x - y, period) < width          # nét gạch -45°
    t = 0.45 * amount
    out = np.ones_like(g)
    out[(g < t) & d1] = 0.0
    out[(g < 0.6 * t) & d2] = 0.0
    out[g < 0.3 * t] = 0.0
    return out


def toon_luminance(rgb: np.ndarray, levels: int, softness: float = 0.35) -> np.ndarray:
    """Lượng tử hoá mềm theo độ sáng (toon shading, Winnemöller 2006).

    Chỉ kênh Y (YCbCr) bị chia thành `levels` bậc; Cb/Cr giữ nguyên nên màu da, màu gỗ... không bị
    gộp vào một cụm chung như k-means toàn ảnh. Chuyển bậc mềm bằng tanh để không bị răng cưa:
    Q(y) = q + (Δ/2)·tanh(φ·(y − q)), q = bậc gần nhất, φ = 1 / (softness·Δ).
    """
    levels = max(2, int(levels))
    ycc = rgb_to_ycbcr(rgb)
    y = ycc[..., 0]
    step = 1.0 / (levels - 1)
    q = np.round(y / step) * step
    ycc[..., 0] = np.clip(q + 0.5 * step * np.tanh((y - q) / (softness * step)), 0, 1)
    return ycbcr_to_rgb(ycc)


def _ink(mask: np.ndarray) -> np.ndarray:
    """Mặt nạ biên -> ảnh nét đen trên nền trắng."""
    return np.where(mask, 0.0, 1.0).astype(np.float32)


def _boost_saturation(rgb: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return rgb
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 1)
    return hsv_to_rgb(hsv)


# =========================================================================== các hiệu ứng
def fx_pencil(img: np.ndarray, p: dict, s: float) -> EffectResult:
    gray = _gray(img, p)
    sm = F.smooth(gray, p["smoothing"], p["smooth_strength"], s)
    dodge, inv, blur = pencil_dodge(sm, p["blur"] * s, p["darkness"])
    out = F.unsharp_mask(dodge * tone_layer(sm, p["shading"], s), p["sharpness"], 1.0 * s)
    return EffectResult(out, [("Ảnh mức xám", gray), ("Làm mịn", sm), ("Âm bản", inv),
                              ("Âm bản làm mờ Gaussian", blur), ("Color dodge + gamma", dodge), ("Làm nét (unsharp)", out)])


def fx_sketch(img: np.ndarray, p: dict, s: float) -> EffectResult:
    gray = _gray(img, p)
    sm = F.smooth(gray, p["smoothing"], p["smooth_strength"], s)
    base, _, _ = pencil_dodge(sm, p["blur"] * s, p["darkness"])
    soft, mask, resp = edge_maps(sm, p["edge_op"], p["detail"], s)
    if p["edge_op"] in ("canny", "log"):
        soft = F.gaussian_blur(M.thicken(mask, 0.6 * s).astype(np.float32), 0.6 * s)
    shade = base * (1.0 - p["edge_strength"] * soft) * tone_layer(sm, p["shading"], s)
    shade = F.unsharp_mask(shade, p["sharpness"], 1.0 * s)
    out = apply_pencil_texture(shade, p["texture"], p["hatch_angle"], s)
    return EffectResult(out, [("Ảnh mức xám", gray), ("Làm mịn", sm), ("Phác thảo color dodge", base),
                              ("Bản đồ biên", resp), ("Kết hợp biên", shade), ("Thêm vân bút chì", out)])


def fx_lineart(img: np.ndarray, p: dict, s: float) -> EffectResult:
    gray = _gray(img, p)
    sm = F.smooth(gray, p["smoothing"], p["smooth_strength"], s)
    soft, mask, resp = edge_maps(sm, p["edge_op"], p["detail"], s)
    if p["clean"]:
        mask = M.remove_isolated(mask, 2)
    thick = M.thicken(mask, p["thickness"] * s)
    out = _ink(thick)
    if p["edge_op"] == "xdog" and p["thickness"] < 0.5:
        out = 1.0 - soft  # giữ nét mềm của XDoG
    fill = ink_hatching(sm, p["fill"], s)
    out = np.minimum(out, fill)
    # Khử răng cưa nhẹ cho nét mực.
    out = F.gaussian_blur(out, 0.35 * s)
    if p["invert"]:
        out = 1.0 - out
    return EffectResult(out, [("Ảnh mức xám", gray), ("Làm mịn", sm), ("Đáp ứng biên", resp),
                              ("Mặt nạ biên", _ink(mask)), ("Làm dày nét (giãn)", out)])


def fx_cartoon(img: np.ndarray, p: dict, s: float) -> EffectResult:
    rgb = ensure_rgb(img)
    sm = rgb
    for _ in range(p["passes"]):
        sm = F.bilateral_filter(sm, sigma_space=p["sigma_space"] * s, sigma_color=p["sigma_color"],
                                radius=max(1, int(round(2 * p["sigma_space"] * s))))
    if p["quantizer"] == "luminance":
        q = toon_luminance(sm, p["colors"])
    elif p["quantizer"] == "kmeans":
        q = P.kmeans_colors(sm, p["colors"])
    else:
        q = P.quantize_uniform(sm, max(2, round(p["colors"] ** (1 / 3) + 1)))
    q = F.unsharp_mask(_boost_saturation(q, p["saturation"]), p["sharpness"], 1.0 * s)
    gray = P.enhance(rgb_to_gray(rgb), p["enhance"])
    if p["edge_op"] == "adaptive":
        g = F.median_filter(gray, 2 * max(1, int(round(2 * s))) + 1)
        mask = T.adaptive_mean_threshold(g, block=2 * int(round(5 * s)) + 1, c=0.01 + 0.06 * (1 - p["detail"]))
        mask = M.remove_isolated(mask, 3)
    else:
        _, mask, _ = edge_maps(F.gaussian_blur(gray, 0.5 * s), p["edge_op"], p["detail"], s)
        mask = M.remove_isolated(M.remove_isolated(mask, 2), 2)  # bỏ chấm vụn (vân lông, nhiễu)
    mask = M.thicken(mask, p["thickness"] * s)
    lines = F.gaussian_blur(_ink(mask), 0.35 * s)
    out = q * lines[..., None]
    return EffectResult(out, [("Bilateral lặp", sm), ("Lượng tử hoá màu", q), ("Mặt nạ biên", lines), ("Kết quả", out)])


def fx_color_pencil(img: np.ndarray, p: dict, s: float) -> EffectResult:
    rgb = ensure_rgb(img)
    gray = _gray(rgb, p)
    sm = F.smooth(gray, "bilateral", 1.0, s)
    base, _, _ = pencil_dodge(sm, p["blur"] * s, p["darkness"])
    soft, _, resp = edge_maps(sm, "sobel", 0.5, s)
    shade = base * (1.0 - p["edge_strength"] * soft) * tone_layer(sm, p["shading"], s)
    shade = F.unsharp_mask(shade, p["sharpness"], 1.0 * s)
    shade = apply_pencil_texture(shade, p["texture"], 45.0, s)
    # Ghép: độ sáng Y lấy từ nét chì, sắc độ Cb/Cr lấy từ ảnh gốc (giảm/tăng theo `color`).
    ycc = rgb_to_ycbcr(F.gaussian_blur(rgb, 0.5 * s))
    ycc[..., 0] = shade
    ycc[..., 1:] = 0.5 + (ycc[..., 1:] - 0.5) * p["color"]
    out = ycbcr_to_rgb(ycc)
    return EffectResult(out, [("Ảnh mức xám", gray), ("Phác thảo chì (kênh Y)", shade),
                              ("Bản đồ biên Sobel", resp), ("Ghép sắc độ Cb/Cr", out)])


def fx_oil(img: np.ndarray, p: dict, s: float) -> EffectResult:
    rgb = ensure_rgb(img)
    k = F.kuwahara_filter(rgb, max(1, int(round(p["radius"] * s))))
    q = P.quantize_uniform(k, p["levels"])
    q = _boost_saturation(q, p["saturation"])
    # Làm sắc nét bằng Laplacian: g = f - c * ∇²f (slide tr.33-34).
    lap = np.stack([E.laplacian(q[..., c]) for c in range(3)], axis=-1)
    out = np.clip(q - p["sharpen"] * lap, 0, 1)
    return EffectResult(out, [("Lọc Kuwahara", k), ("Lượng tử hoá", q), ("Làm sắc nét Laplacian", out)])


EFFECTS: dict[str, Effect] = {e.id: e for e in [
    Effect("pencil", "Phác thảo chì", "Color dodge giữa ảnh xám và âm bản đã làm mờ: nét chì mềm, nhanh.", (
        P_ENHANCE,
        replace(P_SMOOTH, default="none"),
        replace(P_SMOOTH_S, default=1.0),
        Param("blur", "Độ rộng nét (σ blur)", "float", 6.0, 1.0, 40.0, 0.5),
        Param("darkness", "Độ đậm (gamma)", "float", 1.8, 0.5, 5.0, 0.1),
        P_SHADE,
        P_SHARP,
    ), fx_pencil),
    Effect("sketch", "Chì + biên + vân giấy", "Phác thảo chì kết hợp bản đồ biên và nét gạch bút chì.", (
        P_ENHANCE, P_SMOOTH, replace(P_SMOOTH_S, default=1.0), P_EDGE, P_DETAIL,
        Param("edge_strength", "Độ đậm biên", "float", 0.6, 0.0, 1.0, 0.05),
        Param("blur", "Độ rộng nét (σ blur)", "float", 7.0, 1.0, 40.0, 0.5),
        Param("darkness", "Độ đậm (gamma)", "float", 1.5, 0.5, 5.0, 0.1),
        P_SHADE,
        P_SHARP,
        Param("texture", "Vân bút chì", "float", 0.2, 0.0, 1.0, 0.05),
        Param("hatch_angle", "Góc nét gạch (°)", "float", 45.0, 0.0, 180.0, 5.0),
    ), fx_sketch),
    Effect("lineart", "Nét mực (line art)", "Chỉ giữ đường biên: nét đen trên nền trắng như tranh vẽ mực.", (
        P_ENHANCE, P_SMOOTH, replace(P_SMOOTH_S, default=0.5),
        replace(P_EDGE, default="xdog"),
        replace(P_DETAIL, default=0.8),
        Param("thickness", "Độ dày nét", "float", 0.0, 0.0, 4.0, 0.25, help="Bán kính phép giãn (pixel); 0 = nét XDoG tự nhiên"),
        Param("fill", "Tô mảng tối", "float", 0.5, 0.0, 1.0, 0.05, help="Gạch chéo các vùng tối như vẽ mực"),
        Param("clean", "Lọc điểm nhiễu", "bool", True),
        Param("invert", "Nét trắng nền đen", "bool", False),
    ), fx_lineart),
    Effect("cartoon", "Hoạt hình", "Bilateral lặp + lượng tử hoá màu + viền đen.", (
        P_ENHANCE,
        Param("passes", "Số lần bilateral", "int", 1, 1, 4, 1),
        Param("sigma_space", "σ không gian", "float", 1.5, 1.0, 5.0, 0.25),
        Param("sigma_color", "σ màu", "float", 0.1, 0.02, 0.4, 0.01),
        Param("quantizer", "Lượng tử hoá", "choice", "luminance",
              options=(("luminance", "Theo độ sáng"), ("kmeans", "K-means"), ("uniform", "Đều"))),
        Param("colors", "Số màu / bậc sáng", "int", 8, 2, 24, 1),
        Param("saturation", "Độ bão hoà", "float", 1.2, 0.0, 2.0, 0.05),
        Param("edge_op", "Viền", "choice", "xdog",
              options=(("xdog", "XDoG"), ("adaptive", "Ngưỡng thích nghi"), ("canny", "Canny"), ("sobel", "Sobel"))),
        replace(P_DETAIL, default=0.6),
        Param("thickness", "Độ dày viền", "float", 0.0, 0.0, 4.0, 0.25),
        replace(P_SHARP, default=0.4),
    ), fx_cartoon),
    Effect("color_pencil", "Chì màu", "Nét chì làm kênh độ sáng Y, giữ sắc độ Cb/Cr của ảnh gốc.", (
        P_ENHANCE,
        Param("blur", "Độ rộng nét (σ blur)", "float", 6.0, 1.0, 40.0, 0.5),
        Param("darkness", "Độ đậm (gamma)", "float", 1.4, 0.5, 5.0, 0.1),
        Param("color", "Lượng màu", "float", 1.0, 0.0, 2.0, 0.05),
        Param("edge_strength", "Độ đậm biên", "float", 0.4, 0.0, 1.0, 0.05),
        P_SHADE,
        P_SHARP,
        Param("texture", "Vân bút chì", "float", 0.2, 0.0, 1.0, 0.05),
    ), fx_color_pencil),
    Effect("oil", "Sơn dầu", "Lọc Kuwahara bảo toàn biên + lượng tử hoá + làm sắc nét Laplacian.", (
        Param("radius", "Bán kính cọ", "float", 2.0, 1.0, 12.0, 0.5),
        Param("levels", "Số mức màu", "int", 32, 3, 48, 1),
        Param("saturation", "Độ bão hoà", "float", 1.25, 0.0, 2.0, 0.05),
        Param("sharpen", "Làm sắc nét", "float", 0.6, 0.0, 1.5, 0.05),
    ), fx_oil),
]}


# Preset theo loại ảnh: ghi đè tham số mặc định (chỉ áp cho khoá mà hiệu ứng có).
CATEGORIES = {
    "nature": ("Ảnh tự nhiên", {}),
    "medical": ("Ảnh y tế", {"enhance": "stretch", "smoothing": "bilateral", "smooth_strength": 1.5, "fill": 0.0,
                             "detail": 0.6, "edge_strength": 0.7, "darkness": 1.4}),
    "industrial": ("Ảnh công nghiệp", {"enhance": "stretch", "smoothing": "median", "smooth_strength": 1.5,
                                      "detail": 0.65, "edge_op": "canny"}),
}


def preset_params(effect_id: str, category: str) -> dict:
    eff = EFFECTS[effect_id]
    over = CATEGORIES.get(category, ("", {}))[1]
    return eff.normalize({**eff.defaults(), **{k: v for k, v in over.items() if k in eff.defaults()}})


def run_effect(effect_id: str, img: np.ndarray, params: dict | None = None) -> EffectResult:
    if effect_id not in EFFECTS:
        raise ValueError(f"Hiệu ứng không tồn tại: {effect_id}")
    eff = EFFECTS[effect_id]
    p = eff.normalize(params)
    img = img.astype(np.float32, copy=False)
    res = eff.fn(img, p, scale_for(img))
    res.image = np.clip(res.image, 0, 1).astype(np.float32)
    return res


# =========================================================================== so sánh
def compare_filters(img: np.ndarray, strength: float = 2.0) -> list[tuple[str, np.ndarray]]:
    """Đặt cạnh nhau các bộ làm mịn để thấy bộ lọc nào giữ được biên."""
    s = scale_for(img)
    out = [("Ảnh gốc", img)]
    for m, label in [("gaussian", "Gaussian"), ("median", "Trung vị"), ("bilateral", "Bilateral"),
                     ("kuwahara", "Kuwahara"), ("anisotropic", "Perona-Malik")]:
        out.append((label, np.clip(F.smooth(img, m, strength, s), 0, 1)))
    return out


def compare_edges(img: np.ndarray, smooth_sigma: float = 1.0) -> list[tuple[str, np.ndarray]]:
    """Đặt cạnh nhau các toán tử biên (giống slide 56-58): biên trắng trên nền đen."""
    s = scale_for(img)
    gray = F.gaussian_blur(rgb_to_gray(img), smooth_sigma * s)
    out = [("Ảnh xám (đã làm mịn)", gray)]
    for op in ("roberts", "prewitt", "sobel"):
        gx, gy = E.gradient(gray, op)
        out.append((op.capitalize(), E.normalize(E.magnitude(gx, gy))))
    out.append(("Laplacian |∇²f|", E.normalize(np.abs(E.laplacian(gray)))))
    out.append(("LoG zero-crossing", E.marr_hildreth(rgb_to_gray(img), 2.0 * s, 0.04)[0].astype(np.float32)))
    out.append(("Canny", E.canny(rgb_to_gray(img), 1.4 * s, 0.5, 0.2).astype(np.float32)))
    out.append(("XDoG", edge_maps(gray, "xdog", 0.5, s)[0]))
    return out
