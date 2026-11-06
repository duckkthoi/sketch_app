"""Sinh toàn bộ hình minh hoạ + số liệu cho báo cáo và slides từ code thật.

Chạy: python scripts/make_figures.py   ->  docs/figures/*.png, docs/figures/benchmark.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import edges as E  # noqa: E402
from core import filters as F  # noqa: E402
from core.color import rgb_to_gray  # noqa: E402
from core.effects import EFFECTS, compare_edges, compare_filters, preset_params, run_effect  # noqa: E402
from core.io_utils import load_image, resize_max, to_pil  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATHS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def sample(name: str, side: int = 800) -> np.ndarray:
    return resize_max(load_image(ROOT / "samples" / f"{name}.jpg"), side)


def grid(tiles: list[tuple[str, np.ndarray]], cols: int, tile_h: int = 300, path: str | Path = "", pad: int = 12) -> Path:
    """Ghép các ảnh thành lưới có nhãn bên dưới mỗi ảnh."""
    f = font(20)
    ims = []
    for label, a in tiles:
        im = to_pil(np.clip(a, 0, 1)).convert("RGB")
        im = im.resize((max(1, round(im.width * tile_h / im.height)), tile_h), Image.Resampling.LANCZOS)
        ims.append((label, im))
    tw = max(im.width for _, im in ims)
    lab_h = 34
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (tile_h + lab_h + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    for k, (label, im) in enumerate(ims):
        x = pad + (k % cols) * (tw + pad)
        y = pad + (k // cols) * (tile_h + lab_h + pad)
        sheet.paste(im, (x + (tw - im.width) // 2, y))
        d.rectangle([x + (tw - im.width) // 2, y, x + (tw + im.width) // 2 - 1, y + tile_h - 1], outline=(220, 215, 205))
        size = 20
        lf = f
        while d.textlength(label, font=lf) > tw - 6 and size > 11:  # co chữ cho vừa ô
            size -= 1
            lf = font(size)
        w = d.textlength(label, font=lf)
        d.text((x + (tw - w) / 2, y + tile_h + 6), label, fill=(30, 30, 30), font=lf)
    out = OUT / path
    sheet.save(out, optimize=True)
    print("saved", out.name, sheet.size)
    return out


def architecture() -> None:
    """Sơ đồ kiến trúc: trình duyệt -> Flask -> core NumPy; Flask <-> SQLite + file trên volume."""
    W, H = 1600, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    ink, muted, accent, soft = (31, 35, 40), (107, 106, 102), (194, 65, 12), (251, 229, 214)
    fb, fn, fs = font(30), font(23), font(20)

    def box(x, y, w, h, title, lines, fill=(248, 245, 238), outline=(200, 194, 182)):
        d.rounded_rectangle([x, y, x + w, y + h], 18, fill=fill, outline=outline, width=3)
        d.text((x + 22, y + 18), title, fill=ink, font=fb)
        for i, ln in enumerate(lines):
            d.text((x + 22, y + 66 + i * 32), ln, fill=muted, font=fn)

    def arrow(x1, y1, x2, y2, label="", dx=0, dy=-30):
        d.line([x1, y1, x2, y2], fill=accent, width=4)
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        for s in (-1, 1):
            d.line([x2, y2, x2 - 18 * math.cos(a + s * 0.45), y2 - 18 * math.sin(a + s * 0.45)], fill=accent, width=4)
        if label:
            tw = d.textlength(label, font=fs)
            d.text(((x1 + x2) / 2 - tw / 2 + dx, (y1 + y2) / 2 + dy), label, fill=accent, font=fs)

    box(40, 60, 420, 250, "Trình duyệt", ["Giao diện TypeScript (Vite)", "Thanh trượt tham số", "So sánh trước/sau, các bước", "Lịch sử, preset"])
    box(590, 60, 440, 250, "Flask API (gunicorn)", ["/api/upload, /api/process", "/api/download, /api/compare", "/api/results, /api/presets", "Rate limit, cookie ẩn danh"],
        fill=soft, outline=accent)
    box(1160, 60, 400, 250, "core/ (NumPy)", ["Mức xám, histogram", "Gaussian, trung vị, Bilateral", "Sobel, LoG, Canny, XDoG", "Otsu, hình thái học"])
    d.rounded_rectangle([560, 420, 1560, 720], 22, outline=(160, 154, 142), width=3)
    d.text((1300, 432), "Railway Volume /app/storage", fill=muted, font=fs)
    box(600, 480, 420, 210, "SQLite  app.db", ["images, results, presets", "lọc theo client_id", "WAL, tự dọn sau 7 ngày"])
    box(1100, 480, 420, 210, "File ảnh", ["uploads/  ảnh gốc PNG", "results/  kết quả PNG", "thumbs/   thumbnail WEBP"])
    arrow(460, 165, 590, 165, "JSON / ảnh")
    arrow(590, 225, 460, 225)
    arrow(1030, 185, 1160, 185, "gọi hàm")
    arrow(760, 310, 760, 480, "SQL", dx=36, dy=-14)
    arrow(950, 310, 1250, 480, "đọc/ghi", dx=70, dy=-40)
    d.text((40, 360), "Docker image 2 stage:", fill=ink, font=fn)
    d.text((40, 396), "1) node:24 build frontend", fill=muted, font=fn)
    d.text((40, 428), "2) python:3.12-slim + gunicorn", fill=muted, font=fn)
    d.text((40, 480), "Không dùng OpenCV:", fill=ink, font=fn)
    d.text((40, 516), "mọi thuật toán tự cài đặt", fill=muted, font=fn)
    d.text((40, 548), "bằng NumPy", fill=muted, font=fn)
    im.save(OUT / "architecture.png", optimize=True)
    print("saved architecture.png")


def main() -> None:
    architecture()
    bench: dict = {}

    # 1. Mỗi hiệu ứng trên 3 loại ảnh
    for cat, name in [("nature", "cat"), ("medical", "chest_xray"), ("industrial", "pcb")]:
        img = sample(f"{cat}/{name}")
        tiles = [("Ảnh gốc", img)]
        for eid, eff in EFFECTS.items():
            tiles.append((eff.name, run_effect(eid, img, preset_params(eid, cat)).image))
        grid(tiles, 4, 260, f"effects_{cat}.png")

    # 2. Các bước của pipeline chính
    img = sample("nature/portrait")
    for eid in ("pencil", "sketch", "lineart", "cartoon"):
        res = run_effect(eid, img, preset_params(eid, "nature"))
        grid([("Ảnh gốc", img)] + res.steps, 3 if len(res.steps) >= 5 else len(res.steps) + 1, 260, f"steps_{eid}.png")

    # 3. So sánh bộ lọc làm mịn trên ảnh có nhiễu Gaussian
    rng = np.random.default_rng(0)
    clean = sample("nature/mountain_lake", 480)
    noisy = np.clip(clean + rng.normal(0, 0.06, clean.shape), 0, 1).astype(np.float32)
    grid([("Ảnh nhiễu Gaussian σ=0.06", noisy)] + compare_filters(noisy)[1:], 3, 240, "compare_filters.png")

    # 4. So sánh toán tử biên
    grid(compare_edges(sample("medical/brain_mri", 480)), 4, 240, "compare_edges_mri.png")
    grid(compare_edges(sample("industrial/gears", 480)), 4, 220, "compare_edges_gears.png")

    # 5. Các bước Canny
    g = rgb_to_gray(sample("industrial/gears", 640))
    edges, st = E.canny(g, 1.4, 0.5, high_quantile=0.8, return_steps=True)
    grid([("1. Ảnh xám", g), ("2. Làm mịn Gaussian σ=1.4", st["smoothed"]),
          ("3. Độ lớn gradient Sobel", E.normalize(st["magnitude"])),
          ("4. Non-maximum suppression", np.sqrt(E.normalize(st["nms"]))),
          ("5. Ngưỡng kép + nối biên", edges.astype(np.float32))], 3, 260, "canny_steps.png")

    # 6. Vai trò của làm mịn với nhiễu muối tiêu trước khi tìm biên
    base = sample("medical/hand_xray", 520)
    sp = base.copy()
    m = rng.random(sp.shape[:2])
    sp[m < 0.03] = 0
    sp[m > 0.97] = 1
    tiles = [("Ảnh nhiễu muối tiêu 6%", sp)]
    for method, label in [("none", "Không làm mịn"), ("gaussian", "Gaussian"), ("median", "Trung vị"), ("bilateral", "Bilateral")]:
        p = {**preset_params("lineart", "medical"), "smoothing": method, "smooth_strength": 1.5,
             "edge_op": "canny", "detail": 0.5, "thickness": 1.0, "fill": 0.0}
        tiles.append((f"Line art + {label}", run_effect("lineart", sp, p).image))
    grid(tiles, 5, 280, "noise_smoothing.png")

    # 7. Ảnh hưởng tham số
    img = sample("nature/cat", 640)
    grid([(f"Chi tiết = {d}", run_effect("lineart", img, {"detail": d}).image) for d in (0.2, 0.5, 0.8)], 3, 240, "param_detail.png")
    grid([(f"σ blur = {b}", run_effect("pencil", img, {"blur": b}).image) for b in (3, 10, 30)], 3, 240, "param_blur.png")
    grid([(f"Toán tử: {op}", run_effect("lineart", img, {"edge_op": op}).image) for op in ("sobel", "log", "canny", "xdog")], 4, 220, "param_edge_op.png")

    # 8. Tiền xử lý mức xám cho ảnh y tế
    xr = sample("medical/chest_xray", 640)
    grid([(f"Tăng tương phản: {lbl}", run_effect("sketch", xr, {**preset_params("sketch", "medical"), "enhance": e}).image)
          for e, lbl in [("none", "không"), ("stretch", "giãn"), ("equalize", "cân bằng histogram")]], 3, 300, "enhance_medical.png")

    # 8b. Trước / sau khi cải thiện độ chi tiết (tham số cũ được mô phỏng lại)
    old = {
        "lineart": {"edge_op": "canny", "detail": 0.5, "thickness": 1.0, "smooth_strength": 2.0, "fill": 0.0},
        "cartoon": {"passes": 2, "sigma_space": 2.5, "quantizer": "kmeans", "edge_op": "adaptive", "detail": 0.5,
                    "thickness": 0.5, "sharpness": 0.0},
        "oil": {"radius": 4.0, "levels": 14, "sharpen": 0.3},
        "sketch": {"smooth_strength": 2.0, "blur": 12.0, "texture": 0.35, "shading": 0.0, "sharpness": 0.0},
    }
    img = sample("nature/portrait", 900)
    h, w = img.shape[:2]
    crop = (slice(int(0.05 * h), int(0.5 * h)), slice(int(0.15 * w), int(0.85 * w)))
    tiles = [("Ảnh gốc", img[crop])]
    for eid in ("sketch", "lineart", "cartoon", "oil"):
        tiles.append((f"{EFFECTS[eid].name}: cũ", run_effect(eid, img, old[eid]).image[crop]))
        tiles.append((f"{EFFECTS[eid].name}: mới", run_effect(eid, img).image[crop]))
    grid(tiles, 3, 260, "improve_detail.png")

    # 8c. Tự cắt viền letterbox (mô phỏng ảnh chụp màn hình video có viền đen + dải chuyển màu)
    from core.io_utils import trim_borders
    base = sample("nature/cat", 600)
    hh, ww = base.shape[:2]
    lb = np.zeros((hh, ww + 400, 3), np.float32)
    grad = np.linspace(0, 0.08, 80, dtype=np.float32)
    lb[:, 120:200] = grad[None, :, None]
    lb[:, ww + 200:ww + 280] = grad[::-1][None, :, None]
    lb[:, 200:200 + ww] = base
    trimmed, _ = trim_borders(lb)
    grid([("Ảnh có viền (letterbox)", lb), ("Sau khi tự cắt viền", trimmed),
          ("Phác thảo chì trên ảnh có viền", run_effect("pencil", lb).image),
          ("Phác thảo chì sau khi cắt viền", run_effect("pencil", trimmed).image)], 2, 260, "trim_borders.png")

    # 9. Benchmark thời gian
    img800 = sample("nature/mountain_lake", 800)
    img1600 = sample("nature/mountain_lake", 1600)
    for eid in EFFECTS:
        row = {}
        for lbl, im in (("800", img800), ("1600", img1600)):
            run_effect(eid, im)  # warm-up
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                run_effect(eid, im)
                ts.append((time.perf_counter() - t) * 1000)
            row[lbl] = round(float(np.median(ts)))
        bench[eid] = {"name": EFFECTS[eid].name, **row}
        print("bench", eid, row)
    g = rgb_to_gray(img800)
    algo = {}
    for lbl, fn in [("Gaussian σ=3", lambda: F.gaussian_blur(g, 3)), ("Gaussian σ=20 (3 hộp)", lambda: F.gaussian_blur(g, 20)),
                    ("Trung vị 5x5", lambda: F.median_filter(g, 5)), ("Bilateral r=5 (RGB)", lambda: F.bilateral_filter(img800, 2.5, 0.1, 5)),
                    ("Kuwahara r=4 (RGB)", lambda: F.kuwahara_filter(img800, 4)), ("Sobel", lambda: E.sobel(g)),
                    ("Canny", lambda: E.canny(g, 1.4)), ("LoG + zero-crossing", lambda: E.marr_hildreth(g, 2.0))]:
        fn()
        t = time.perf_counter()
        for _ in range(3):
            fn()
        algo[lbl] = round((time.perf_counter() - t) * 1000 / 3)
    out = {"image_800": list(img800.shape), "image_1600": list(img1600.shape), "effects_ms": bench, "algorithms_ms_800": algo}
    (OUT / "benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
