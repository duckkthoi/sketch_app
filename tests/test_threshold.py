"""Kiểm tra phân ngưỡng, gồm ví dụ Otsu (Slide5 tr.80) và đẳng liệu (tr.85)."""
import numpy as np

from core import threshold as T
from core.point_ops import histogram

SLIDE_OTSU_I = np.array([
    [0, 1, 2, 3, 4, 5],
    [0, 0, 1, 2, 3, 4],
    [0, 0, 0, 1, 2, 3],
    [0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 1],
])


def test_otsu_slide_example():
    hist = np.bincount(SLIDE_OTSU_I.ravel(), minlength=6)
    k, sb = T.otsu_from_hist(hist)
    assert k == 1                   # "Như vậy ngưỡng T = 1"
    assert abs(sb - 1.68) < 0.01    # "với σ_B²(k) = 1.68"


def test_otsu_matches_bruteforce_within_class_variance(rng):
    vals = np.concatenate([rng.normal(60, 10, 4000), rng.normal(170, 20, 6000)])
    hist = np.bincount(np.clip(vals, 0, 255).astype(int), minlength=256)
    k, _ = T.otsu_from_hist(hist)
    z = np.arange(256)
    best, arg = np.inf, -1
    for t in range(1, 255):  # cực tiểu phương sai trong lớp <=> cực đại phương sai giữa lớp
        a, b = hist[:t + 1], hist[t + 1:]
        if a.sum() == 0 or b.sum() == 0:
            continue
        va = (a * (z[:t + 1] - (a * z[:t + 1]).sum() / a.sum()) ** 2).sum()
        vb = (b * (z[t + 1:] - (b * z[t + 1:]).sum() / b.sum()) ** 2).sum()
        if va + vb < best:
            best, arg = va + vb, t
    assert abs(k - arg) <= 1
    assert 90 < k < 140


def test_isodata_slide_example():
    p = np.array([0.500, 0.167, 0.133, 0.100, 0.067, 0.033])
    t1 = T.isodata_from_hist(p, t0=3.0, max_iter=1)
    assert abs(t1 - 2.574) < 0.01   # t1 = 2.574 trên slide
    t = T.isodata_from_hist(p, t0=3.0)
    assert int(t) == 2              # "Vậy ngưỡng t = 2"


def test_otsu_threshold_float_image(square):
    t = T.otsu_threshold(square)
    assert 0.2 <= t < 0.8
    mask = T.binarize(square, "otsu")
    assert mask[40, 40] and not mask[5, 5]


def test_triangle_threshold_on_skewed_hist():
    hist = np.zeros(256)
    hist[200:256] = np.linspace(1000, 50, 56)  # đỉnh nền sáng
    hist[20:120] = 15                            # đối tượng tối, đuôi dài
    t = T.triangle_from_hist(hist)
    assert 100 <= t <= 205


def test_adaptive_threshold_handles_uneven_light():
    grad = np.tile(np.linspace(0.3, 1.0, 100, dtype=np.float32), (60, 1))
    img = grad.copy()
    img[28:32, 10:90] -= 0.2  # nét chữ tối trên nền chiếu sáng không đều
    mask = T.adaptive_mean_threshold(img, 15, 0.05)
    assert mask[30, 15] and mask[30, 85]       # bắt được nét ở cả vùng tối và sáng
    assert not mask[10, 15] and not mask[10, 85]
    # Ngưỡng toàn cục Otsu thì thất bại ở một phía.
    g = T.binarize(img, "otsu")
    assert g[10, 15] == g[30, 15] or g[10, 85] == g[30, 85]


def test_histogram_counts():
    img = np.array([[0, 0.5, 1.0, 1.0]], np.float32)
    h = histogram(img)
    assert h.sum() == 4 and h[0] == 1 and h[255] == 2 and h[128] == 1
