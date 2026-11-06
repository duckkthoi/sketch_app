"""Kiểm tra phát hiện biên, gồm bài tập Slide5 tr.27-29."""
import numpy as np
from scipy import ndimage as ndi

from core import edges as E
from core.filters import correlate2d

# ---- Bài tập Slide5 tr.27: ảnh I, bộ lọc Hx, Hy; đệm 0 để giữ kích thước ảnh.
SLIDE_I = np.array([
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 15, 15, 15, 2, 2],
    [2, 2, 13, 12, 16, 2, 2],
    [2, 2, 15, 15, 15, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
], dtype=np.float32)
HX = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
HY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
# Đáp án Slide5 tr.29: I4 = |I1| + |I2|
SLIDE_I4 = np.array([
    [8, 6, 6, 6, 6, 6, 8],
    [6, 26, 39, 39, 39, 26, 6],
    [6, 35, 44, 38, 47, 41, 6],
    [6, 37, 36, 3, 36, 40, 6],
    [6, 35, 44, 38, 47, 41, 6],
    [6, 26, 39, 39, 39, 26, 6],
    [8, 6, 6, 6, 6, 6, 8],
], dtype=np.float32)


def test_slide_exercise_gradient_magnitude():
    i1 = correlate2d(SLIDE_I, HX, mode="constant")
    i2 = correlate2d(SLIDE_I, HY, mode="constant")
    i4 = np.abs(i1) + np.abs(i2)
    np.testing.assert_array_equal(i4, SLIDE_I4)
    # Các mặt nạ này chính là Prewitt.
    np.testing.assert_array_equal(HX, E.PREWITT_Y)
    np.testing.assert_array_equal(HY, E.PREWITT_X)


def test_slide_exercise_threshold_25_outlines_object():
    i4 = np.abs(correlate2d(SLIDE_I, HX, mode="constant")) + np.abs(correlate2d(SLIDE_I, HY, mode="constant"))
    edges = i4 > 25
    # Ngưỡng 25: vòng biên quanh đối tượng 3x3 (hàng/cột 1..5) là biên, tâm (3,3) và viền ảnh không phải biên.
    expected = np.zeros((7, 7), bool)
    expected[1:6, 1:6] = True
    expected[3, 3] = False
    expected[1, 1] = expected[1, 5] = expected[5, 1] = expected[5, 5] = True
    np.testing.assert_array_equal(edges, expected)


def test_sobel_matches_scipy(rng):
    img = rng.random((40, 40)).astype(np.float32)
    gx, gy = E.sobel(img)
    np.testing.assert_allclose(gx, ndi.sobel(img, axis=1, mode="mirror"), atol=1e-5)
    np.testing.assert_allclose(gy, ndi.sobel(img, axis=0, mode="mirror"), atol=1e-5)


def test_gradient_direction_of_vertical_edge(square):
    gx, gy = E.sobel(square)
    # Cạnh trái của hình vuông: cường độ tăng theo x -> gx > 0, hướng ~0°.
    assert gx[40, 20] > 0 and abs(gy[40, 20]) < 1e-6
    assert abs(E.direction(gx, gy)[40, 20]) < 1e-3


def test_roberts_and_prewitt_respond_on_edges_only(square):
    for op in ("roberts", "prewitt", "sobel"):
        gx, gy = E.gradient(square, op)
        mag = E.magnitude(gx, gy)
        assert mag[40, 40] < 1e-5 and mag[5, 5] < 1e-5
        assert mag[40, 19:21].max() > 0.3


def test_laplacian_zero_on_linear_ramp():
    ramp = np.tile(np.linspace(0, 1, 30, dtype=np.float32), (30, 1))
    lap = E.laplacian(ramp)
    assert np.abs(lap[2:-2, 2:-2]).max() < 1e-5


def test_log_kernel_sums_to_zero():
    k = E.log_kernel(1.5)
    assert abs(k.sum()) < 1e-5 and k[k.shape[0] // 2, k.shape[1] // 2] < 0


def test_marr_hildreth_finds_circle():
    y, x = np.mgrid[:80, :80]
    disk = (((x - 40) ** 2 + (y - 40) ** 2) < 20 ** 2).astype(np.float32)
    edges, _ = E.marr_hildreth(disk, 1.5, 0.05)
    ys, xs = np.nonzero(edges)
    r = np.hypot(xs - 40, ys - 40)
    assert len(r) > 80
    assert np.abs(r - 20).max() < 2.5


def test_canny_thin_and_well_localized(square, rng):
    noisy = np.clip(square + rng.normal(0, 0.02, square.shape), 0, 1).astype(np.float32)
    edges = E.canny(noisy, sigma=1.4)
    ys, xs = np.nonzero(edges)
    # Mỗi điểm biên cách đường biên thật (19.5 hoặc 59.5) tối đa 1.5 px.
    dist = np.minimum.reduce([np.abs(xs - 19.5), np.abs(xs - 59.5), np.abs(ys - 19.5), np.abs(ys - 59.5)])
    assert dist.max() <= 1.5
    # Biên mảnh: số pixel ~ chu vi (4*40), không bị dày gấp đôi.
    assert 140 <= edges.sum() <= 200
    # Không có biên giả trong vùng phẳng.
    assert not edges[30:50, 30:50].any() and not edges[:10, :10].any()


def test_hysteresis_keeps_weak_connected_only():
    nms = np.zeros((5, 9), np.float32)
    nms[2, 1] = 1.0           # điểm mạnh
    nms[2, 2:5] = 0.4         # chuỗi yếu nối với điểm mạnh
    nms[2, 7] = 0.4           # điểm yếu cô lập
    out = E.hysteresis(nms, low=0.3, high=0.8)
    assert out[2, 1:5].all()
    assert not out[2, 7]


def test_canny_quantile_mode(square):
    assert E.canny(square, 1.0, high_quantile=0.5).any()


def test_xdog_ranges(square):
    x = E.xdog(square, 1.0)
    assert x.min() >= 0 and x.max() <= 1
    xl = E.xdog_lines(square, 1.0)
    assert xl[40, 40] == 1.0 and xl.min() < 0.2  # nền phẳng trắng, có nét đậm ở biên


def test_zero_crossings_basic():
    resp = np.array([[1, 1, -1, -1]] * 3, np.float32)
    zc = E.zero_crossings(resp, 0.5)
    assert zc[:, 1].all() and zc[:, 2].all() and not zc[:, 0].any()
