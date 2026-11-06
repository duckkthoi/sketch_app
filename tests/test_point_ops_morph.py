import numpy as np
from scipy import ndimage as ndi

from core import morphology as M
from core import point_ops as P
from core.color import hsv_to_rgb, rgb_to_gray, rgb_to_hsv, rgb_to_ycbcr, ycbcr_to_rgb
from core.io_utils import encode, load_image, resize_max


def test_gray_weights():
    px = np.array([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]], np.float32)
    np.testing.assert_allclose(rgb_to_gray(px)[0], [0.299, 0.587, 0.114], atol=1e-6)


def test_color_roundtrips(rng):
    img = rng.random((20, 20, 3)).astype(np.float32)
    np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(img)), img, atol=1e-5)
    np.testing.assert_allclose(ycbcr_to_rgb(rgb_to_ycbcr(img)), img, atol=2e-3)


def test_negative_gamma():
    x = np.linspace(0, 1, 11, dtype=np.float32)
    np.testing.assert_allclose(P.negative(x), 1 - x)
    assert (P.gamma_correct(x, 2.0) <= x + 1e-7).all()  # gamma > 1 làm tối


def test_equalize_spreads_histogram(rng):
    low = (0.4 + 0.1 * rng.random((64, 64))).astype(np.float32)  # tương phản thấp
    eq = P.equalize_hist(low)
    assert eq.max() - eq.min() > 0.9
    assert abs(eq.mean() - 0.5) < 0.05


def test_contrast_stretch():
    x = np.linspace(0.3, 0.6, 100, dtype=np.float32)
    s = P.contrast_stretch(x, 0, 100)
    assert abs(s.min()) < 1e-6 and abs(s.max() - 1) < 1e-6


def test_quantize_and_kmeans(color_img):
    q = P.quantize_uniform(color_img, 4)
    assert len(np.unique(q)) <= 4
    km = P.kmeans_colors(color_img, 2)
    assert len(np.unique(km.reshape(-1, 3), axis=0)) == 2


def test_binary_morphology_matches_scipy(rng):
    mask = rng.random((40, 40)) > 0.8
    se = M.se_cross(1)
    np.testing.assert_array_equal(M.dilate(mask, se), ndi.binary_dilation(mask, se))
    np.testing.assert_array_equal(M.erode(mask, se), ndi.binary_erosion(mask, se, border_value=1))


def test_opening_removes_specks():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    mask[1, 1] = True
    out = M.opening(mask, M.se_square(1))
    assert not out[1, 1] and out[10, 10]
    assert not M.remove_isolated(mask, 1)[1, 1]


def test_thicken():
    mask = np.zeros((11, 11), bool)
    mask[5, 5] = True
    assert M.thicken(mask, 2).sum() > 9
    assert M.thicken(mask, 0).sum() == 1


def test_io_roundtrip_and_resize(color_img):
    png = encode(color_img, "PNG")
    back = load_image(png)
    assert back.shape == color_img.shape
    assert np.abs(back - color_img).max() <= 1 / 255 + 1e-6
    small = resize_max(color_img, 30)
    assert max(small.shape[:2]) == 30


def test_io_16bit_grayscale():
    from PIL import Image
    import io
    a = (np.linspace(0, 4000, 64 * 64).reshape(64, 64)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    img = load_image(buf.getvalue())
    assert img.ndim == 2 and img.min() >= 0 and img.max() <= 1 and img.max() > 0.99


def test_io_rejects_garbage():
    import pytest
    from core.io_utils import InvalidImageError
    with pytest.raises(InvalidImageError):
        load_image(b"not an image at all")
