"""Kiểm tra bộ lọc tự cài đặt, đối chiếu với scipy.ndimage (scipy CHỈ dùng trong test).

Chế độ biên 'reflect' của numpy tương ứng 'mirror' của scipy.
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

from core import filters as F


@pytest.mark.parametrize("k", [3, 5, 7])
def test_correlate_and_convolve_match_scipy(rng, k):
    img = rng.random((40, 50)).astype(np.float32)
    ker = rng.normal(size=(k, k)).astype(np.float32)
    np.testing.assert_allclose(F.correlate2d(img, ker), ndi.correlate(img, ker, mode="mirror"), atol=1e-5)
    np.testing.assert_allclose(F.convolve2d(img, ker), ndi.convolve(img, ker, mode="mirror"), atol=1e-5)


def test_zero_padding_matches_scipy_constant(rng):
    img = rng.random((20, 20)).astype(np.float32)
    ker = rng.normal(size=(3, 3)).astype(np.float32)
    np.testing.assert_allclose(F.correlate2d(img, ker, mode="constant"),
                               ndi.correlate(img, ker, mode="constant", cval=0.0), atol=1e-5)


def test_color_image_filtered_per_channel(color_img):
    ker = np.ones((3, 3), np.float32) / 9
    out = F.correlate2d(color_img, ker)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], ndi.correlate(color_img[..., c], ker, mode="mirror"), atol=1e-5)


@pytest.mark.parametrize("sigma", [0.8, 1.5, 3.0])
def test_gaussian_matches_scipy(rng, sigma):
    img = rng.random((64, 48)).astype(np.float32)
    ref = ndi.gaussian_filter(img, sigma, mode="mirror", truncate=3.0)
    np.testing.assert_allclose(F.gaussian_blur(img, sigma), ref, atol=1e-5)


def test_gaussian_kernel_normalized():
    k = F.gaussian_kernel1d(2.0)
    assert len(k) == 2 * 6 + 1  # bán kính 3σ
    assert abs(k.sum() - 1) < 1e-6
    assert np.argmax(k) == len(k) // 2


@pytest.mark.parametrize("size", [3, 5])
def test_median_matches_scipy(rng, size):
    img = rng.random((30, 41)).astype(np.float32)
    np.testing.assert_allclose(F.median_filter(img, size), ndi.median_filter(img, size, mode="mirror"), atol=1e-6)


def test_median_removes_salt_and_pepper(rng, square):
    noisy = square.copy()
    m = rng.random(square.shape)
    noisy[m < 0.05] = 0.0
    noisy[m > 0.95] = 1.0
    den = F.median_filter(noisy, 3)
    assert np.abs(den - square).mean() < 0.2 * np.abs(noisy - square).mean()


@pytest.mark.parametrize("size", [3, 7, 15])
def test_box_filter_matches_scipy(rng, size):
    img = rng.random((50, 60)).astype(np.float32)
    np.testing.assert_allclose(F.box_filter(img, size), ndi.uniform_filter(img, size, mode="mirror"), atol=1e-5)


def test_bilateral_preserves_edge_better_than_gaussian(rng, square):
    noisy = np.clip(square + rng.normal(0, 0.05, square.shape), 0, 1).astype(np.float32)
    bil = F.bilateral_filter(noisy, sigma_space=3, sigma_color=0.1)
    gau = F.gaussian_blur(noisy, 3)
    # Độ tương phản ngay tại biên (cột 19 vs 20) gần như giữ nguyên với bilateral.
    contrast = lambda a: float(a[30:50, 21].mean() - a[30:50, 18].mean())
    assert contrast(bil) > 0.5
    assert contrast(bil) > contrast(gau) + 0.2
    # Nhiễu trong vùng phẳng vẫn giảm.
    assert bil[30:50, 30:50].std() < 0.5 * noisy[30:50, 30:50].std()


def test_bilateral_constant_and_color(color_img):
    const = np.full((20, 20), 0.4, np.float32)
    np.testing.assert_allclose(F.bilateral_filter(const, 2, 0.1), const, atol=1e-6)
    out = F.bilateral_filter(color_img, 2, 0.1)
    assert out.shape == color_img.shape


def test_kuwahara_keeps_step_edge(square):
    out = F.kuwahara_filter(square, 3)
    np.testing.assert_allclose(out, square, atol=1e-5)


def test_kuwahara_color_shape(color_img):
    out = F.kuwahara_filter(color_img, 3)
    assert out.shape == color_img.shape
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6


def test_anisotropic_diffusion_smooths_flat_keeps_edge(rng, square):
    noisy = np.clip(square + rng.normal(0, 0.03, square.shape), 0, 1).astype(np.float32)
    out = F.anisotropic_diffusion(noisy, 15, kappa=0.1)
    assert out[30:50, 30:50].std() < noisy[30:50, 30:50].std()
    assert out[30:50, 22].mean() - out[30:50, 17].mean() > 0.5


@pytest.mark.parametrize("method", ["none", "gaussian", "median", "bilateral", "kuwahara", "anisotropic"])
def test_smooth_dispatch(method, color_img):
    assert F.smooth(color_img, method, 1.5).shape == color_img.shape


def test_motion_kernel_normalized():
    k = F.motion_kernel(9, 45)
    assert k.shape == (9, 9) and abs(k.sum() - 1) < 1e-6
    assert k[2, 6] > 0 and k[6, 2] > 0 and k[4, 8] == 0  # nằm trên đường chéo 45°


def test_large_sigma_box_approximation_close_to_gaussian(rng):
    img = rng.random((120, 140)).astype(np.float32)
    img = ndi.gaussian_filter(img, 1)  # có cấu trúc thay vì nhiễu trắng
    ref = ndi.gaussian_filter(img, 8.0, mode="mirror", truncate=3.0)
    out = F.gaussian_blur(img, 8.0)
    assert np.abs(out - ref)[20:-20, 20:-20].max() < 0.01
