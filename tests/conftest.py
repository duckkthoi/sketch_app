import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def square():
    """Ảnh tổng hợp: hình vuông sáng 0.8 trên nền 0.2, cạnh từ hàng/cột 20 đến 59."""
    img = np.full((80, 80), 0.2, dtype=np.float32)
    img[20:60, 20:60] = 0.8
    return img


@pytest.fixture
def color_img(rng):
    img = np.zeros((60, 90, 3), dtype=np.float32)
    img[:, :45] = (0.9, 0.2, 0.1)
    img[:, 45:] = (0.1, 0.3, 0.8)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)
