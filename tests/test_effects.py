import numpy as np
import pytest

from core.effects import CATEGORIES, EFFECTS, compare_edges, compare_filters, preset_params, run_effect


@pytest.mark.parametrize("eid", list(EFFECTS))
@pytest.mark.parametrize("kind", ["color", "gray"])
def test_every_effect_runs(eid, kind, color_img):
    img = color_img if kind == "color" else color_img.mean(-1)
    res = run_effect(eid, img, {})
    assert res.image.shape[:2] == img.shape[:2]
    assert res.image.dtype == np.float32
    assert 0 <= res.image.min() and res.image.max() <= 1
    assert np.isfinite(res.image).all()
    assert len(res.steps) >= 2
    for name, step in res.steps:
        assert isinstance(name, str) and step.shape[:2] == img.shape[:2]


@pytest.mark.parametrize("eid", list(EFFECTS))
def test_effect_is_deterministic(eid, color_img):
    a = run_effect(eid, color_img, {}).image
    b = run_effect(eid, color_img, {}).image
    np.testing.assert_array_equal(a, b)


def test_every_edge_operator_in_sketch_and_lineart(color_img):
    ops = [v for v, _ in next(p for p in EFFECTS["sketch"].params if p.key == "edge_op").options]
    for op in ops:
        for eid in ("sketch", "lineart"):
            assert run_effect(eid, color_img, {"edge_op": op}).image.shape[:2] == color_img.shape[:2]


def test_params_are_validated():
    eff = EFFECTS["pencil"]
    p = eff.normalize({"blur": 1e9, "darkness": "abc", "enhance": "hack", "unknown": 1})
    assert p["blur"] == 40.0
    assert p["darkness"] == eff.defaults()["darkness"]
    assert p["enhance"] == "none"
    assert "unknown" not in p
    assert EFFECTS["lineart"].normalize({"clean": "false"})["clean"] is False
    assert EFFECTS["cartoon"].normalize({"colors": 7.6})["colors"] == 8


def test_unknown_effect():
    with pytest.raises(ValueError):
        run_effect("nope", np.zeros((10, 10), np.float32))


def test_presets_apply_known_keys_only():
    for cat in CATEGORIES:
        for eid in EFFECTS:
            p = preset_params(eid, cat)
            assert set(p) == set(EFFECTS[eid].defaults())
    assert preset_params("sketch", "medical")["enhance"] == "stretch"
    assert preset_params("oil", "medical") == EFFECTS["oil"].normalize({})


def test_schema_json():
    js = EFFECTS["sketch"].to_json()
    assert js["id"] == "sketch" and js["params"]
    edge = next(p for p in js["params"] if p["key"] == "edge_op")
    assert {"value": "canny", "label": "Canny"} in edge["options"]


def test_line_art_draws_square_outline(square):
    out = run_effect("lineart", square, {"smoothing": "none", "edge_op": "canny", "thickness": 0}).image
    assert out[40, 40] > 0.9 and out[5, 5] > 0.9       # giấy trắng
    assert out[40, 18:22].min() < 0.5                    # có nét tại biên


def test_compare_lists(color_img):
    f = compare_filters(color_img)
    e = compare_edges(color_img)
    assert len(f) == 6 and len(e) == 8
    assert all(img.shape[:2] == color_img.shape[:2] for _, img in f + e)
