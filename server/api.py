"""REST API cho web. Mọi dữ liệu lịch sử/preset đều lọc theo client_id (cookie ẩn danh)."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, g, jsonify, request, send_file

from core.color import rgb_to_gray
from core.effects import CATEGORIES, EFFECTS, compare_edges, compare_filters, preset_params, run_effect
from core.io_utils import InvalidImageError, encode, load_image, resize_max, to_data_url
from core.point_ops import histogram

from . import db as D
from . import schemas as S
from .schemas import ApiError

bp = Blueprint("api", __name__, url_prefix="/api")

STEP_SIDE = 360       # ảnh trung gian gửi về ở kích thước nhỏ cho nhẹ
COMPARE_SIDE = 480
MAX_PRESETS = 50


def _app():
    return current_app.config["APP"]


def _store():
    return current_app.config["STORE"]


def _limit(heavy: bool = False) -> None:
    lim_c, lim_ip = current_app.config["LIMITS_HEAVY" if heavy else "LIMITS"]
    lim_c.check(g.client_id)
    lim_ip.check(request.remote_addr or "?")


def _hist(img: np.ndarray) -> list[int]:
    h = histogram(rgb_to_gray(img), 64)
    return (h * 1000 // max(1, h.max())).astype(int).tolist()


def _image_json(row, preview: np.ndarray) -> dict:
    return {
        "id": row["id"], "filename": row["filename"], "category": row["category"],
        "width": row["width"], "height": row["height"],
        "preview": to_data_url(preview, "WEBP", 90), "histogram": _hist(preview),
    }


def _load_owned_image(image_id: str):
    row = D.get_image(D.get_db(), S.image_id(image_id), g.client_id)
    if row is None:
        raise ApiError(404, "Ảnh không tồn tại hoặc đã hết hạn, hãy tải lại ảnh")
    try:
        work, preview = _store().load(row["id"], row["path"])
    except (OSError, InvalidImageError) as e:
        raise ApiError(410, "File ảnh đã bị xoá, hãy tải lại ảnh") from e
    return row, work, preview


def _result_json(r: dict) -> dict:
    return {
        "id": r["id"], "effect": r["effect"], "effect_name": EFFECTS[r["effect"]].name if r["effect"] in EFFECTS else r["effect"],
        "params": r["params"], "category": r["category"], "filename": r["filename"],
        "width": r["width"], "height": r["height"], "duration_ms": r["duration_ms"], "created_at": r["created_at"],
        "image_id": r["image_id"], "image_available": bool(r.get("image_available")),
        "thumb_url": f"/api/results/{r['id']}/thumb", "image_url": f"/api/results/{r['id']}/image",
    }


def _ingest(img: np.ndarray, filename: str, category: str):
    if min(img.shape[:2]) < 16:
        raise ApiError(400, "Ảnh quá nhỏ (cần tối thiểu 16×16 px)")
    cfg = _app()
    img = resize_max(img, cfg.work_side)
    image_id, rel = _store().save_upload(img)
    h, w = img.shape[:2]
    db = D.get_db()
    D.insert_image(db, id=image_id, client_id=g.client_id, filename=filename, category=category, width=w, height=h, path=rel)
    row = D.get_image(db, image_id, g.client_id)
    return jsonify({"image": _image_json(row, _store().load(image_id, rel)[1])}), 201


# =========================================================================== metadata
@bp.get("/effects")
def effects():
    return jsonify({
        "effects": [e.to_json() for e in EFFECTS.values()],
        "categories": [{"id": k, "label": v[0]} for k, v in CATEGORIES.items()],
        "presets": {cat: {eid: preset_params(eid, cat) for eid in EFFECTS} for cat in CATEGORIES},
        "limits": {"max_upload_mb": _app().max_upload_mb, "work_side": _app().work_side},
    })


def _sample_files() -> dict[str, Path]:
    root = _app().samples_dir
    out = {}
    for cat in CATEGORIES:
        for p in sorted((root / cat).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"):
                out[f"{cat}/{p.stem}"] = p
    return out


_sample_thumbs: dict[str, bytes] = {}


@bp.get("/samples")
def samples():
    return jsonify({"samples": [
        {"id": sid, "category": sid.split("/")[0], "name": sid.split("/")[1].replace("_", " "),
         "thumb_url": f"/api/samples/{sid}/thumb"}
        for sid in _sample_files()
    ]})


@bp.get("/samples/<cat>/<name>/thumb")
def sample_thumb(cat: str, name: str):
    sid = f"{cat}/{name}"
    files = _sample_files()
    if sid not in files:
        raise ApiError(404, "Không có ảnh mẫu này")
    if sid not in _sample_thumbs:
        _sample_thumbs[sid] = encode(resize_max(load_image(files[sid]), 240), "WEBP", 80)
    resp = send_file(io.BytesIO(_sample_thumbs[sid]), mimetype="image/webp")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@bp.post("/samples/use")
def use_sample():
    _limit(heavy=True)
    sid = S.body(request).get("id")
    files = _sample_files()
    if sid not in files:
        raise ApiError(404, "Không có ảnh mẫu này")
    return _ingest(load_image(files[sid]), files[sid].name, sid.split("/")[0])


# =========================================================================== ảnh
@bp.post("/upload")
def upload():
    _limit(heavy=True)
    f = request.files.get("file")
    if f is None or not f.filename:
        raise ApiError(400, "Chưa chọn file ảnh")
    try:
        img = load_image(f.stream.read())
    except InvalidImageError as e:
        raise ApiError(400, str(e)) from e
    return _ingest(img, S.safe_filename(f.filename), S.category(request.form.get("category")))


@bp.get("/images/<image_id>")
def get_image(image_id: str):
    row, _, preview = _load_owned_image(image_id)
    return jsonify({"image": _image_json(row, preview)})


# =========================================================================== xử lý
@bp.post("/process")
def process():
    _limit()
    data = S.body(request)
    eid, params = S.effect_and_params(data)
    _, _, preview = _load_owned_image(data.get("image_id"))
    t0 = time.perf_counter()
    with current_app.config["GATE"].slot():
        res = run_effect(eid, preview, params)
    ms = int((time.perf_counter() - t0) * 1000)
    out = {"image": to_data_url(res.image, "WEBP", 92), "params": params, "duration_ms": ms, "histogram": _hist(res.image)}
    if data.get("steps"):
        out["steps"] = [{"label": lbl, "image": to_data_url(resize_max(np.clip(im, 0, 1), STEP_SIDE), "WEBP", 85)}
                        for lbl, im in res.steps]
    return jsonify(out)


@bp.post("/compare")
def compare():
    _limit()
    data = S.body(request)
    kind = data.get("kind")
    if kind not in ("filters", "edges"):
        raise ApiError(400, "kind phải là 'filters' hoặc 'edges'")
    _, _, preview = _load_owned_image(data.get("image_id"))
    small = resize_max(preview, COMPARE_SIDE)
    t0 = time.perf_counter()
    with current_app.config["GATE"].slot():
        items = compare_filters(small) if kind == "filters" else compare_edges(small)
    return jsonify({"items": [{"label": lbl, "image": to_data_url(np.clip(im, 0, 1), "WEBP", 88)} for lbl, im in items],
                    "duration_ms": int((time.perf_counter() - t0) * 1000)})


def _render_full(data: dict):
    eid, params = S.effect_and_params(data)
    row, work, _ = _load_owned_image(data.get("image_id"))
    t0 = time.perf_counter()
    with current_app.config["GATE"].slot():
        res = run_effect(eid, work, params)
    return row, eid, params, res.image, int((time.perf_counter() - t0) * 1000)


@bp.post("/download")
def download():
    _limit(heavy=True)
    data = S.body(request)
    fmt, mime, ext = S.out_format(data.get("format"))
    row, eid, _, img, _ = _render_full(data)
    stem = Path(row["filename"] or "image").stem
    return send_file(io.BytesIO(encode(img, fmt, 95)), mimetype=mime, as_attachment=True,
                     download_name=f"{stem}_{eid}.{ext}")


# =========================================================================== lịch sử
@bp.post("/results")
def save_result():
    _limit(heavy=True)
    data = S.body(request)
    row, eid, params, img, ms = _render_full(data)
    rid, rel, thumb = _store().save_result(img)
    h, w = img.shape[:2]
    db = D.get_db()
    D.insert_result(db, id=rid, image_id=row["id"], client_id=g.client_id, effect=eid, params=params,
                    category=row["category"], filename=row["filename"], path=rel, thumb_path=thumb,
                    width=w, height=h, duration_ms=ms)
    return jsonify({"result": _result_json(D.get_result(db, rid, g.client_id))}), 201


@bp.get("/results")
def list_results():
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(60, max(1, int(request.args.get("per_page", 24))))
    except ValueError as e:
        raise ApiError(400, "page/per_page không hợp lệ") from e
    items, total = D.list_results(D.get_db(), g.client_id, page, per_page)
    return jsonify({"items": [_result_json(r) for r in items], "total": total, "page": page, "per_page": per_page})


def _owned_result(result_id: str) -> dict:
    r = D.get_result(D.get_db(), S.image_id(result_id), g.client_id)
    if r is None:
        raise ApiError(404, "Không tìm thấy kết quả")
    return r


@bp.get("/results/<result_id>")
def get_result(result_id: str):
    return jsonify({"result": _result_json(_owned_result(result_id))})


@bp.get("/results/<result_id>/thumb")
def result_thumb(result_id: str):
    r = _owned_result(result_id)
    resp = send_file(_store().abspath(r["thumb_path"]), mimetype="image/webp")
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


@bp.get("/results/<result_id>/image")
def result_image(result_id: str):
    r = _owned_result(result_id)
    stem = Path(r["filename"] or "image").stem
    return send_file(_store().abspath(r["path"]), mimetype="image/png",
                     as_attachment=request.args.get("download") == "1", download_name=f"{stem}_{r['effect']}.png")


@bp.delete("/results/<result_id>")
def delete_result(result_id: str):
    r = D.delete_result(D.get_db(), S.image_id(result_id), g.client_id)
    if r is None:
        raise ApiError(404, "Không tìm thấy kết quả")
    _store().delete(r["path"], r["thumb_path"])
    return jsonify({"ok": True})


# =========================================================================== preset
@bp.get("/presets")
def list_presets():
    return jsonify({"items": D.list_presets(D.get_db(), g.client_id)})


@bp.post("/presets")
def create_preset():
    data = S.body(request)
    eid, params = S.effect_and_params(data)
    name = S.preset_name(data.get("name"))
    db = D.get_db()
    if D.count_presets(db, g.client_id) >= MAX_PRESETS:
        raise ApiError(400, f"Tối đa {MAX_PRESETS} preset")
    return jsonify({"preset": D.insert_preset(db, g.client_id, name, eid, params)}), 201


@bp.delete("/presets/<int:preset_id>")
def delete_preset(preset_id: int):
    if not D.delete_preset(D.get_db(), preset_id, g.client_id):
        raise ApiError(404, "Không tìm thấy preset")
    return jsonify({"ok": True})
