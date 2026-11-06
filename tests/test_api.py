"""Test API + SQLite bằng Flask test client, trên storage tạm (không đụng dữ liệu thật)."""
import io
import time

import numpy as np
import pytest
from PIL import Image

from server import create_app
from server.cleanup import purge
from server.config import Config


def make_cfg(tmp_path, **kw):
    cfg = Config()
    cfg.storage_dir = tmp_path / "storage"
    cfg.cleanup_interval_s = 0
    cfg.secret_key = "test"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def app(tmp_path):
    return create_app(make_cfg(tmp_path))


@pytest.fixture
def client(app):
    return app.test_client()


def png_bytes(w=120, h=90, color=True):
    rng = np.random.default_rng(0)
    a = np.zeros((h, w, 3), np.uint8)
    a[:, : w // 2] = (200, 60, 30)
    a[:, w // 2:] = (30, 80, 200)
    a[h // 3: 2 * h // 3, w // 4: 3 * w // 4] = 240
    a = np.clip(a + rng.integers(0, 10, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a if color else a[..., 0]).save(buf, format="PNG")
    return buf.getvalue()


def upload(client, data=None, category="nature", name="test.png"):
    return client.post("/api/upload", data={"file": (io.BytesIO(data or png_bytes()), name), "category": category},
                       content_type="multipart/form-data")


def test_health_and_effects(client):
    assert client.get("/healthz").get_json() == {"ok": True}
    js = client.get("/api/effects").get_json()
    ids = [e["id"] for e in js["effects"]]
    assert {"pencil", "sketch", "lineart", "cartoon", "color_pencil", "oil"} <= set(ids)
    assert {c["id"] for c in js["categories"]} == {"nature", "medical", "industrial"}
    assert js["presets"]["medical"]["sketch"]["enhance"] == "stretch"


def test_security_headers_and_cookie(client):
    r = client.get("/api/effects")
    assert "sid=" in r.headers.get("Set-Cookie", "")
    assert "HttpOnly" in r.headers["Set-Cookie"] and "SameSite=Lax" in r.headers["Set-Cookie"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    # Lần sau đã có cookie thì không đặt lại.
    assert "Set-Cookie" not in client.get("/api/effects").headers


def test_upload_process_download_flow(client):
    r = upload(client)
    assert r.status_code == 201, r.get_json()
    img = r.get_json()["image"]
    assert img["width"] == 120 and img["height"] == 90 and img["preview"].startswith("data:image/webp;base64,")

    r = client.post("/api/process", json={"image_id": img["id"], "effect": "sketch", "params": {"blur": 999}, "steps": True})
    js = r.get_json()
    assert r.status_code == 200, js
    assert js["image"].startswith("data:image/webp") and js["params"]["blur"] == 40.0
    assert len(js["steps"]) >= 3 and all(s["image"].startswith("data:") for s in js["steps"])
    assert len(js["histogram"]) == 64

    for fmt, magic in (("png", b"\x89PNG"), ("jpg", b"\xff\xd8")):
        r = client.post("/api/download", json={"image_id": img["id"], "effect": "pencil", "params": {}, "format": fmt})
        assert r.status_code == 200 and r.data[:4].startswith(magic)
        assert f"test_pencil.{fmt}" in r.headers["Content-Disposition"]

    r = client.post("/api/compare", json={"image_id": img["id"], "kind": "edges"})
    assert r.status_code == 200 and len(r.get_json()["items"]) == 8


def test_grayscale_upload_all_effects(client):
    img = upload(client, png_bytes(color=False), category="medical").get_json()["image"]
    assert img["category"] == "medical"
    for eid in ("pencil", "sketch", "lineart", "cartoon", "color_pencil", "oil"):
        r = client.post("/api/process", json={"image_id": img["id"], "effect": eid, "params": {}})
        assert r.status_code == 200, (eid, r.get_json())


def test_upload_rejects_bad_files(client, app):
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"<?php evil ?>"), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()
    r = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = upload(client, png_bytes(8, 8))
    assert r.status_code == 400  # quá nhỏ


def test_upload_too_large(tmp_path):
    client = create_app(make_cfg(tmp_path, max_upload_mb=1)).test_client()
    big = io.BytesIO(b"\0" * (2 * 1024 * 1024))
    r = client.post("/api/upload", data={"file": (big, "big.png")}, content_type="multipart/form-data")
    assert r.status_code == 413 and "MB" in r.get_json()["error"]


def test_large_image_downscaled_to_work_side(tmp_path):
    client = create_app(make_cfg(tmp_path, work_side=100)).test_client()
    img = upload(client, png_bytes(400, 200)).get_json()["image"]
    assert (img["width"], img["height"]) == (100, 50)


def test_validation_errors(client):
    img = upload(client).get_json()["image"]
    assert client.post("/api/process", json={"image_id": img["id"], "effect": "nope"}).status_code == 400
    assert client.post("/api/process", json={"image_id": "../../etc", "effect": "pencil"}).status_code == 400
    assert client.post("/api/process", json={"image_id": "0" * 32, "effect": "pencil"}).status_code == 404
    assert client.post("/api/process", data="not json", content_type="text/plain").status_code == 400
    assert client.post("/api/download", json={"image_id": img["id"], "effect": "pencil", "format": "gif"}).status_code == 400
    assert client.post("/api/compare", json={"image_id": img["id"], "kind": "x"}).status_code == 400
    assert client.get("/api/nothing").status_code == 404


def test_history_crud_and_files(client, app):
    img = upload(client).get_json()["image"]
    r = client.post("/api/results", json={"image_id": img["id"], "effect": "oil", "params": {"radius": 3}})
    assert r.status_code == 201
    res = r.get_json()["result"]
    assert res["effect"] == "oil" and res["params"]["radius"] == 3.0 and res["image_available"]

    lst = client.get("/api/results").get_json()
    assert lst["total"] == 1 and lst["items"][0]["id"] == res["id"]
    with client.get(res["thumb_url"]) as th:  # đóng response để Windows nhả file handle
        assert th.data[:4] == b"RIFF"  # webp
    with client.get(res["image_url"] + "?download=1") as full:
        assert full.data[:4] == b"\x89PNG" and "attachment" in full.headers["Content-Disposition"]
    assert client.get(f"/api/results/{res['id']}").get_json()["result"]["id"] == res["id"]

    store = app.config["STORE"]
    path = store.abspath(f"results/{res['id']}.png")
    assert path.exists()
    assert client.delete(f"/api/results/{res['id']}").status_code == 200
    assert not path.exists()
    assert client.get("/api/results").get_json()["total"] == 0
    assert client.delete(f"/api/results/{res['id']}").status_code == 404


def test_history_isolated_between_clients(app):
    a, b = app.test_client(), app.test_client()
    img = upload(a).get_json()["image"]
    res = a.post("/api/results", json={"image_id": img["id"], "effect": "pencil"}).get_json()["result"]
    b.get("/api/effects")  # b nhận cookie riêng
    assert b.get("/api/results").get_json()["total"] == 0
    assert b.get(res["thumb_url"]).status_code == 404
    assert b.delete(f"/api/results/{res['id']}").status_code == 404
    assert b.post("/api/process", json={"image_id": img["id"], "effect": "pencil"}).status_code == 404
    assert a.get("/api/results").get_json()["total"] == 1


def test_history_survives_restart(tmp_path):
    cfg = make_cfg(tmp_path)
    c1 = create_app(cfg).test_client()
    img = upload(c1).get_json()["image"]
    c1.post("/api/results", json={"image_id": img["id"], "effect": "pencil"})
    cookie = c1.get_cookie("sid").value

    c2 = create_app(make_cfg(tmp_path)).test_client()  # "khởi động lại" server: app mới, cache RAM trống
    c2.set_cookie("sid", cookie)
    assert c2.get("/api/results").get_json()["total"] == 1
    assert c2.post("/api/process", json={"image_id": img["id"], "effect": "pencil"}).status_code == 200
    assert c2.get(f"/api/images/{img['id']}").get_json()["image"]["id"] == img["id"]


def test_presets_crud(client):
    r = client.post("/api/presets", json={"name": "  Chì đậm  ", "effect": "pencil", "params": {"darkness": 3}})
    assert r.status_code == 201
    p = r.get_json()["preset"]
    assert p["name"] == "Chì đậm" and p["params"]["darkness"] == 3.0
    assert client.post("/api/presets", json={"name": "", "effect": "pencil"}).status_code == 400
    assert len(client.get("/api/presets").get_json()["items"]) == 1
    assert client.delete(f"/api/presets/{p['id']}").status_code == 200
    assert client.delete(f"/api/presets/{p['id']}").status_code == 404


def test_samples(client):
    js = client.get("/api/samples").get_json()["samples"]
    assert len(js) >= 3 and {s["category"] for s in js} == {"nature", "medical", "industrial"}
    assert client.get(js[0]["thumb_url"]).status_code == 200
    r = client.post("/api/samples/use", json={"id": js[0]["id"]})
    assert r.status_code == 201 and r.get_json()["image"]["category"] == js[0]["category"]
    assert client.post("/api/samples/use", json={"id": "../app"}).status_code == 404


def test_rate_limit(tmp_path):
    client = create_app(make_cfg(tmp_path, rate_limit=(3, 60.0))).test_client()
    img = upload(client).get_json()["image"]
    codes = [client.post("/api/process", json={"image_id": img["id"], "effect": "pencil"}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_cleanup_removes_expired(tmp_path, app):
    client = app.test_client()
    img = upload(client).get_json()["image"]
    res = client.post("/api/results", json={"image_id": img["id"], "effect": "pencil"}).get_json()["result"]
    cfg = app.config["APP"]
    store = app.config["STORE"]
    # Chưa hết hạn: không xoá gì.
    assert purge(cfg.storage_dir, cfg.db_path, 7, 10_000) == {"results": 0, "images": 0, "orphans": 0}
    # Đặt lùi thời gian tạo 8 ngày -> bị xoá cả file lẫn dòng DB.
    import sqlite3
    con = sqlite3.connect(cfg.db_path)
    con.execute("UPDATE results SET created_at = ?", (time.time() - 8 * 86400,))
    con.execute("UPDATE images SET created_at = ?", (time.time() - 8 * 86400,))
    con.commit()
    con.close()
    st = purge(cfg.storage_dir, cfg.db_path, 7, 10_000)
    assert st["results"] == 1 and st["images"] == 1
    assert not store.abspath(f"results/{res['id']}.png").exists()
    assert not store.abspath(f"uploads/{img['id']}.png").exists()
    assert client.get("/api/results").get_json()["total"] == 0


def test_cleanup_enforces_storage_limit(app):
    client = app.test_client()
    for _ in range(3):
        upload(client)
    cfg = app.config["APP"]
    st = purge(cfg.storage_dir, cfg.db_path, 7, 0)  # giới hạn 0 MB -> xoá hết
    assert st["images"] == 3


def test_index_without_build(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.static_dir = tmp_path / "nostatic"
    r = create_app(cfg).test_client().get("/")
    assert r.status_code == 503 and "npm run build" in r.get_data(as_text=True)
