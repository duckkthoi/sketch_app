"""Tạo Flask app: cấu hình, DB, cookie client_id ẩn danh, header bảo mật, lỗi dạng JSON."""
from __future__ import annotations

import logging
import re
import secrets

from flask import Flask, g, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import db as D
from .api import bp as api_bp
from .cleanup import purge, start_background
from .config import Config
from .image_store import ImageStore
from .limits import JobGate, RateLimiter
from .schemas import ApiError

COOKIE = "sid"
CID_RE = re.compile(r"^[0-9a-f]{32}$")
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'")


def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or Config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__, static_folder=str(cfg.static_dir), static_url_path="/static-files")
    app.config.update(
        SECRET_KEY=cfg.secret_key,
        MAX_CONTENT_LENGTH=cfg.max_upload_mb * 1024 * 1024,
        JSON_AS_ASCII=False,
        APP=cfg,
        STORE=ImageStore(cfg.storage_dir, cfg.preview_side),
        LIMITS=(RateLimiter(*cfg.rate_limit), RateLimiter(cfg.rate_limit[0] * 4, cfg.rate_limit[1])),
        LIMITS_HEAVY=(RateLimiter(*cfg.heavy_rate_limit), RateLimiter(cfg.heavy_rate_limit[0] * 4, cfg.heavy_rate_limit[1])),
        GATE=JobGate(cfg.max_concurrent_jobs),
    )
    app.json.ensure_ascii = False

    if cfg.trust_proxy:  # Railway đặt app sau reverse proxy -> lấy IP/scheme thật từ X-Forwarded-*
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    D.init_db(cfg.db_path, cfg.schema_path)
    try:
        purge(cfg.storage_dir, cfg.db_path, cfg.retention_days, cfg.max_storage_mb)
    except Exception:  # noqa: BLE001
        app.logger.exception("initial cleanup failed")
    start_background(cfg)

    app.teardown_appcontext(D.close_db)
    app.register_blueprint(api_bp)

    # ------------------------------------------------------------------ client id ẩn danh
    @app.before_request
    def _client_id():
        cid = request.cookies.get(COOKIE, "")
        g.new_client = not CID_RE.match(cid)
        g.client_id = secrets.token_hex(16) if g.new_client else cid

    @app.after_request
    def _after(resp):
        if getattr(g, "new_client", False):
            # Secure khi bật cấu hình hoặc khi request đến qua HTTPS (Railway: X-Forwarded-Proto).
            resp.set_cookie(COOKIE, g.client_id, max_age=365 * 86400, httponly=True,
                            secure=cfg.secure_cookies or request.is_secure, samesite="Lax")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        if request.path.startswith("/api/") and "Cache-Control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ------------------------------------------------------------------ lỗi -> JSON
    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        return jsonify({"error": e.message}), e.status

    @app.errorhandler(413)
    def _too_large(_e):
        return jsonify({"error": f"File vượt quá {cfg.max_upload_mb} MB"}), 413

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"error": e.description or e.name}), e.code
        return e

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        app.logger.exception("unhandled error")
        return jsonify({"error": "Lỗi máy chủ"}), 500

    # ------------------------------------------------------------------ health + frontend
    @app.get("/healthz")
    def healthz():
        D.get_db().execute("SELECT 1").fetchone()
        return jsonify({"ok": True})

    @app.get("/")
    def index():
        if not (cfg.static_dir / "index.html").exists():
            return ("Frontend chưa được build. Chạy: cd web && npm install && npm run build", 503,
                    {"Content-Type": "text/plain; charset=utf-8"})
        resp = send_from_directory(cfg.static_dir, "index.html")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        resp = send_from_directory(cfg.static_dir / "assets", filename)
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"  # tên file có hash
        return resp

    @app.get("/favicon.svg")
    def favicon():
        return send_from_directory(cfg.static_dir, "favicon.svg")

    return app
