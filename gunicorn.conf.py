"""Cấu hình gunicorn cho production (Docker/Railway)."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"   # Railway truyền cổng qua biến PORT
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
timeout = 120            # xử lý ảnh độ phân giải cao có thể mất vài giây
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
forwarded_allow_ips = "*"
