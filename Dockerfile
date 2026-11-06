# syntax=docker/dockerfile:1

# ---------- Stage 1: build frontend TypeScript -> /static
FROM node:24-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- Stage 2: Python runtime
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    STORAGE_DIR=/app/storage \
    TRUST_PROXY=true \
    PORT=8000
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py gunicorn.conf.py docker-entrypoint.sh ./
COPY core ./core
COPY server ./server
COPY db ./db
COPY samples ./samples
COPY --from=web /static ./static
RUN chmod +x docker-entrypoint.sh && mkdir -p /app/storage && chown -R app:app /app/storage
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
