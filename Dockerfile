# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRIDGE_LOG_DIR=/app/logs \
    BRIDGE_DATABASE_PATH=/app/data/bridge.sqlite3 \
    BRIDGE_TEMP_DIR=/app/tmp

RUN groupadd --system --gid 10001 bridge \
    && useradd --system --uid 10001 --gid bridge --home-dir /app --shell /usr/sbin/nologin bridge \
    && install -d -o bridge -g bridge -m 0700 /app /app/data /app/logs /app/tmp

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER bridge:bridge

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.environ.get('BRIDGE_BIND_PORT', '8080'); path=os.environ.get('BRIDGE_HEALTH_PATH', '/health'); urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3).read()"

ENTRYPOINT ["kaiten-ai-bridge"]
