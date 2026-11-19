# syntax=docker/dockerfile:1
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1 \
    PORT=8080

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.29 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY travel_concierge ./travel_concierge

RUN uv sync --frozen --no-install-project \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8080

CMD ["uv", "run", "python", "-m", "travel_concierge.app_server"]
