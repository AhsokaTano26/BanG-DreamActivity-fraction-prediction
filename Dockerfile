# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv
FROM python:3.14.8-slim-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --locked --no-default-groups --no-install-project
COPY production/ /app/production/
COPY training/*.py /app/training/
COPY training/analysis/*.py /app/training/analysis/
COPY training/legacy/*.py /app/training/legacy/
COPY training/legacy/API/*.py /app/training/legacy/API/
COPY deploy/entrypoint.py /app/deploy/entrypoint.py
COPY tests/test_deployment*.py /app/tests/
RUN uv sync --locked --no-default-groups \
    && chmod -R a+rX /app && mkdir -p /state/models /bootstrap && chown -R app:app /state
USER app
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).read()"]
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-P", "/app/deploy/entrypoint.py"]
CMD ["serve"]
