# syntax=docker/dockerfile:1
FROM python:3.14.8-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY deploy/requirements-web.txt /app/deploy/requirements-web.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements-web.txt
COPY production/ /app/production/
COPY analysis/*.py /app/analysis/
COPY API/GetEventInfo.py API/GetEventTracker.py /app/API/
COPY download_ranks.py /app/download_ranks.py
# Keep the Mac fingerprint's historical file, without importing it at startup.
COPY XGBoost.py /app/xgboost.py
COPY deploy/entrypoint.py /app/deploy/entrypoint.py
COPY tests/test_deployment*.py /app/tests/
RUN chmod -R a+rX /app && mkdir -p /state/models /bootstrap && chown -R app:app /state
USER app
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).read()"]
ENTRYPOINT ["python", "-P", "/app/deploy/entrypoint.py"]
CMD ["serve"]
