# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHON_ARTIFACT_DIR=/app/data/artifacts

WORKDIR /app

# Dependencies change less frequently than source; keep this layer reusable.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Explicit copies keep application sources included and environment files excluded.
COPY agent/ ./agent/
COPY agents/ ./agents/
COPY api/ ./api/
COPY database/ ./database/
COPY services/ ./services/
COPY prompts/ ./prompts/
COPY main_api.py orchestrator.py ./
COPY docker/ ./docker/

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home appuser \
    && mkdir -p /app/data/artifacts \
    && chown -R appuser:appuser /app/data

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "/app/docker/healthcheck.py"]

ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["python", "-m", "uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
