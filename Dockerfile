# Serving image for the Heart Disease Risk API
FROM python:3.12-slim

# Must match the scikit-learn version used for training (pickle compatibility).
# `make docker-build` passes it automatically from the local environment.
ARG SKLEARN_VERSION=1.5.2

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best_model.joblib \
    METADATA_PATH=/app/models/model_metadata.json \
    LOG_DIR=/app/logs

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt "scikit-learn==${SKLEARN_VERSION}"

COPY src/ ./src/
COPY models/best_model.joblib models/model_metadata.json ./models/

RUN useradd --create-home --uid 1000 appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
