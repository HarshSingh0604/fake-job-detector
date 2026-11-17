# API image: FastAPI + the trained pipeline.
#   Local:  docker build -t fake-job-detector . && docker run -p 8000:8000 fake-job-detector
#   Cloud:  the model isn't in git, so pass its download URL: --build-arg MODEL_URL=https://...
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first: this layer stays cached until requirements-api.txt changes, so code edits rebuild in seconds
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# src/ is needed at runtime: the pickled pipeline refers to src.preprocess.build_features by name
COPY src/ src/
COPY app/ app/
COPY models/ models/

# Use the local model if it was copied in, otherwise download it. Loading it here makes a missing or broken
# model fail the build instead of the live service.
ARG MODEL_URL=""
RUN set -e; \
    if [ -n "$MODEL_URL" ]; then \
      python -c "import sys, urllib.request as u; u.urlretrieve(sys.argv[1], sys.argv[2])" "$MODEL_URL" models/fraud_detector.joblib; \
    fi; \
    test -f models/fraud_detector.joblib || { echo "No model: run 'python -m src.train' first or pass --build-arg MODEL_URL=<url>"; exit 1; }; \
    python -c "import joblib; b = joblib.load('models/fraud_detector.joblib'); print('Model OK:', b['model'], b['imbalance'], 'threshold', round(b['threshold'], 3))"

# Don't run as root. Code stays owned by root, so the app can read it but not modify it.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:' + os.getenv('PORT', '8000') + '/health')"
# Cloud hosts set $PORT. `exec` makes uvicorn PID 1, so it receives the stop signal and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
