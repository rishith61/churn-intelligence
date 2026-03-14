FROM python:3.11-slim

LABEL description="Churn Intelligence · FastAPI"

WORKDIR /code

# System deps for scikit-learn / xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first — maximises layer cache on redeploy
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/   ./app/
COPY model/churn_pipeline.joblib ./model/churn_pipeline.joblib

# FIX: create outputs dir so predict_batch() can write results
RUN mkdir -p /code/outputs

# Security: run as non-root
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /code
USER appuser

# FIX: Render uses port 10000, not 8000
EXPOSE 10000

# FIX: port matches Render's expected port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]