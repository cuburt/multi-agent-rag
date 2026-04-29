FROM python:3.11-slim

WORKDIR /app

# psycopg2 needs libpq; build-essential is needed for any wheel-less deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY frontend ./frontend
COPY evals ./evals
COPY scripts ./scripts
COPY docs ./docs

ENV PORT=8000
EXPOSE 8000

# Honor $PORT so the same image runs under docker-compose (PORT=8000) and
# Cloud Run (PORT injected by the platform, default 8080).
# Seeding runs in the background so uvicorn binds the port immediately —
# Cloud Run's startup health-check won't time out waiting for 12 embedding
# API calls to finish.
CMD ["sh", "-c", "python -m src.db.seed & uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
