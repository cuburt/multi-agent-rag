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

EXPOSE 8000

# Default command — overridable from docker-compose.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
