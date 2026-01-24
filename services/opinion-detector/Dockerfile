FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    openai \
    pydantic \
    tenacity

# Copy application
COPY app /app/app

# Create data directory for SQLite
RUN mkdir -p /app/data

# Set database path
ENV OPINION_DB_PATH=/app/data/opinions.db

# Health check for Docker orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
