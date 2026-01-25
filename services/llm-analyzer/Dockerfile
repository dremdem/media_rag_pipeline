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

# Create data and exports directories
RUN mkdir -p /app/data /app/exports

# Set paths
ENV LLM_ANALYZER_DB_PATH=/app/data/analyzer.db
ENV EXPORTS_DIR=/app/exports

# Health check for Docker orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
