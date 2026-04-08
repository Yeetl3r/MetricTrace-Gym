# ESG-Audit-Gym: Dockerfile
# =========================
# Isolated execution container for the OpenEnv environment server.
# Builds from the openenv-base image and exposes port 8000.

FROM python:3.11-slim AS openenv-base

# Metadata
LABEL maintainer="ESG-Audit-Gym Team"
LABEL description="OpenEnv-compliant ESG sustainability auditing environment"
LABEL version="1.0.0"

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
