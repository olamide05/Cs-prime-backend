# 1 build stage
FROM python:3.11-slim-bullseye AS builder

# Environment vars for faster Python startup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies (only in this stage)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# 2. Production Stage

FROM python:3.11-slim-bullseye

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Create non-root user
RUN useradd -m appuser

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY . .

# Switch to non-root
USER appuser

# Health check endpoint for container orchestration
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose FastAPI port
EXPOSE $PORT

# Start Gunicorn with Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "ai-chatbot:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--graceful-timeout", "30"]
