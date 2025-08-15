# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install system dependencies only needed for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY . .

# Runtime configuration
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GUNICORN_WORKERS=4

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s CMD python -c "import requests; requests.get('http://localhost:$PORT/health', timeout=5)"

EXPOSE $PORT

# Start command
CMD ["gunicorn", "ai-chatbot:app", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "${GUNICORN_WORKERS}", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--timeout", "120"]

ENTRYPOINT ["sh", "-c", "gunicorn ai-chatbot:app --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS:-4} --worker-class uvicorn.workers.UvicornWorker --timeout 120"]