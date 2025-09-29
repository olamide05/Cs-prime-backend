# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port (Render will override this, but good practice)
EXPOSE 8000

# Run FastAPI with Uvicorn, binding to Render's $PORT
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
