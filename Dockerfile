# Use an official Python runtime as a parent image.
# We choose a slim-buster image for a smaller final image size.
FROM python:3.9-slim-buster

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container at /app.
# This step is placed before copying the rest of the application
# to leverage Docker's build cache. If requirements don't change,
# this layer won't be rebuilt.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt.
# We use --no-cache-dir to prevent pip from storing downloaded packages
# which keeps the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app.
COPY . .

# Expose the port that the FastAPI application will run on.
# This informs Docker that the container listens on the specified network ports at runtime.
EXPOSE 8000

# Set environment variables for Uvicorn workers.
# Adjust as necessary based on your server's core count.
# A common recommendation is (2 * number_of_cores) + 1.
ENV WEB_CONCURRENCY=3

# Command to run the application using Uvicorn.
# We use gunicorn as a process manager to run multiple uvicorn workers
# for better production performance and robustness.
# Adjust --workers based on WEB_CONCURRENCY or a fixed number.
# `main:app` refers to the `app` object in `main.py`.
# `--bind 0.0.0.0:8000` makes the server accessible from outside the container.
#CMD ["gunicorn", "main:app", "--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]

# Alternatively, for a simpler setup (less robust for production, but fine for development):
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
