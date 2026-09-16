# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run injects PORT at runtime (defaults to 8080) and expects the
# container to bind to it -- shell form (not exec-array form) so $PORT
# actually expands. Single worker: each worker loads its own copy of the
# SegFormer/YOLO models into memory, and multiple threads give some
# concurrency for I/O-bound waits without multiplying that memory cost.
# 300s timeout because a single /analyze call runs pose + face mesh + YOLO
# + a full transformer forward pass synchronously.
ENV PORT=8080
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 app:app
