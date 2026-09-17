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
# actually expands. Single worker, single thread: /analyze runs pose + face
# mesh + YOLO + a full SegFormer transformer forward pass synchronously
# against models shared as global state, all inside one process's memory
# budget -- letting a second request run concurrently in another thread
# risks an OOM kill with no catchable exception, not just a slowdown. Real
# concurrency has to come from Cloud Run running more instances (see
# --concurrency=1 on the deploy command), not threads inside one.
# 300s timeout because that synchronous pipeline is genuinely slow.
ENV PORT=8080
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 app:app
