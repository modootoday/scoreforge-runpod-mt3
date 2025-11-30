# RunPod Serverless Worker for MT3
# Google Magenta's Multi-Task Multitrack Music Transcription
# NOTE: This is a placeholder - MT3 requires complex JAX/T5X setup

FROM python:3.10-slim

WORKDIR /app

# Prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
