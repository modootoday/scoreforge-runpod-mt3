# RunPod Serverless Worker for MT3
# Google Magenta's Multi-Task Multitrack Music Transcription
# Supports: Piano, Strings, Winds, Brass, Percussion, and more

# Use NVIDIA CUDA with Python for JAX GPU support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python and JAX environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Install Python 3.11 (required by Flax >= 0.8) and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    libfluidsynth3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python -m pip install --upgrade pip setuptools wheel

# Install T5X and MT3 first (they have older dependency requirements)
# Clone and install T5X with modified setup (jax instead of jax[tpu])
RUN git clone --branch=main https://github.com/google-research/t5x.git /tmp/t5x && \
    cd /tmp/t5x && \
    sed -i "s/jax\[tpu\]/jax/" setup.py && \
    pip install --no-cache-dir -e . && \
    rm -rf /tmp/t5x/.git

# Install MT3
RUN git clone --branch=main https://github.com/magenta/mt3.git /tmp/mt3 && \
    cd /tmp/mt3 && \
    pip install --no-cache-dir -e . && \
    rm -rf /tmp/mt3/.git

# Install additional dependencies BEFORE JAX upgrade
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MT3 model checkpoints from Google Cloud Storage
# This makes models part of the image for faster cold starts
RUN mkdir -p /models && \
    pip install --no-cache-dir gsutil && \
    gsutil -q -m cp -r gs://mt3/checkpoints/mt3 /models/ && \
    gsutil -q -m cp -r gs://mt3/checkpoints/ismir2021 /models/ && \
    gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 /models/

# Verify models are downloaded
RUN ls -la /models/ && \
    ls -la /models/mt3/ && \
    ls -la /models/ismir2021/

# Copy handler
COPY handler.py .

# CRITICAL: Force upgrade JAX and ml_dtypes LAST (after ALL other installs)
# This ensures nothing can overwrite these versions
RUN pip install --no-cache-dir --force-reinstall --no-deps "ml_dtypes==0.5.0" && \
    pip install --no-cache-dir --upgrade \
    "jax[cuda12]==0.4.35" \
    "jaxlib[cuda12_cudnn89]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify ml_dtypes version is correct
RUN python -c "import ml_dtypes; print(f'ml_dtypes version: {ml_dtypes.__version__}')" && \
    python -c "import jax; print(f'JAX version: {jax.__version__}')"

# Verify JAX can see GPU (will show CPU if no GPU during build, that's OK)
RUN python -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'Backend: {jax.default_backend()}')"

# Verify MT3 imports work
RUN python -c "\
from mt3 import models, network, spectrograms, vocabularies; \
print('MT3 imports successful'); \
"

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
