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

# Fix dependency conflicts: pin numpy and protobuf for tensorflow compatibility
# Then install JAX with CUDA support (bundled CUDA libs, no local CUDA needed)
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "protobuf>=3.20.3,<5.0.0"

# Install JAX 0.4.35 with bundled CUDA 12 (works without system CUDA)
RUN pip install --no-cache-dir \
    "jax[cuda12_pip]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Downgrade Flax ecosystem to be compatible with JAX 0.4.35
# T5X uses optax.ConditionallyTransformState which was added in optax 0.1.8+
RUN pip install --no-cache-dir --force-reinstall \
    "flax==0.8.5" \
    "orbax-checkpoint==0.5.23" \
    "chex==0.1.86" \
    "optax==0.1.9" \
    "ml_dtypes==0.4.0"

# Verify versions are correct
RUN python -c "\
import numpy; print(f'numpy: {numpy.__version__}'); \
import ml_dtypes; print(f'ml_dtypes: {ml_dtypes.__version__}'); \
import jax; print(f'jax: {jax.__version__}'); \
import flax; print(f'flax: {flax.__version__}'); \
import optax; print(f'optax: {optax.__version__}'); \
"

# Skip GPU verification during build (no GPU available in build environment)
# GPU will be available at runtime on RunPod
RUN JAX_PLATFORMS=cpu python -c "import jax; print('JAX import OK - GPU will be used at runtime')"

# Verify MT3 imports work (use CPU mode during build)
RUN JAX_PLATFORMS=cpu python -c "\
from mt3 import models, network, spectrograms, vocabularies; \
print('MT3 imports successful'); \
"

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
