# RunPod Serverless Worker for MT3
# Google Magenta's Multi-Task Multitrack Music Transcription
# Supports: Piano, Strings, Winds, Brass, Percussion, and more
#
# Strategy: Use CUDA 11.8 base + Python 3.10 + pinned legacy versions
# MT3/T5X are legacy projects that require older package versions

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python and JAX environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    libfluidsynth3 \
    build-essential \
    libasound2-dev \
    libjack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install core dependencies with PINNED versions that work together
# This is the critical step - all versions must be mutually compatible
# JAX 0.4.13 is the last version that works well with TF 2.11 and older Flax
RUN pip install --no-cache-dir \
    "numpy==1.23.5" \
    "scipy==1.10.1" \
    "tensorflow==2.11.0" \
    "flax==0.6.10" \
    "optax==0.1.5" \
    "chex==0.1.7" \
    "orbax-checkpoint==0.2.3" \
    "clu==0.0.8" \
    "ml_dtypes==0.2.0" \
    "jax==0.4.13" \
    "jaxlib==0.4.13+cuda11.cudnn86" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install T5X with --no-deps to avoid overwriting our pinned versions
RUN git clone --branch=main https://github.com/google-research/t5x.git /tmp/t5x && \
    cd /tmp/t5x && \
    pip install --no-cache-dir --no-deps -e . && \
    rm -rf /tmp/t5x/.git

# Install MT3 with --no-deps
RUN git clone --branch=main https://github.com/magenta/mt3.git /tmp/mt3 && \
    cd /tmp/mt3 && \
    pip install --no-cache-dir --no-deps -e . && \
    rm -rf /tmp/mt3/.git

# Install protobuf first (use version compatible with TF 2.11)
RUN pip install --no-cache-dir "protobuf==3.20.3"

# Install audio processing dependencies
RUN pip install --no-cache-dir \
    "librosa==0.10.1" \
    "soundfile==0.12.1" \
    "gin-config==0.5.0" \
    "pyfluidsynth==1.3.2" \
    "nest-asyncio==1.6.0"

# Install note-seq with --no-deps to avoid protobuf conflict
RUN pip install --no-cache-dir --no-deps "note-seq==0.0.5"

# Install seqio and t5 with --no-deps
RUN pip install --no-cache-dir --no-deps "seqio==0.0.18" "t5==0.9.4"

# Install tensorflow-text matching TF version
RUN pip install --no-cache-dir "tensorflow-text==2.11.0"

# Install missing dependencies for note-seq/seqio manually
RUN pip install --no-cache-dir \
    "pretty-midi>=0.2.6" \
    "intervaltree>=2.1.0" \
    "pandas>=0.18.1" \
    "bokeh>=0.12.0" \
    "pydub" \
    "mir_eval" \
    "editdistance" \
    "sentencepiece" \
    "tfds-nightly"

# Install RunPod SDK and HTTP
RUN pip install --no-cache-dir \
    "runpod==1.7.7" \
    "requests==2.32.3"

# Install gsutil for downloading models
RUN pip install --no-cache-dir gsutil

# Download MT3 model checkpoints from Google Cloud Storage
RUN mkdir -p /models && \
    gsutil -q -m cp -r gs://mt3/checkpoints/mt3 /models/ && \
    gsutil -q -m cp -r gs://mt3/checkpoints/ismir2021 /models/ && \
    gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 /models/

# Verify models are downloaded
RUN ls -la /models/ && \
    ls -la /models/mt3/ && \
    ls -la /models/ismir2021/

# Copy handler
COPY handler.py .

# Verify versions are correct
RUN python -c "\
import numpy; print(f'numpy: {numpy.__version__}'); \
import tensorflow; print(f'tensorflow: {tensorflow.__version__}'); \
import jax; print(f'jax: {jax.__version__}'); \
import flax; print(f'flax: {flax.__version__}'); \
"

# Verify JAX works (CPU mode during build)
RUN JAX_PLATFORMS=cpu python -c "import jax; print('JAX import OK')"

# Verify MT3 imports work
RUN JAX_PLATFORMS=cpu python -c "\
from mt3 import models, network, spectrograms, vocabularies; \
print('MT3 imports successful'); \
"

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
