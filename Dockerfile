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
# JAX 0.3.x + T5X commit 2e05ad41 + Flax 0.5.3 (last version with flax.optim)
# Reference: https://github.com/google/flax/issues/2273
RUN pip install --no-cache-dir \
    "numpy==1.23.5" \
    "scipy==1.10.1" \
    "tensorflow==2.11.0" \
    "flax==0.5.3" \
    "optax==0.1.3" \
    "chex==0.1.5" \
    "orbax==0.0.2" \
    "clu==0.0.7" \
    "jax==0.3.25" \
    "jaxlib==0.3.25+cuda11.cudnn82" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install T5X with --no-deps to avoid overwriting our pinned versions
# CRITICAL: Use specific commit BEFORE airio dependency was added
# Commit 2e05ad41 is from jsphweid/mt3-docker, known to work with MT3
RUN git clone https://github.com/google-research/t5x.git /tmp/t5x && \
    cd /tmp/t5x && \
    git checkout 2e05ad41778c25521738418de805757bf2e41e9e && \
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

# Install missing dependencies for note-seq/seqio/t5 manually
# These are transitive dependencies that were skipped with --no-deps
# Note: Use tensorflow-datasets instead of tfds-nightly to avoid protobuf conflicts
RUN pip install --no-cache-dir \
    "pretty-midi>=0.2.6" \
    "intervaltree>=2.1.0" \
    "pandas>=0.18.1" \
    "bokeh>=0.12.0" \
    "pydub" \
    "mir_eval" \
    "editdistance" \
    "sentencepiece" \
    "tensorflow-datasets==4.9.2" \
    "pyglove" \
    "ipython" \
    "babel" \
    "immutabledict" \
    "rouge-score" \
    "sacrebleu"

# Install mesh-tensorflow with --no-deps (it has complex TF dependencies)
RUN pip install --no-cache-dir --no-deps "mesh-tensorflow"

# Install RunPod SDK and HTTP
RUN pip install --no-cache-dir \
    "runpod==1.7.7" \
    "requests==2.32.3"

# Install gsutil for downloading models
RUN pip install --no-cache-dir gsutil

# CRITICAL: Force protobuf back to 3.20.x LAST (some packages upgrade it)
# This must be the final pip install to ensure TF 2.11 compatibility
RUN pip install --no-cache-dir --force-reinstall "protobuf==3.20.1"

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

# Print build environment info
RUN echo "=== Build Environment ===" && \
    cat /etc/os-release | head -5 && \
    python --version && \
    nvcc --version 2>/dev/null || echo "NVCC not in PATH (OK for build)" && \
    echo "========================="

# Verify versions are correct
RUN python -c "\
import numpy; print(f'numpy: {numpy.__version__}'); \
import tensorflow; print(f'tensorflow: {tensorflow.__version__}'); \
import jax; print(f'jax: {jax.__version__}'); \
import flax; print(f'flax: {flax.__version__}'); \
import google.protobuf; print(f'protobuf: {google.protobuf.__version__}'); \
"

# Verify JAX works (CPU mode during build)
RUN JAX_PLATFORMS=cpu python -c "import jax; print('JAX import OK')"

# Verify MT3 imports work (step by step for debugging)
RUN JAX_PLATFORMS=cpu python -c "\
print('Testing imports...'); \
import t5; print('  t5 OK'); \
import seqio; print('  seqio OK'); \
import note_seq; print('  note_seq OK'); \
from mt3 import spectrograms; print('  mt3.spectrograms OK'); \
from mt3 import vocabularies; print('  mt3.vocabularies OK'); \
from mt3 import network; print('  mt3.network OK'); \
from mt3 import models; print('  mt3.models OK'); \
print('MT3 imports successful!'); \
"

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
