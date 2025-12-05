FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui

FROM nvidia/cuda:12.6.3-base-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/pbusenius/asr-api"

ENV PYTHON_VERSION=3.10

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libcudnn8 \
    libcublas-12-6 \
    libcublas-dev-12-6 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

# Install CUDA dependencies from PyTorch CUDA index
# Strategy: Install torch/torchaudio separately with uv pip install (respects UV_NO_VERIFY_HASHES)
# Then install rest of dependencies with uv sync (without cuda extra)
# Also install NeMo toolkit if Parakeet engine is used
ARG ASR_ENGINE=faster_whisper
RUN uv sync && \
    UV_NO_VERIFY_HASHES=1 uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1 torchaudio==2.7.1 && \
    if [ "$ASR_ENGINE" = "parakeet" ]; then \
        echo "Installing NeMo toolkit for Parakeet engine..." && \
        UV_NO_VERIFY_HASHES=1 uv pip install "nemo_toolkit[asr]>=2.4.0"; \
    fi

# Download model during build
# Use BuildKit cache mount to share model cache between builds
ARG ASR_MODEL=base
ENV ASR_MODEL=${ASR_MODEL}
ENV ASR_ENGINE=${ASR_ENGINE}
ENV ASR_MODEL_PATH=/app/models

# Download model using cache mount - models are cached between builds
# The script checks if model exists in cache, downloads if not, then copies to /app/models
RUN --mount=type=cache,target=/root/.cache \
    mkdir -p /app/models && \
    # Use cache directory for downloads, then copy to /app/models
    ASR_MODEL_PATH=/root/.cache uv run python scripts/download_model.py && \
    # Copy models from cache to /app/models for runtime use
    if [ -d "/root/.cache/faster-whisper" ]; then \
        cp -r /root/.cache/faster-whisper /app/models/ 2>/dev/null || true; \
    fi && \
    if [ -d "/root/.cache/whisper" ]; then \
        cp -r /root/.cache/whisper /app/models/ 2>/dev/null || true; \
    fi && \
    if [ -d "/root/.cache/huggingface" ]; then \
        cp -r /root/.cache/huggingface /app/models/ 2>/dev/null || true; \
    fi && \
    echo "Model checked/downloaded to /app/models"

EXPOSE 9000

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["asr-api"]
