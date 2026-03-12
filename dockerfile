FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf_cache \
    TRANSFORMERS_CACHE=/workspace/.hf_cache/transformers \
    HUGGINGFACE_HUB_CACHE=/workspace/.hf_cache/hub

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install deps first (better caching)
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install faiss-cpu

# copy project
COPY . .

CMD ["python", "-u", "-m", "scripts.run_pipeline", "--config", "configs/experiments/base.yaml"]