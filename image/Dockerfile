# NVIDIA Linux/Vast worker only. Do NOT build this on an Apple Silicon Mac:
# Docker Desktop's VM fills up (~60 GB), and CUDA cannot use Metal anyway.
# On a Mac run:  image/setup-macos.sh
#
# Build on an NVIDIA host:
#   docker build --platform linux/amd64 -t millicloud-image:klein image/
#
# Weights are NOT baked in (downloaded on first job).
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers.git" \
      transformers accelerate safetensors sentencepiece protobuf pillow huggingface_hub

COPY generate.py /opt/image/generate.py
COPY NOTICE /opt/image/NOTICE

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["python3", "/opt/image/generate.py"]
