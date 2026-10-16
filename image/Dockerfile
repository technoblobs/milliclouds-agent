# Image-gen runner (generate.py). Build on NVIDIA Linux / Vast, not a Mac.
#
#   docker build -t milliclouds/worker:image .
#   docker push milliclouds/worker:image
#
# Agent env: IMAGE_DOCKER_IMAGE=milliclouds/worker:image
# Weights are NOT in this image (downloaded on first job into IMAGE_CACHE).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
         "git+https://github.com/huggingface/diffusers.git" \
         transformers accelerate safetensors sentencepiece protobuf pillow huggingface_hub

COPY generate.py /opt/image/generate.py
COPY NOTICE /opt/image/NOTICE

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["python3", "/opt/image/generate.py"]
