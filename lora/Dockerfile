# LoRA trainer (train.py). Build on NVIDIA Linux / Vast, not a Mac.
#
#   docker build -t millicloud-lora:qwen lora/
#
# Agent env: LORA_DOCKER_IMAGE=millicloud-lora:qwen
# Base weights download on first job into LORA_CACHE. They are not in this image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
         "transformers==4.46.3" \
         "peft==0.13.2" \
         "bitsandbytes==0.44.1" \
         "accelerate==1.1.1" \
         "datasets==3.1.0" \
         safetensors sentencepiece protobuf huggingface_hub

COPY train.py /opt/lora/train.py

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["python3", "/opt/lora/train.py"]
