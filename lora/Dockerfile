# LoRA trainer (train.py). Build on NVIDIA Linux, not a Mac.
#
#   docker build -t millicloud-lora:qwen lora/
#
# Agent env: LORA_DOCKER_IMAGE=millicloud-lora:qwen
# Base weights download on first job into LORA_CACHE. They are not in this image.
#
# transformers 5.19 loads SmolLM3, Phi-4, and Gemma 4 (12B needs >= 5.10).
# Torch 2.5 in the base image is the minimum Transformers 5 accepts.
# peft 0.21 matches that Transformers line. Qwen uses this same image.
#
# Keep that CUDA torch. An unpinned pip install replaces it with a PyPI wheel.
# Transformers then reports AutoModel as missing and the job exits 2.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN python -c "import torch; open('/tmp/constraints.txt','w').write('torch=='+torch.__version__.split('+')[0]+'\n')"
ENV PIP_CONSTRAINT=/tmp/constraints.txt

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
         "transformers==5.19.0" \
         "peft==0.21.2" \
         "bitsandbytes==0.50.2" \
         "accelerate==1.15.0" \
         "datasets==3.6.0" \
         "safetensors>=0.8.0" \
         sentencepiece protobuf "huggingface_hub>=1.31.0,<3" \
    && pip install --no-cache-dir --no-deps \
         torchvision==0.20.1 torchaudio==2.5.1 \
         --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch, torchvision; from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; from peft import LoraConfig; from datasets import Dataset; assert torch.version.cuda; print('trainer import ok', torch.__version__)"

COPY train.py /opt/lora/train.py

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=debug
WORKDIR /work
ENTRYPOINT ["python3", "/opt/lora/train.py"]
