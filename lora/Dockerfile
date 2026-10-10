# LoRA trainer (train.py). Build on NVIDIA Linux, not a Mac.
#
#   docker build -t millicloud-lora:qwen lora/
#
# Agent env: LORA_DOCKER_IMAGE=millicloud-lora:qwen
# Base weights download on first job into LORA_CACHE. They are not in this image.
#
# transformers 5.19 loads SmolLM3, Phi-4, and Gemma 4 (12B needs >= 5.10).
# It also calls torch.accelerator, which exists from torch 2.6. The base image
# is torch 2.5.1 and that call is what the AutoModel import error hides.
# Stay on CUDA 12.4 so the current NVIDIA driver still matches.
# peft 0.21 matches this Transformers line. Qwen uses this same image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
         torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
         --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch; open('/tmp/constraints.txt','w').write('torch=='+torch.__version__.split('+')[0]+'\n')" \
    && PIP_CONSTRAINT=/tmp/constraints.txt pip install --no-cache-dir \
         "transformers==5.19.0" \
         "peft==0.21.2" \
         "bitsandbytes==0.50.2" \
         "accelerate==1.15.0" \
         "datasets==3.6.0" \
         "safetensors>=0.8.0" \
         sentencepiece protobuf "huggingface_hub>=1.31.0,<3" \
    && python -c "import torch; assert hasattr(torch, 'accelerator') and torch.version.cuda; print('trainer torch ok', torch.__version__)"
# AutoModelForCausalLM is not imported here. Torch 2.6 raises if that import
# runs with no GPU, and docker build does not see the GPU. The job container
# is started with --gpus all.

COPY train.py /opt/lora/train.py

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=debug
WORKDIR /work
ENTRYPOINT ["python3", "/opt/lora/train.py"]
