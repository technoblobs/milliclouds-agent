# LoRA training (template + one runtime)

Fourth engine next to Blender, OpenFOAM, and image. The website queues a job,
an NVIDIA worker claims one try, and the adapter uploads to the job page.

The runtime stays `causal-qlora@1` (`millicloud-lora:latest`). A new causal model
is a JSON template. SmolLM3, Gemma 4, and Phi-4 also need the trainer image in
the section below.

Bundled templates:

| Id | VRAM floor | Disk |
|---|---|---|
| `millicloud/templates/qwen2.5-1.5b-instruct@v1` | 6 GB free | 15 GB |
| `millicloud/templates/smollm2-1.7b-instruct@v1` | 6 GB free | 15 GB |
| `millicloud/templates/smollm3-3b@v1` | 8 GB free | 20 GB |
| `millicloud/templates/phi-4-mini-instruct@v1` | 8 GB free | 20 GB |
| `millicloud/templates/qwen2.5-7b-instruct@v1` | 12 GB free | 30 GB |
| `millicloud/templates/gemma-4-e2b-it@v1` | 12 GB free | 25 GB |
| `millicloud/templates/qwen2.5-14b-instruct@v1` | 20 GB free | 40 GB |
| `millicloud/templates/gemma-4-12b-it@v1` | 20 GB free | 40 GB |
| `millicloud/templates/phi-4@v1` | 20 GB free | 40 GB |

The 20 GB rows are the 4090-sized models. Gemma 4 31B does not fit a 24 GB card
for QLoRA. SmolLM's largest instruct model is the 3B. Every row is still
`{"prompt", "response"}`. The trainer wraps each row as
`### Prompt:` / `### Response:`, including these models. Compare uses that
same wrap.

---

## SmolLM

Apache-2.0. Llama-style attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

| Template | Hugging Face repo | Card |
|---|---|---|
| `millicloud/templates/smollm2-1.7b-instruct@v1` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B. 6 GB floor. Fits the 3070 Ti. This checkpoint is `LlamaForCausalLM` and also loads on an older trainer image. |
| `millicloud/templates/smollm3-3b@v1` | `HuggingFaceTB/SmolLM3-3B` | 3B instruct model with think / no-think. 8 GB floor. This is the largest SmolLM. `model_type` is `smollm3`, so it needs the image below. |

Import: `millicloud templates import millicloud/templates/smollm2-1.7b-instruct@v1`

---

## Gemma 4

Apache-2.0. Text rows only. The checkpoints also accept images and, on E2B,
audio. This trainer does not send those.

| Template | Hugging Face repo | Card |
|---|---|---|
| `millicloud/templates/gemma-4-e2b-it@v1` | `google/gemma-4-E2B-it` | About 2.3B effective, 5.1B with embeddings. 12 GB floor. The bf16 download is about 10 GB. Attention targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`. |
| `millicloud/templates/gemma-4-12b-it@v1` | `google/gemma-4-12B-it` | 12B unified, encoder-free. 20 GB floor, 40 GB disk. Keys and values share one projection, so the targets are `q_proj`, `k_proj`, `o_proj`. |

Gemma 4 is not a plain causal class. `train.py` and `infer.py` try
`AutoModelForCausalLM`, then `AutoModelForImageTextToText`. Both files are
bind-mounted or copied from the agent checkout, so the NVIDIA host needs those
copies as well as the new image. Transformers must be 5.10 or newer. The
Dockerfile pins 5.19.0.

---

## Phi-4

MIT. Layers are fused. Targets are `qkv_proj`, `o_proj`, `gate_up_proj`,
`down_proj`. There is no separate `q_proj`.

| Template | Hugging Face repo | Card |
|---|---|---|
| `millicloud/templates/phi-4-mini-instruct@v1` | `microsoft/Phi-4-mini-instruct` | 3.8B. 8 GB floor, 20 GB disk. The small Phi-4. |
| `millicloud/templates/phi-4@v1` | `microsoft/phi-4` | 14B. 20 GB floor, 40 GB disk. The bf16 weights are about 29 GB. This is the 4090 Phi-4. |

Both configs are `Phi3ForCausalLM`. Transformers 4.46 does not load them
reliably. Use the image below.

---

## Build the trainer image

Build this on the NVIDIA Linux host. A Mac image cannot see the GPU. One tag,
`millicloud-lora:latest`, trains Qwen and the models above. The pins live in
`milliclouds-agent/lora/Dockerfile`:

```dockerfile
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
    && python -c "import torch; assert hasattr(torch, 'accelerator'); print('trainer torch ok', torch.__version__)"
```

Transformers 5.19 calls `torch.accelerator`. Torch 2.5.1 does not have it, and
the import error is reported as `AutoModel` / `AutoModelForCausalLM`. The build
installs torch 2.6.0 with CUDA 12.4, then keeps later pip installs from
replacing that wheel. A finished build prints `trainer torch ok`. The model
import itself runs in the job container, which is started with the GPU.

Copy `lora/Dockerfile`, `lora/train.py`, and `lora/infer.py` onto the worker
checkout, then build. On the Fedora host that is `~/work/milliclouds-agent`:

```bash
cd ~/work/milliclouds-agent
docker build --platform linux/amd64 -f lora/Dockerfile -t millicloud-lora:latest .
```

Leave the agent running. It reads `train.py` from the checkout on the next
job, and the next `docker run` uses the new `millicloud-lora:latest` tag. Queue
again only after the build prints `trainer torch ok`. Weights still cache in
`~/.cache/millicloud-lora`.
The first Gemma 4 or Phi-4 job downloads that model's weights. Qwen jobs keep
using the weights already cached.

The API catalog update is separate. It does not require this image. A job
queued before the image exists will fail at model load.

A sweep is the try list inside the template. Each free NVIDIA worker that meets
the VRAM floor takes one try. The job keeps the adapter with the lowest
validation loss. Tries do not share gradients.

---

## What you need

| Host | Runtime |
|---|---|
| NVIDIA Linux / Vast with free VRAM at or above the template floor | Docker + GPU |
| Mac | Does not claim these jobs. The API requires `gpu_name` and `vram_mb`. |

Also: API + website running, desktop agent started, **~30 GB free disk**
(weights plus checkpoints).

---

## Dataset

UTF-8 `.jsonl`. Column names come from the template. The bundled templates use:

```json
{"prompt": "Summarize this note", "response": "The note says the render farm is full."}
```

The trainer holds out the last 5% for validation and never trains on those
rows. The template's `max_rows` caps how many lines are read.

---

## Publish a template

Copy `api/app/lora_catalog/qwen2.5-1.5b-instruct.json`. Change the id, revision,
model repo, VRAM floor, dataset column names, and recipe. `runtime` stays
`causal-qlora@1`. Commit it to a public git repo and tag it.

Import on the new-job page with any of:

- `millicloud/templates/qwen2.5-1.5b-instruct@v1` (files shipped with the API)
- `your-org/your-repo/templates/my-model.json@v1`
- `https://raw.githubusercontent.com/your-org/your-repo/v1/templates/my-model.json`
- the same JSON uploaded from disk

The workspace stores a copy. The job snapshots that copy, so editing the file
later does not change a run that already started. A host heartbeats total VRAM.
The claim skips it when that number is under `hardware.min_vram_mb`.

Checkpoints follow `recipe.save_steps`. A worker that picks the try up again
continues from the last checkpoint.
