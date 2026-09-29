# LoRA training (template + one runtime)

Fourth engine next to Blender, OpenFOAM, and image. The website queues a job,
an NVIDIA worker claims one try, and the adapter uploads to the job page.

The trainer image is fixed: `causal-qlora@1` (`millicloud-lora:qwen`). Models
are JSON templates. A new causal language model is a new file, not a code change.

Bundled templates:

| Id | VRAM floor | Disk |
|---|---|---|
| `millicloud/templates/qwen2.5-7b-instruct@v1` | 12 GB free | 30 GB |
| `millicloud/templates/qwen2.5-1.5b-instruct@v1` | 6 GB free | 15 GB |

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

## Build the trainer image

On a Linux machine with an NVIDIA GPU (not an Apple Silicon Mac):

```bash
cd /path/to/renderyard
docker build --platform linux/amd64 -t millicloud-lora:qwen milliclouds-agent/lora/
```

The agent looks for `millicloud-lora:qwen`, or `LORA_DOCKER_IMAGE` if you tag
it something else. Hugging Face cache defaults to `~/.cache/millicloud-lora`.

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
