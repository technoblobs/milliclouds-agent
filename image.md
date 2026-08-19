# Image generation (FLUX.2 Klein 4B)

Third engine next to Blender and OpenFOAM. Same job path: website or desktop
queues a job → registered worker claims it → PNG(s) upload to the job page.

**Model:** [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)  
**License:** Apache 2.0 (commercial-safe). Weights are **downloaded on first
job**, not baked into the Docker image.

Do **not** point this runner at FLUX.2 [dev] or Klein 9B — those are
non-commercial.

---

## What you need

| Host | Runtime | One-time setup |
|---|---|---|
| **This Mac (M2 / Metal)** | Native Python + MPS | `image/setup-macos.sh` |
| **NVIDIA Linux / Vast** | Docker + GPU | `docker build` below |

Also:

- API + website running
- Desktop signed in, **this computer registered**, **agent started** (Trusted host)
- Disk: **~20 GB free** on the first job (weights ~8–12 GB + cache). Later jobs ~2 GB.
- Memory: **16 GB unified** on Mac is comfortable; 8 GB often fails (`no_vram`)
- NVIDIA: **~8–13 GB free VRAM**

No GPU → job fails with `no_gpu`. The worker does not run diffusion on CPU.

---

## 1. Restart the API (new `engine=image`)

From `api/`:

```bash
docker compose cp app/schemas.py api:/srv/app/schemas.py
docker compose cp app/routers/jobs.py api:/srv/app/routers/jobs.py
docker compose restart api
```

Or rebuild the API image if you do not bind-mount source.

---

## 2. Mac worker (your M2)

Docker on Mac **cannot** use Metal. Use the venv:

```bash
cd /Users/bimal/development/own/fxrender-new/renderyard/image
chmod +x setup-macos.sh
./setup-macos.sh
```

That creates `image/.venv` with PyTorch (MPS) and Diffusers (`Flux2KleinPipeline`).

Start MilliCloud Desktop → **Agent → Start**. Leave it running.

First image job downloads Klein 4B into `~/.cache/huggingface` (several minutes,
needs network). Later jobs reuse the cache.

---

## 3. NVIDIA Linux / Vast worker (not this Mac)

Do **not** `docker build` `image/Dockerfile` on an M1/M2/M3 Mac. That image is
CUDA/`linux/amd64`. On Apple Silicon Docker runs as `aarch64`, the build
installs a CPU torch wheel, and Docker Desktop’s **own disk** (often ~64 GB,
separate from `df -h`) fills up with `No space left on device`.

```bash
# On a Linux box with an NVIDIA GPU only:
cd /path/to/renderyard
docker build --platform linux/amd64 -t millicloud-image:klein image/
# smoke: docker run --rm --gpus all millicloud-image:klein --help
```

Needs `nvidia-container-toolkit` (`docker run --gpus all` must work).

### Fedora + RTX 3070 Ti (8 GB)

The 3070 Ti is tight for Klein 4B. The runner **CPU-offloads**; expect slower
jobs, not a hard fail, as long as ~5 GB VRAM is free.

```bash
# GPU in containers
sudo dnf install -y nvidia-container-toolkit python3-requests
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Agent next to the image (do not reuse the Mac's worker token)
sudo mkdir -p /opt/millicloud && sudo chown "$USER" /opt/millicloud
# copy agent/agent.py from this repo → /opt/millicloud/agent.py

# Website → Workers → new host (Trusted) → paste token once
cat > /opt/millicloud/env <<'EOF'
export RY_API=https://YOUR-PUBLIC-API   # not localhost — Fedora cannot see your Mac
export RY_WORKER_TOKEN=paste-once-token
export WORKER_MODE=1
export RY_RENDER_MODE=trusted
export IMAGE_DOCKER_IMAGE=millicloud-image:klein
export IMAGE_CACHE=/var/cache/millicloud-image
EOF
mkdir -p /var/cache/millicloud-image

# Reachability: Tailscale/Cloudflare Tunnel to the Mac API, or host the API
# where this box can POST /agent/poll. R2 uploads do not go through the API.

set -a && source /opt/millicloud/env && set +a
python3 /opt/millicloud/agent.py
```

`3070 Ti` + first job still downloads ~8–12 GB weights into `IMAGE_CACHE`.

The agent sees `nvidia-smi` + Docker and runs:

`docker run --rm --gpus all -v work:/work -v ~/.cache/millicloud-image:/cache millicloud-image:klein --job /work/image-job.json`

Override the tag with env `IMAGE_DOCKER_IMAGE` if you rename it.

---

## 4. Queue a job

**Website:** New render → **Image gen**

- **Text to image** — prompt only
- **Image to image** — prompt + reference PNG/JPEG/WebP

Defaults: 1024×1024, 4 steps, guidance 1.0 (Klein is distilled).

**Desktop:** Jobs tab (signed in) → same prompt form → **Queue on this host**.

The registered agent picks the job on the next poll (~10 s). Watch the **job
page Render log** (`[image] …`), not `docker compose` API logs.

---

## Outputs

PNG files `image_01.png`, … on the job page. Download as usual.

---

## Failures

| Code | Meaning | What to do |
|---|---|---|
| `no_gpu` | No CUDA / Metal | Use a Mac with MPS setup or an NVIDIA host |
| `no_vram` | Not enough GPU / unified memory | Close other GPU apps; use 16 GB+ Mac or 12 GB+ NVIDIA |
| `no_disk` | Not enough free disk for weights | Free ~20 GB |
| `no_libs` | No venv and no Docker image | Run `setup-macos.sh` or build `millicloud-image:klein` |
| `model_load` | Hugging Face download / Diffusers API | Check network; `pip install git+https://github.com/huggingface/diffusers.git` |
| `image_failed` | Pipeline error | Read the job log |

---

## Optional env (agent process)

| Variable | Purpose |
|---|---|
| `IMAGE_PYTHON` | Python with torch (default: `image/.venv/bin/python`) |
| `IMAGE_GENERATE` | Path to `generate.py` |
| `IMAGE_DOCKER_IMAGE` | Default `millicloud-image:klein` |
| `IMAGE_CACHE` | NVIDIA Docker HF cache (default `~/.cache/millicloud-image`) |
| `HF_HOME` | Hugging Face cache for native runs |

---

## License (short)

Shipping this **runner** in Docker is fine. Klein 4B is Apache 2.0: keep
`image/NOTICE` if you redistribute weights. Do not vendor [dev] / Klein 9B
weights into the image without a Black Forest Labs commercial license.
