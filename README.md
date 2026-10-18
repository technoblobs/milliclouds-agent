# MilliClouds agent

The program that runs on your NVIDIA machine and does the training for [MilliClouds](https://milliclouds.com).

MilliClouds keeps the dataset, the versions, and the comparison. This agent claims a queued version, trains a small LoRA, and uploads the adapter. The website never opens a port on your server. The agent calls out over HTTPS.

Training is supervised fine-tuning. Each row is a prompt and the response you want. The base model stays frozen. The file you download is a Safetensors adapter.

## What you need

- Linux on x86_64. These steps cover Ubuntu, Debian, and Fedora. A Mac cannot claim a training job.
- An NVIDIA driver. `nvidia-smi` must work on the host before Docker is involved.
- Docker Engine, and permission to run `docker` as the user that starts the agent.
- The NVIDIA Container Toolkit, so a container can see the GPU.
- Python 3.11 or newer, and `requests` (`pip install 'requests>=2.31,<3'`).
- Free disk for weights and checkpoints. A small model wants about 30 GB. Weights are cached in `~/.cache/millicloud-lora`.
- Outbound HTTPS to the MilliClouds API. No inbound port, SSH, or VPN is required.

VRAM depends on the template you import on the website. A 1.7B model wants about 6 GB free. A 7B model wants about 12 GB. A 14B model wants about 20 GB. Model notes are in [lora.md](lora.md).

A server with no GPU can still run the agent. It reports online and does not claim a training version.

## Install the GPU runtime

Check the toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Ubuntu or Debian:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ct.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Fedora:

```bash
sudo dnf install -y nvidia-container-toolkit python3-requests
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Build the trainer image

The image is the trainer: PyTorch, Transformers, PEFT, and bitsandbytes. It does not contain Qwen, SmolLM, Gemma, or Phi. The first job for a model downloads those weights. Later jobs reuse the cache.

Build on the GPU server. The tag is `millicloud-lora:latest`. Wait until the build prints `trainer torch ok`. Keep this clone after the build. The agent reads `lora/train.py` from it.

```bash
git clone https://github.com/technoblobs/milliclouds-agent.git
cd milliclouds-agent
docker build --platform linux/amd64 -f lora/Dockerfile -t millicloud-lora:latest .
```

## Create a host token

1. Sign in at [https://milliclouds.com](https://milliclouds.com).
2. Open Hosts.
3. Name the machine and choose **Trusted host**.
4. Click **Register host**.
5. Copy `RY_WORKER_TOKEN` from the one-time command on that page. Do not run that command for training. It starts a different worker. The token is shown once. If you close the panel without saving it, revoke the host and register a new one.

One token is one machine. Revoking the host on the website makes the token stop working.

## Start the agent

`RY_API` must be an address this server can reach. Use the API URL from your MilliClouds account, not `localhost` on another computer.

Foreground, while you are setting the machine up:

```bash
cd milliclouds-agent
export RY_API=https://your-api.example
export RY_WORKER_TOKEN=paste-the-token-once
export RY_RENDER_MODE=trusted
export LORA_DOCKER_IMAGE=millicloud-lora:latest
export WORKER_MODE=1
python3 agent/agent.py
```

Daemon, so it comes back after a reboot. Change `User=gpu` and `/opt/milliclouds-agent` to the account and directory you cloned into.

```bash
sudo install -d -m 755 /etc/millicloud
sudo tee /etc/millicloud/agent.env >/dev/null <<'EOF'
RY_API=https://your-api.example
RY_WORKER_TOKEN=paste-the-token-once
RY_RENDER_MODE=trusted
LORA_DOCKER_IMAGE=millicloud-lora:latest
WORKER_MODE=1
EOF
sudo chmod 600 /etc/millicloud/agent.env

sudo tee /etc/systemd/system/millicloud-agent.service >/dev/null <<'EOF'
[Unit]
Description=MilliClouds training agent
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=gpu
Group=docker
WorkingDirectory=/opt/milliclouds-agent
EnvironmentFile=/etc/millicloud/agent.env
ExecStart=/usr/bin/python3 agent/agent.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now millicloud-agent
```

Do not commit `agent.env`. It holds the host token.

## Rented GPU

A Vast.ai container or a RunPod Pod is already the machine. Do not install a second Docker inside it. Start this image with the GPU and the same token. `WORKER_MODE=1` makes the container poll and run each job in that same environment.

```bash
docker run --gpus all --name millicloud-agent \
  -e RY_API=https://api.milliclouds.com \
  -e RY_WORKER_TOKEN=paste-the-token-once \
  -e RY_RENDER_MODE=trusted \
  -e WORKER_MODE=1 \
  millicloud-lora:latest
```

The log says `lora local` when a version starts. One container is one host and needs its own token. The host setup above stays the right one for a machine you administer: the agent stays a small process and can mount a new `train.py` without rebuilding the image.

## Check that it is working

- `systemctl status millicloud-agent` shows `active (running)`. A foreground run stays in the terminal.
- `journalctl -u millicloud-agent -n 40 --no-pager` includes `worker mode` and `long poll`. `worker token invalid or revoked` means the token is wrong. `poll failed` means this server cannot reach `RY_API`.
- On Hosts, the machine is online and a GPU name is filled in. Online with a blank GPU means `nvidia-smi` failed for that user, so it will not claim a training version.
- `docker image inspect millicloud-lora:latest` succeeds.
- Queue a version on the website. Within a few seconds the log says `claimed chunk`, then `lora docker millicloud-lora:latest`. The version leaves the queue and shows progress.

The first job for a model prints download percent in the version log. A later job for the same model loads the weights from disk. After one version finishes, the next queued version is claimed immediately.

## What a run does

The agent polls MilliClouds, claims the oldest version this GPU is allowed to run, and starts `docker run --gpus all` with `millicloud-lora:latest`. The website chooses the model, the rank, the learning rate, and the epochs. The trainer uses a 4-bit base model, a cosine learning rate, and early stopping.

Comparison questions for a finished adapter run on the same host. The first question downloads that adapter. Later questions on the same adapter reuse it while the model stays loaded, about 10 minutes after the last question.

## Layout

| Path | What it is |
|---|---|
| `agent/agent.py` | The poller. Run this on the GPU server. |
| `lora/` | Trainer image and the training, inference, and comparison scripts. |
| `lora.md` | Model cards, VRAM floors, and template ids. |
| `.env.example` | Variable names only. No tokens. |
| `LICENSE` | MIT license for this agent and trainer. |

This repository is MIT licensed. Model weights keep the license of the checkpoint you download. The MilliClouds website and API are not in this repository.
