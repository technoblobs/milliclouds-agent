# Renderyard GPU agent

Headless Blender + the coordinator/renderer script in one image. Secure workers run
two containers from this image: the coordinator has outbound HTTPS and the renderer
has no network. The machine needs no open ports, no SSH, no VPN. Works anywhere
Docker + an NVIDIA driver exist:
home PC, Vast.ai instance, AWS/GCP GPU VM.

## Host prerequisites (once per machine)

1. NVIDIA driver (`nvidia-smi` must work on the host)
2. Docker
3. NVIDIA Container Toolkit (lets containers see the GPU):
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ct.gpg
   curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://#' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   ```
   Verify: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
   (On Vast.ai templates this is already done.)

## Build & publish

```bash
docker build --platform linux/amd64 -t ghcr.io/YOURORG/renderyard-agent:latest .
docker push ghcr.io/YOURORG/renderyard-agent:latest
```
Update the image name in the API's launch-command endpoints to match.

## Render isolation modes

`RY_RENDER_MODE=secure` is the default. The API-generated secure command starts a
networked coordinator and a `--network=none` renderer sidecar over a shared temporary
volume. The renderer receives no API or R2 tokens.

`RY_RENDER_MODE=trusted` runs Blender as a child of the coordinator for trusted
machines only. Even in this mode the Blender environment is sanitized and automatic
scene Python execution is disabled. Trusted mode is rejected for pool jobs.

Set `RY_CONTROL_PORT=8787` to enable a local diagnostics panel. Publish it only on
the host loopback interface, for example `-p 127.0.0.1:8787:8787`. Set
`RY_CONTROL_TOKEN` when the panel is exposed beyond a private local machine. It shows
agent state, current work, runtime metrics, and a local stop control. Render mode is
startup configuration and cannot be downgraded from the main site while running.

Do not give the coordinator access to `/var/run/docker.sock`. The generated secure
commands start the sidecar as a separate container before starting the coordinator.

## Run — one-shot (what the job page hands users)

```bash
The job page generates a complete command. It creates a temporary volume, starts the
secure renderer sidecar with `--network=none`, starts the coordinator, and removes the
volume when the one-shot job exits.
```

## Run — always-on worker (what the workers page hands users)

```bash
The workers page generates the coordinator and renderer commands. Both containers
share a worker volume. The renderer is networkless and the coordinator uses outbound
HTTPS for API/R2 traffic.
```

## What the agent guarantees

- EXR-first: renders scene-linear EXR masters; PNG produced via Blender's own
  view transform (Filmic/AgX correct), uploads per requested format
- Zip-slip and zip-bomb guards on archive extraction (20 GB / 20k files caps)
- Heartbeats every 20s renew the lease; a 409 means the lease was lost -> render aborts
- **Render logs**: Blender + agent stdout is batched (~1s) to
  `POST /agent/jobs/{id}/logs`. Redis is never contacted from the agent; the API
  is the only write path (outbound HTTPS only).
- All state changes go through the API's guarded transitions; a zombie agent's
  results are discarded server-side
- Cleans its temp dir on exit; nothing persists on the GPU machine
- Reports cgroup and NVIDIA runtime metrics through the existing worker heartbeat
- Exposes renderer mode and sandbox capability in worker specs

## Local smoke test without a GPU

CPU rendering works too (slow): omit `--gpus all`. The GPU-enable expression
falls back to CPU automatically when no compute device is present.
