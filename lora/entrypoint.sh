#!/bin/sh
# Two starts, one image.
#   docker run --gpus all -e WORKER_MODE=1 -e RY_API=... -e RY_WORKER_TOKEN=... millicloud-lora:latest
#     The container is the host. The agent polls and runs train.py in this environment.
#   docker run --rm --gpus all millicloud-lora:latest --job /work/lora-job.json
#     A host-side agent started this container for one training job.
if [ "${1:-}" = "--job" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  exec python3 /opt/lora/train.py "$@"
fi
if [ "$WORKER_MODE" = "1" ]; then
  exec python3 /opt/agent/agent.py
fi
exec python3 /opt/lora/train.py "$@"
