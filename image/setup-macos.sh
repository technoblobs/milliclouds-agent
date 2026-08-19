#!/bin/bash
# Native MPS venv for Apple Silicon. Docker on Mac cannot use Metal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
echo
echo "OK. Agent will use: $ROOT/.venv/bin/python"
echo "First image job downloads FLUX.2 Klein 4B (~8–12 GB) into ~/.cache/huggingface"
echo "Need ~20 GB free disk and 16 GB unified memory (8 GB may fail)."
