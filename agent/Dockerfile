# Renderyard GPU agent — Blender headless + Cycles GPU (OptiX/CUDA via host driver)
# Blender is downloaded as a Linux x86_64 build; publish this image for x86_64 GPU hosts.
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG BLENDER_VERSION=4.2.3
ARG BLENDER_SERIES=4.2

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-requests curl xz-utils ca-certificates \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 \
      libxkbcommon0 libsm6 libice6 libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
      | tar -xJ -C /opt \
 && ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender

RUN useradd -m render \
 && mkdir -p /sandbox /work \
 && chown -R render:render /sandbox /work
USER render
WORKDIR /work

COPY --chown=render agent.py /opt/agent/agent.py

# Cycles needs host GPU: run with `--gpus all`
ENV BLENDER_BIN=blender
ENV RY_RENDER_MODE=secure
ENTRYPOINT ["python3", "/opt/agent/agent.py"]
