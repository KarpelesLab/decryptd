# decryptd — GPU worker for decrypt, as a small self-contained container.
#
# The image carries just the release binary. Everything GPU-related — libcuda and
# libnvidia-ml — is injected at runtime by the NVIDIA container runtime, so there's
# no CUDA toolkit here. decryptd also needs no GUI libraries (the tray degrades to
# headless) and bundles its own TLS roots, so nothing else is required at runtime.
#
# Build:
#   docker build -t decryptd .
# Run (needs the NVIDIA container runtime; `--restart` makes it permanent):
#   docker run -d --name decryptd --restart unless-stopped --gpus all \
#     -v decryptd-data:/data decryptd
#
# On RunPod: push this image to a registry and use it as the pod's image — RunPod
# runs the entrypoint on every start, so the worker comes up automatically and
# survives restarts. Mount a volume at /data to keep the worker id + cache.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/KarpelesLab/decryptd" \
      org.opencontainers.image.description="GPU worker for decrypt"

# Expose the host GPU through the NVIDIA runtime: `compute` = CUDA (libcuda),
# `utility` = NVML/nvidia-smi (the tray's temperature/power readout).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Fetch the latest release binary. curl + ca-certificates are needed only to
# download it at build time and are removed in the same layer to keep the image
# lean. Override --build-arg DECRYPTD_URL=... to pin a specific version.
ARG DECRYPTD_URL=https://github.com/KarpelesLab/decryptd/releases/latest/download/decryptd-linux-x86_64.tar.gz
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    curl -fSL "$DECRYPTD_URL" | tar -xz --no-same-owner -C /usr/local/bin decryptd; \
    chmod +x /usr/local/bin/decryptd; \
    apt-get purge -y --auto-remove curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Persist the worker id + download cache across restarts by mounting a volume here.
VOLUME /data
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/decryptd", "--workdir", "/data"]
