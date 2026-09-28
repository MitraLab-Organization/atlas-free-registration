# CPU (default):
#   docker build --platform linux/amd64 -t atlas-free-registration:cpu .
# CUDA Jenkins agents:
#   docker build --platform linux/amd64 \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#     -t atlas-free-registration:cuda .
#
# CUDA wheels already ship the CUDA runtime. The agent needs an NVIDIA driver
# and nvidia-container-toolkit; run with --gpus all. Matplotlib is Agg-only,
# so Mesa/libGL is not installed.

FROM python:3.10-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && pip install -r /app/requirements.txt

COPY src /app/src
WORKDIR /app/src

ENTRYPOINT ["python", "/app/src/slice_alignment.py"]
