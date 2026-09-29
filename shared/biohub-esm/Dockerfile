# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BIOHUB_ESM_DESIGN_COMMIT=43b4548b86762edfa747b07d5f440aad3c33acee
ARG CUEQUIVARIANCE_VERSION=0.12.0

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    ESMCFOLD_CCD_PATH=/ref/biohub-esm/ESMFold2/ccd.pkl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 \
    && python -m pip install esm==3.4.1.post1 "huggingface_hub[cli]" modal abnumber \
    && python -m pip install --force-reinstall --no-deps \
        "cuequivariance-ops-cu12==${CUEQUIVARIANCE_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQUIVARIANCE_VERSION}" \
    && mkdir -p "$HF_HOME" /opt/esm-binder-design \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Biohub/esm/${BIOHUB_ESM_DESIGN_COMMIT}/cookbook/tutorials/binder_design.py" \
        -o /opt/esm-binder-design/binder_design.py

COPY support/shared/biohub-esm/entrypoint.sh /usr/local/bin/biohub-esm-entrypoint
COPY support/shared/biohub-esm/smoke_test.py /usr/local/bin/biohub-esm-smoke-test
COPY support/shared/biohub-esm/binder_design.py /usr/local/bin/biohub-esm-binder-design

RUN chmod +x \
        /usr/local/bin/biohub-esm-entrypoint \
        /usr/local/bin/biohub-esm-smoke-test \
        /usr/local/bin/biohub-esm-binder-design

WORKDIR /opt
ENTRYPOINT ["biohub-esm-entrypoint"]
CMD ["biohub-esm-smoke-test"]
