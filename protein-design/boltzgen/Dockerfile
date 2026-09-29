FROM nvidia/cuda:12.8.1-cudnn9-devel-ubuntu22.04

ARG BOLTZGEN_COMMIT=58c1eed2b07f00fd5263f78fe2821c80d6875699

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache \
    HF_HUB_DISABLE_XET=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git libboost-all-dev libffi-dev \
        libgl1 libhdf5-dev libssl-dev libxml2-dev libxslt1-dev pkg-config \
        python3.10 python3.10-dev python3-pip python3-venv python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

COPY scripts/fetch-pinned-source.sh /usr/local/bin/mn-fetch-source
COPY patches/boltzgen.patch /opt/mn-tool-patches/boltzgen.patch
COPY support/protein-design/boltzgen/fold_benchmark_pae.yaml /tmp/fold_benchmark_pae.yaml
RUN chmod +x /usr/local/bin/mn-fetch-source \
    && python -m pip install --upgrade pip setuptools setuptools_scm wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    && mn-fetch-source https://github.com/HannesStark/boltzgen.git "${BOLTZGEN_COMMIT}" /app boltzgen.patch \
    && cp /tmp/fold_benchmark_pae.yaml /app/config/fold_benchmark_pae.yaml \
    && python -m pip install -e /app \
    && mkdir -p /cache /workspace /root/.cache/huggingface \
    && chmod 777 /cache /workspace /root/.cache /root/.cache/huggingface \
    && rm -f /tmp/fold_benchmark_pae.yaml

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "boltzgen --help"]
