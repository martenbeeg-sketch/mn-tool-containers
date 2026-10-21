FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG BOLTZGEN_COMMIT=a07fea1153111ef453aaea7a15697ee05de9291f

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
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3.11 -m venv /opt/venv

ENV PATH=/opt/venv/bin:${PATH}

COPY scripts/fetch-pinned-source.sh /usr/local/bin/mn-fetch-source
COPY patches/boltzgen.patch /opt/mn-tool-patches/boltzgen.patch
COPY support/protein-design/boltzgen/fold_benchmark_pae.yaml /tmp/fold_benchmark_pae.yaml
COPY support/protein-design/boltzgen/constraints.txt /tmp/boltzgen-constraints.txt
RUN chmod +x /usr/local/bin/mn-fetch-source \
    && python -m pip install --upgrade pip setuptools setuptools_scm wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

RUN mn-fetch-source https://github.com/HannesStark/boltzgen.git "${BOLTZGEN_COMMIT}" /app boltzgen.patch \
    && mkdir -p /app/config \
    && cp /tmp/fold_benchmark_pae.yaml /app/config/fold_benchmark_pae.yaml \
    && cp /tmp/fold_benchmark_pae.yaml /app/src/boltzgen/resources/config/fold_benchmark_pae.yaml \
    && python -m pip install --constraint /tmp/boltzgen-constraints.txt -e /app \
    && mkdir -p /cache /workspace /root/.cache/huggingface \
    && chmod 777 /cache /workspace /root/.cache /root/.cache/huggingface \
    && rm -f /tmp/fold_benchmark_pae.yaml /tmp/boltzgen-constraints.txt

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "boltzgen --help"]
