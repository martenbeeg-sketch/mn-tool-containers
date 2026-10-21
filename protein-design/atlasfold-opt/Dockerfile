FROM python:3.11-slim-bookworm

ARG ATLASFOLD_UPSTREAM_COMMIT=992067e67df29b665c501e0d2e9ead9dd4ba9b69
ARG ATLASFOLD_KIT_REV=unknown

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/cache/home \
    XDG_CACHE_HOME=/cache/xdg \
    ATLASFOLD_WEIGHTS_DIR=/reference/atlasfold \
    MODEL_OPT_JIT_ROOT=/cache/atlasfold/jit

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install \
      "cuequivariance-torch==0.10.0" \
      "cuequivariance-ops-torch-cu12==0.10.0" \
      "numpy==1.26.4" \
      "scipy==1.17.1" \
      "einops==0.8.0" \
      "gemmi==0.7.5" \
      "omegaconf==2.3.0" \
      "numba==0.61.0" \
      "huggingface_hub==0.36.2"

COPY atlasfold/ /kit/atlasfold/
COPY common/opt_core/ /kit/common/opt_core/
COPY protein-design/atlasfold-opt/mn-atlasfold-runner /usr/local/bin/mn-atlasfold-runner

RUN chmod 0755 /usr/local/bin/mn-atlasfold-runner \
    && python /kit/atlasfold/stock/check_pins.py \
    && python -c "import torch; assert torch.__version__.startswith('2.7.1+cu128'), torch.__version__"

LABEL org.opencontainers.image.title="MN AtlasFold optimized" \
      org.opencontainers.image.description="AtlasFold v1.0.0 with the pinned AtlasFold optimization kit modes" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.revision="${ATLASFOLD_UPSTREAM_COMMIT}" \
      org.mn.atlasfold.optimizer-kit-revision="${ATLASFOLD_KIT_REV}" \
      org.mn.atlasfold.cuda-stack="torch-2.7.1-cu128"

ENTRYPOINT ["/usr/local/bin/mn-atlasfold-runner"]
