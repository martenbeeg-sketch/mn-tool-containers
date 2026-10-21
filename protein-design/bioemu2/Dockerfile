# BioEmu-2 public preview runtime, pinned to the latest upstream source commit.
# The checkpoint is intentionally fetched at runtime into the mounted HF cache.
ARG BIOEMU2_COMMIT=f99e0d899a2548ee6d9d54b6854b2bc35b6652d5
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG BIOEMU2_COMMIT
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3.12 python3.12-dev python3.12-venv python3-pip python-is-python3 \
      build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/bioemu2/.venv

ENV PATH="/opt/bioemu2/.venv/bin:${PATH}"

WORKDIR /opt/bioemu2
RUN git init . \
    && git remote add origin https://github.com/microsoft/bioemu2.git \
    && git fetch --depth 1 origin "${BIOEMU2_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${BIOEMU2_COMMIT}" \
    && python -m pip install --upgrade pip \
    && python -m pip install '.[cuda]' gemmi==0.7.3 \
    && python - <<'PY'
import torch
import bioemu2

if torch.version.cuda is None:
    raise SystemExit("BioEmu-2 image has a CPU-only PyTorch installation")
print(f"BioEmu-2 source installed; torch={torch.__version__}; CUDA={torch.version.cuda}")
PY
COPY protein-design/bioemu2/studio_runner.py /opt/bioemu2/studio_runner.py
RUN bioemu2 --help >/dev/null

WORKDIR /work
ENTRYPOINT ["python", "/opt/bioemu2/studio_runner.py"]
