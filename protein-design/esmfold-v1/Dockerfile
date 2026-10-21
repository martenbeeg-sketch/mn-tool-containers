# Reproducible classic ESMFold V1 runtime. The fair-esm source archive is the
# same pinned 2.0.1 source used by the OVO ESM-IF1 bundle; ESMFold itself is a
# separate structure-prediction workflow.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git build-essential python3.10 python3.10-dev python3-pip python-is-python3 \
      ninja-build libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python --version

RUN python -m pip install --upgrade pip==25.1.1 \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.0 torchvision==0.22.0 \
    && python -m pip install \
      numpy==1.26.4 scipy==1.15.3 biopython==1.86 \
      einops==0.8.1 dm-tree==0.1.8 ml-collections==1.1.0 \
      omegaconf==2.3.0 pytorch-lightning==1.9.5 ninja==1.11.1.4

# OVO's ESM-IF1 environment pins fair-esm 2.0.1 to this source archive.
COPY protein-design/esmfold-v1/fair-esm-2b369911.tar.gz /tmp/fair-esm.tar.gz
RUN echo "bbef420a03769762e8ac85b0672adb07b876a5ab25c9dab4ed31aacc60b056dd  /tmp/fair-esm.tar.gz" | sha256sum -c - \
    && python -m pip install --no-deps /tmp/fair-esm.tar.gz \
    && rm /tmp/fair-esm.tar.gz \
    && python - <<'PY'
from pathlib import Path
import esm.pretrained

# Torch 2.7 defaults to weights_only=True; fair-esm V1 checkpoints store an
# argparse.Namespace in the trusted model checkpoint. Explicitly allow it.
source = Path(esm.pretrained.__file__)
text = source.read_text()
anchor = "import torch\n"
patch = "import torch\nimport argparse\ntorch.serialization.add_safe_globals([argparse.Namespace])\n"
if patch not in text:
    if anchor not in text:
        raise SystemExit(f"Cannot patch torch checkpoint compatibility in {source}")
    source.write_text(text.replace(anchor, patch, 1))
PY

# FAIR's ESMFold dependencies include OpenFold's CUDA attention extension.
# Pin the source commit and patch its C++ standard for PyTorch 2.7's C++17 ABI.
RUN git clone https://github.com/aqlaboratory/openfold.git /tmp/openfold \
    && git -C /tmp/openfold checkout --detach 4b41059694619831a7db195b7e0988fc4ff3a307 \
    && sed -i 's/-std=c++14/-std=c++17/g' /tmp/openfold/setup.py \
    && python - <<'PY'
from pathlib import Path
import re

setup = Path("/tmp/openfold/setup.py")
text = setup.read_text()
text, count = re.subn(
    r"compute_capabilities = set\(\[.*?\ncc_flag = \[\]",
    "compute_capabilities = {(8, 9)}\ncc_flag = []",
    text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("Could not set the OpenFold extension target to the RTX 4090 architecture")
setup.write_text(text)
PY
RUN git -C /tmp/openfold diff --check \
    && TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python -m pip install \
      --no-build-isolation --no-deps /tmp/openfold \
    && rm -rf /tmp/openfold

# NVIDIA's lightweight logger is required by OpenFold's ESMFold modules.
RUN git clone https://github.com/NVIDIA/dllogger.git /tmp/dllogger \
    && git -C /tmp/dllogger checkout --detach 0478734ff7be75adde8d160e04872664d1c62e5f \
    && python -m pip install --no-deps /tmp/dllogger \
    && rm -rf /tmp/dllogger

RUN python - <<'PY'
import esm
import esm.esmfold.v1.pretrained
import openfold
import torch

print(f"torch={torch.__version__}; cuda={torch.version.cuda}; esm={esm.__version__}")
PY
RUN esm-fold --help >/dev/null

ENV TORCH_HOME=/reference/esmfold_v1_torch
WORKDIR /work
