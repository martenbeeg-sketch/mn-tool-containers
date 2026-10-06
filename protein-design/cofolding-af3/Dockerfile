FROM python:3.12-slim-bookworm

ARG COLABFOLD_COMMIT=f850f0f22f007355bd264d48e9ed263d59331426
ARG ALPHAFOLD3_VERSION=3.1.14
ARG JAX_CUDA_SPEC="jax[cuda13]>=0.7.2,<0.11.2"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        libstdc++6 \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/fetch-pinned-source.sh /usr/local/bin/mn-fetch-source
RUN chmod +x /usr/local/bin/mn-fetch-source \
    && mn-fetch-source https://github.com/sokrypton/ColabFold.git "${COLABFOLD_COMMIT}" /opt/ColabFold \
    && python -m pip install --no-cache-dir --upgrade pip \
    && cd /opt/ColabFold \
    && python -m pip install --no-cache-dir '.[alphafold3]' \
    && python -m pip install --no-cache-dir --upgrade "${JAX_CUDA_SPEC}" \
    && mkdir -p /opt/alphafold3 /cache \
    && curl -fsSL "https://raw.githubusercontent.com/sokrypton/alphafold3/v${ALPHAFOLD3_VERSION}/run_alphafold.py" \
       -o /opt/alphafold3/run_alphafold.py \
    && python - <<'PY'
from pathlib import Path
import importlib.util

# Match the AF3 preview notebook's compatibility patches at image-build time.
haiku = importlib.util.find_spec("haiku")
if haiku and haiku.origin:
    source = Path(haiku.origin).parent / "_src" / "jaxpr_info.py"
    if source.is_file():
        text = source.read_text()
        source.write_text(text.replace("jax.core.DropVar", "jax.extend.core.DropVar"))

tokamax = importlib.util.find_spec("tokamax")
if tokamax and tokamax.origin:
    source = Path(tokamax.origin).parent / "_src" / "gpu_utils.py"
    if source.is_file():
        text = source.read_text()
        old = "return float(device.compute_capability) >= 8.0"
        new = "cc = float(device.compute_capability)\n  return cc == 8.0 or cc >= 9.0  # Ada uses XLA attention"
        if old in text:
            source.write_text(text.replace(old, new))

import jax
import alphafold3
print("AlphaFold 3 package:", alphafold3.__file__)
print("JAX:", jax.__version__)
PY

COPY protein-design/cofolding-af3/gpu-smoke-test.py /usr/local/bin/cofolding-gpu-smoke-test
COPY protein-design/cofolding-af3/cofolding-af3-runner.py /usr/local/bin/cofolding-af3-runner.py

WORKDIR /work
ENV HOME=/cache/home \
    XDG_CACHE_HOME=/cache/xdg \
    HF_HOME=/cache/huggingface \
    AF3_CCD_CACHE_DIR=/cache/alphafold3/ccd \
    AF3_MSA_CACHE_DIR=/msa_cache/mn-cofolding/alphafold3 \
    AF3_SHARED_MSA_CACHE_DIR=/msa_cache
VOLUME ["/cache", "/work"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/usr/local/bin/cofolding-af3-runner.py", "--help"]
