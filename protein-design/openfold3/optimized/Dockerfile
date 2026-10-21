# Native OpenFold3 0.5.0 + OpenBind-0, using the pinned openfold3_ob0 kit.
# The build context is assembled from tools-to-implement/uplifting-biomolecular-modeling;
# no model weights are baked into this image.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG BUILD_JOBS=0
ARG OPENFOLD3_UPSTREAM_COMMIT=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
ARG OPENFOLD3_OB0_KIT_REV=unknown

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential libxrender1 libxext6 libsm6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/python.tar.gz \
      "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RL --preserve=mode,timestamps /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

ENV CUDA_HOME=/usr/local/cuda \
    PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/cache/home \
    XDG_CACHE_HOME=/cache/xdg

COPY openfold3_ob0/environment/requirements.lock /tmp/requirements.lock
COPY openfold3_ob0/stock/PINS.json openfold3_ob0/stock/check_pins.py openfold3_ob0/stock/install_upstream.py openfold3_ob0/stock/*.whl /tmp/stock/
RUN grep -v -E '^(#|openfold3==|deepspeed==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -I /tmp/stock/install_upstream.py --wheel-only \
 && J=${BUILD_JOBS:-0} && { [ "$J" -gt 0 ] || J=$(nproc --all); } && echo "deepspeed DS4Sci build: $J jobs on $(nproc --all) cores" \
 && MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 DS_NVCC_THREADS=4 \
    CC=gcc CXX=g++ LDSHARED='gcc -pthread -shared' LDCXXSHARED='g++ -pthread -shared' \
    DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0" \
    python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary deepspeed "$(grep -E '^deepspeed==' /tmp/requirements.lock)" \
 && python -c "from deepspeed.git_version_info import installed_ops; assert installed_ops.get('evoformer_attn'), installed_ops; print('DS4Sci evoformer_attn op: built')" \
 && python -m pip check \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock /root/.cache

COPY common/opt_core /kit/common/opt_core
COPY openfold3_ob0 /kit/openfold3_ob0
WORKDIR /kit/openfold3_ob0
RUN bash run.sh install

COPY openfold3_ob0/environment/Dockerfile _jitcache/openfold3_ob0-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

LABEL org.opencontainers.image.title="MN native OpenFold3 OpenBind-0 optimized" \
      org.opencontainers.image.description="OpenFold3 0.5.0 with OpenBind-0 and the pinned openfold3_ob0 optimization kit" \
      org.opencontainers.image.version="0.5.0-ob0" \
      org.opencontainers.image.revision="${OPENFOLD3_UPSTREAM_COMMIT}" \
      org.mn.openfold3.optimizer-kit-revision="${OPENFOLD3_OB0_KIT_REV}" \
      org.mn.openfold3.cuda-stack="cuda-12.8.1-torch-2.10.0-cu128"
