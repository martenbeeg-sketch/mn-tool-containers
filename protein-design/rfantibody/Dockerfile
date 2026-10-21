FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DGL_VERSION=v2.4.0
ARG DGL_COMMIT=c6c874bf7ea085beb04ea1487cfd216a0bacd6c1

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/rfantibody/.venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    RFANTIBODY_ROOT=/opt/rfantibody \
    RFANTIBODY_WEIGHTS=/ref/rfantibody \
    TORCH_CUDA_ARCH_LIST="8.9;12.0+PTX" \
    CMAKE_CUDA_ARCHITECTURES="89;120" \
    CUDAARCHS="89;120" \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/rfantibody/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git libgl1 libglib2.0-0 \
    libgomp1 libopenblas-dev libomp-dev ninja-build pkg-config \
    python3.10 python3.10-dev python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/
RUN uv pip install --system --python /usr/bin/python3.10 "cmake==3.31.6"
WORKDIR /opt/rfantibody
COPY . /opt/rfantibody

# Keep the pinned RFantibody Python environment while substituting the old
# cu118 Torch/DGL wheels with CUDA 12.8 builds for Ada and Blackwell.
RUN uv sync --frozen --no-dev \
      --no-install-package torch \
      --no-install-package torchvision \
      --no-install-package torchaudio \
      --no-install-package dgl \
      --no-install-package cuda-python \
      --no-install-package triton \
      --no-install-package nvidia-cublas-cu11 \
      --no-install-package nvidia-cuda-cupti-cu11 \
      --no-install-package nvidia-cuda-nvrtc-cu11 \
      --no-install-package nvidia-cuda-runtime-cu11 \
      --no-install-package nvidia-cudnn-cu11 \
      --no-install-package nvidia-cufft-cu11 \
      --no-install-package nvidia-curand-cu11 \
      --no-install-package nvidia-cusolver-cu11 \
      --no-install-package nvidia-cusparse-cu11 \
      --no-install-package nvidia-nccl-cu11 \
      --no-install-package nvidia-nvtx-cu11 && \
    uv pip install --python /opt/rfantibody/.venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128"

# DGL's published 2.4.0 binary is tied to Torch 2.3/cu118. Build its CUDA
# extension against the same Torch 2.7/cu128 stack, embedding both sm_89 and
# sm_120 code so this one image runs on RTX 4090 and RTX 5090.
RUN git clone --depth 1 --branch "${DGL_VERSION}" --recurse-submodules \
      https://github.com/dmlc/dgl.git /tmp/dgl && \
    test "$(git -C /tmp/dgl rev-parse HEAD)" = "${DGL_COMMIT}" && \
    cmake -S /tmp/dgl -B /tmp/dgl/build \
      -DUSE_CUDA=ON -DCUDA_ARCH_NAME=Manual \
      "-DCMAKE_CUDA_ARCHITECTURES=89;120" \
      "-DCUDA_ARCH_BIN=89;120" "-DCUDA_ARCH_PTX=120" \
      -DBUILD_TYPE=release && \
    cmake --build /tmp/dgl/build --parallel 8 && \
    DGL_LIBRARY_PATH=/tmp/dgl/build \
      uv pip install --python /opt/rfantibody/.venv/bin/python --no-cache-dir --no-deps /tmp/dgl/python && \
    rm -rf /tmp/dgl

RUN mkdir -p /opt/rfantibody/weights /ref/rfantibody && \
    chmod -R a+rX /opt/rfantibody && \
    chmod a+w /opt/rfantibody

CMD ["bash"]
