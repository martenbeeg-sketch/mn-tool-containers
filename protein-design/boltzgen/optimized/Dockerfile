ARG BASE_IMAGE=mn-boltzgen:a07fea1-test
FROM ${BASE_IMAGE} AS boltzgen_source

# The optimization kit has a tested CUDA 13 / Torch 2.13 runtime. Preserve the
# patched application source from the existing image while rebuilding only
# the runtime around the kit's pinned stack.
FROM nvidia/cuda:13.0.1-base-ubuntu22.04

ARG BASE_IMAGE=mn-boltzgen:a07fea1-test
ARG BOLTZGEN_SOURCE_COMMIT=a07fea1153111ef453aaea7a15697ee05de9291f
ARG BOLTZGEN_OPT_KIT_REV=unknown

LABEL org.mn.boltzgen.source-commit="${BOLTZGEN_SOURCE_COMMIT}" \
      org.mn.boltzgen.opt-kit-rev="${BOLTZGEN_OPT_KIT_REV}" \
      org.mn.boltzgen.opt-stack="BoltzGen kit pinned CUDA 13 / Torch 2.13; source from ${BASE_IMAGE}"

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_OPT=/kit/boltzgen \
    BOLTZGEN_CACHE=/cache \
    HF_HOME=/cache \
    HF_HUB_DISABLE_XET=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CFLAGS=-g0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential patch \
    && rm -rf /var/lib/apt/lists/*

# Match the Python interpreter and runtime recorded by the kit.
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
    && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/python.tar.gz -C /tmp \
    && cp -a /tmp/python/. /usr/local/ \
    && rm -rf /tmp/python /tmp/python.tar.gz \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && python -V

COPY boltzgen/environment/requirements.lock /tmp/requirements.lock
COPY boltzgen /kit/boltzgen
COPY common/opt_core /kit/common/opt_core
COPY protein-design/boltzgen/optimized/boltzgen-write-all-samples.patch /tmp/boltzgen-write-all-samples.patch
COPY protein-design/boltzgen/optimized/boltzgen-a3m-refolding.patch /tmp/boltzgen-a3m-refolding.patch
COPY --from=boltzgen_source /app /app

RUN patch -p1 -d /app < /tmp/boltzgen-write-all-samples.patch \
    && patch -p1 -d /app < /tmp/boltzgen-a3m-refolding.patch \
    && rm /tmp/boltzgen-write-all-samples.patch /tmp/boltzgen-a3m-refolding.patch

# Install the kit's complete lock, then package the exact patched source tree
# carried by the existing MN BoltzGen image as the BoltzGen wheel.
RUN set -eux; \
    grep -E '^(pip|setuptools|wheel)==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps; \
    grep -v -E '^(#|boltzgen==)' /tmp/requirements.lock > /tmp/stack.txt; \
    python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt; \
    mkdir -p /tmp/boltzgen-local-wheel; \
    python -m pip wheel --no-deps --no-build-isolation /app -w /tmp/boltzgen-local-wheel; \
    wheel="$(find /tmp/boltzgen-local-wheel -maxdepth 1 -name 'boltzgen-0.3.2-*.whl' -print -quit)"; \
    test -n "$wheel"; \
    cp "$wheel" /kit/boltzgen/stock/boltzgen-0.3.2-py3-none-any.whl; \
    rm -rf /tmp/boltzgen-local-wheel /tmp/stack.txt /tmp/requirements.lock

RUN python - "$BOLTZGEN_SOURCE_COMMIT" "$BASE_IMAGE" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path("/kit/boltzgen")
wheel = root / "stock/boltzgen-0.3.2-py3-none-any.whl"
pins_path = root / "stock/PINS.json"
pins = json.loads(pins_path.read_text())
pins["wheel"] = {
    "file": "stock/boltzgen-0.3.2-py3-none-any.whl",
    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "bytes": wheel.stat().st_size,
    "source": "wheel built from the retained MN BoltzGen source tree",
}
pins["upstream"]["mn_source_commit"] = sys.argv[1]
pins["stack_of_record"]["source_image"] = sys.argv[2]
pins_path.write_text(json.dumps(pins, indent=2) + "\n")
PY

RUN set -eux; \
    python -m pip uninstall -y boltzgen; \
    python -m pip install --no-cache-dir --no-deps /kit/boltzgen/stock/boltzgen-0.3.2-py3-none-any.whl; \
    cd /kit/boltzgen; \
    bash run.sh install; \
    python -I /kit/boltzgen/stock/check_pins.py; \
    mkdir -p /cache /workspace; \
    chmod 777 /cache /workspace

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "boltzgen --help"]
