ARG BASE_IMAGE=mn-colabfold-opt-kit:1.6.1-jax0.5.3-cu124
FROM ${BASE_IMAGE}

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /cache/params /work

COPY support/shared/colabfold/entrypoint.sh /usr/local/bin/colabfold-entrypoint
COPY support/shared/colabfold/gpu-smoke-test.py /usr/local/bin/colabfold-gpu-smoke-test
RUN chmod +x /usr/local/bin/colabfold-entrypoint /usr/local/bin/colabfold-gpu-smoke-test

# Keep the existing `colabfold_batch ...` invocation contract. The stock image
# remains the default; this candidate opts into the kit's fast path by default.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/cache/matplotlib \
    XDG_CACHE_HOME=/cache/xdg-cache \
    COLABFOLD_DATA=/cache \
    COLABFOLD_OPT_DATA_DIR=/cache \
    COLABFOLD_OPT=fast \
    JAX_COMPILATION_CACHE_DIR=/cache/params/.jax \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /work
VOLUME ["/cache", "/work"]

ENTRYPOINT ["/usr/bin/tini", "--", "colabfold-entrypoint"]
CMD ["colabfold_batch", "--help"]
