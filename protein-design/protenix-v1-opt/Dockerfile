ARG BASE_IMAGE=mn-protenix-v1-kit:1.1.0-cu130
FROM ${BASE_IMAGE}

COPY protein-design/protenix-v1-opt/profiles/rtx4090.env /kit/protenix_v1/configs/rtx4090.env
COPY protein-design/protenix-v1-opt/profiles/rtx5090.env /kit/protenix_v1/configs/rtx5090.env

ENV PROTENIX_ROOT_DIR=/reference/protenix \
    MODEL_OPT_JIT_ROOT=/cache/protenix_v1/jit

LABEL org.opencontainers.image.title="MN Protenix V1 optimized" \
      org.opencontainers.image.description="Protenix 1.1.0 with the pinned Protenix V1 optimization kit; RTX 4090 validated and RTX 5090 deployment profile" \
      org.opencontainers.image.version="1.1.0-cu130"

ENTRYPOINT ["bash", "/kit/protenix_v1/run.sh"]
