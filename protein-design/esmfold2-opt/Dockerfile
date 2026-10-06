ARG BASE_IMAGE=mn-esmfold2-opt-kit:0.1.0-cu130
FROM ${BASE_IMAGE}

COPY protein-design/esmfold2-opt/mn-esmfold2-runner /usr/local/bin/mn-esmfold2-runner
COPY protein-design/esmfold2-opt/registry.py /kit/esmfold2/opt/esmfold2_opt/registry.py
COPY protein-design/esmfold2-opt/ef2_msa_v2.py /kit/esmfold2/opt/forward/fast_inference/driver/ef2_msa_v2.py
RUN chmod 0755 /usr/local/bin/mn-esmfold2-runner

LABEL org.opencontainers.image.title="MN ESMFold2 optimized"
LABEL org.opencontainers.image.description="Pinned ESMFold2 acceleration kit with RTX 4090 and RTX 5090 CUDA targets"
