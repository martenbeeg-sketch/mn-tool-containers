ARG BASE_IMAGE=mn-chai1-opt-kit:0.6.1-cu130
FROM ${BASE_IMAGE}

COPY protein-design/chai1-opt/mn-chai1-runner /usr/local/bin/mn-chai1-runner
COPY protein-design/chai1-opt/mn_chai1_inputs.py /usr/local/bin/mn_chai1_inputs.py
RUN chmod 0755 /usr/local/bin/mn-chai1-runner

LABEL org.opencontainers.image.title="MN Chai-1 optimized"
LABEL org.opencontainers.image.description="Pinned Chai-1 optimization kit with RTX 4090 and RTX 5090 deployment configs"
