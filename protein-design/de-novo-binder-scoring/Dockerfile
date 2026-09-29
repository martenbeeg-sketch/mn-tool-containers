ARG BASE_IMAGE

FROM debian:bookworm-slim AS scoring-source
ARG DE_NOVO_SCORING_COMMIT=3d4fc2710ae23e578190503164777240f99ce976
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/fetch-pinned-source.sh /usr/local/bin/mn-fetch-source
COPY patches/de-novo-binder-scoring.patch /opt/mn-tool-patches/de-novo-binder-scoring.patch
RUN chmod +x /usr/local/bin/mn-fetch-source \
    && mn-fetch-source https://github.com/DigBioLab/de_novo_binder_scoring.git \
        "${DE_NOVO_SCORING_COMMIT}" /opt/de_novo_binder_scoring de-novo-binder-scoring.patch \
    && rm -f /opt/de_novo_binder_scoring/analysis/data/prepared_training_dataset.csv

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="mn-protein-design scoring adapters" \
      org.opencontainers.image.source="https://github.com/DigBioLab/de_novo_binder_scoring" \
      org.opencontainers.image.revision="3d4fc2710ae23e578190503164777240f99ce976"

COPY --from=scoring-source /opt/de_novo_binder_scoring /opt/de_novo_binder_scoring
COPY support/protein-design/ipsae/run_ipsae.py /opt/mn-ipsae/run_ipsae.py
ENV PYTHONPATH=/opt/de_novo_binder_scoring
