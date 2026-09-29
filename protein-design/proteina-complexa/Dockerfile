FROM nvcr.io/nvidia/pytorch:24.08-py3

ARG PROTEINA_COMPLEXA_COMMIT=5ae24b055d828918296f2aad63616b1f4cc0e491

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["bash", "-lc"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends git patch rsync libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/protein-foundation-models
COPY --from=complexa / /workspace/protein-foundation-models/
COPY patches/proteina-complexa.patch /tmp/proteina-complexa.patch
RUN patch --batch --forward -p1 < /tmp/proteina-complexa.patch \
    && bash /workspace/protein-foundation-models/env/build_uv_env.sh --root /workspace/ \
    && rm -f /tmp/proteina-complexa.patch

ENV IGNOREEOF=10 \
    SC_EXEC=/usr/local/bin/sc \
    DSSP_EXEC=/usr/local/bin/dssp \
    RF3_CKPT_PATH=/workspace/protein-foundation-models/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt \
    RF3_EXEC_PATH=/workspace/.venv/bin/rf3 \
    AF2_DIR=/workspace/protein-foundation-models/community_models/ckpts/AF2 \
    PYTHONPATH=/workspace/protein-foundation-models/src

COPY --from=complexa /env/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
