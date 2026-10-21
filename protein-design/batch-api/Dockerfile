ARG BASE_IMAGE=mn-rf3-opt:0.2.1-dev13-cu130
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG BASE_IMAGE_ID=unknown
ARG BATCH_ENGINE=rf3_foundry
LABEL org.mn.batch.api="mn-predict/v1" \
      org.mn.batch.engine="${BATCH_ENGINE}" \
      org.mn.batch.base-image="${BASE_IMAGE}" \
      org.mn.batch.base-image-id="${BASE_IMAGE_ID}"
ENV MN_BATCH_ENGINE=${BATCH_ENGINE} \
    MN_BATCH_BASE_IMAGE=${BASE_IMAGE} \
    MN_BATCH_BASE_IMAGE_ID=${BASE_IMAGE_ID} \
    MN_BATCH_API_VERSION=mn-predict/v1 \
    PYTHONPATH=/opt

COPY shared/batch_api /opt/mn_batch_api
COPY protein-design/batch-api/mn-predict /usr/local/bin/mn-predict
RUN chmod 0755 /usr/local/bin/mn-predict

ENTRYPOINT ["/usr/local/bin/mn-predict"]
CMD ["--help"]
