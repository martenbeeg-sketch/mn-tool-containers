FROM mn-biohub-esm:cu128

WORKDIR /opt/PeSTo
ARG PESTO_COMMIT=ba651aa29aaa839d0ee2c458dee90ca1f934d90a
COPY scripts/fetch-pinned-source.sh /usr/local/bin/mn-fetch-source
RUN chmod +x /usr/local/bin/mn-fetch-source \
    && mn-fetch-source https://github.com/LBM-EPFL/PeSTo.git "${PESTO_COMMIT}" /opt/PeSTo/source \
    && cp -a /opt/PeSTo/source/src ./src \
    && mkdir -p ./model/i_v4_1 \
    && cp /opt/PeSTo/source/model/save/i_v4_1_2021-09-07_11-21/config.py ./model/i_v4_1/config.py \
    && cp /opt/PeSTo/source/model/save/i_v4_1_2021-09-07_11-21/model.py ./model/i_v4_1/model.py \
    && rm -rf /opt/PeSTo/source
COPY support/shared/pesto/predict.py ./predict.py

ENV PYTHONPATH=/opt/PeSTo
ENTRYPOINT ["python", "/opt/PeSTo/predict.py"]
