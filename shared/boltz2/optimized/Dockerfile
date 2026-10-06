# Boltz-2 2.2.1 with the selectable off/fast/big optimization kit.
# Build with the `boltzkit` named context pointing at
# tools-to-implement/uplifting-biomolecular-modeling.
FROM python:3.11.12-slim-bookworm

LABEL org.opencontainers.image.title="mn-boltz2-opt"
LABEL org.opencontainers.image.description="Boltz-2 2.2.1 with selectable stock, fast, and memory modes"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=boltzkit boltz2/environment/requirements.lock /tmp/requirements.lock
COPY --from=boltzkit boltz2/stock/boltz-2.2.1-py3-none-any.whl /tmp/boltz-2.2.1-py3-none-any.whl
RUN grep -E '^(pip|wheel)==' /tmp/requirements.lock > /tmp/pip.txt \
 && python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/pip.txt \
 && printf 'setuptools==82.0.1\n' > /tmp/build-constraints.txt \
 && grep -v -E '^(#|pip==|wheel==|boltz==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps --build-constraint /tmp/build-constraints.txt -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/boltz-2.2.1-py3-none-any.whl \
 && rm -f /tmp/requirements.lock /tmp/pip.txt /tmp/build-constraints.txt /tmp/stack.txt /tmp/boltz-2.2.1-py3-none-any.whl

COPY --from=boltzkit common/opt_core /kit/common/opt_core
COPY --from=boltzkit boltz2 /kit/boltz2
WORKDIR /kit/boltz2
RUN bash run.sh install

ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY shared/boltz2/optimized/entrypoint.py /usr/local/bin/mn-boltz2-entrypoint.py
COPY support/protein-design/boltz2/prepare_inputs.py /opt/mn-protein-design/prepare_inputs.py
WORKDIR /work
ENTRYPOINT ["python3.11", "/usr/local/bin/mn-boltz2-entrypoint.py"]
CMD ["--help"]
