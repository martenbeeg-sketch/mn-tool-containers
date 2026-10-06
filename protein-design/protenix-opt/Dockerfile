# Selectable off / exact / fast / big modes for the pinned Protenix 2.0.0 image.
# Build with the `protenixkit` named context pointing at
# tools-to-implement/uplifting-biomolecular-modeling.
ARG PROTENIX_BASE_IMAGE=mn-protenix:cu128
FROM ${PROTENIX_BASE_IMAGE}

LABEL org.opencontainers.image.title="mn-protenix-opt" \
      org.opencontainers.image.description="Protenix 2.0.0 with selectable off, exact, fast, and big inference modes"

COPY --from=protenixkit protenix_v2/opt /opt/protenix-opt/protenix_v2/opt
COPY --from=protenixkit protenix_v2/stock /opt/protenix-opt/protenix_v2/stock
COPY --from=protenixkit common/opt_core /opt/protenix-opt/common/opt_core
COPY protein-design/protenix-opt/LICENSE.optimizer-kit /opt/protenix-opt/LICENSE.optimizer-kit
COPY protein-design/protenix-opt/protenix-opt /usr/local/bin/protenix-opt
COPY protein-design/protenix-opt/protenix-opt.py /usr/local/libexec/mn-protenix-opt.py

# The editable installs ship the kit's interpreter-start .pth hook. The CLI
# path is used for normal runs; the .pth hook is also needed by big's TP ranks.
RUN python -m pip install --no-cache-dir --no-deps \
        -e /opt/protenix-opt/common/opt_core \
        -e /opt/protenix-opt/protenix_v2/opt \
    && chmod 0755 /usr/local/bin/protenix-opt

ENV PROTENIX_ROOT_DIR=/ref/protenix \
    PROTENIX_OPT_MODE=off \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/protenix-opt"]
CMD ["--help"]
