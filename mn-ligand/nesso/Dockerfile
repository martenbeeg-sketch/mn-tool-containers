FROM mn-biohub-esm:cu128

ARG NESSO_COMMIT=64e8a92b48d1dd3b0b33a4b64b613758fba04a17
LABEL org.opencontainers.image.title="mn-ligand Nesso-1 affinity adapter" \
      org.opencontainers.image.source="https://github.com/recursionpharma/nesso" \
      org.opencontainers.image.revision="${NESSO_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /opt/nesso
RUN git init . \
    && git remote add origin https://github.com/recursionpharma/nesso.git \
    && git fetch --depth 1 origin "${NESSO_COMMIT}" \
    && git checkout --detach FETCH_HEAD

# The CUDA 12.8 base already carries its Blackwell-compatible PyTorch,
# Transformers, RDKit, and scientific stack. Installing the local, pinned
# source adds Nesso and its remaining Python dependencies without replacing
# that GPU runtime.
RUN python -m pip install --no-cache-dir . \
    && rm -rf .git

ENTRYPOINT ["nesso"]
CMD ["--help"]
