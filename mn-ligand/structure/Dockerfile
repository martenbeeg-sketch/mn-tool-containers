FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="mn-structure"
LABEL org.opencontainers.image.description="Structure preparation container for mn-ligand, built from Ligand-X base.yml"

COPY --from=app --chown=$MAMBA_USER:$MAMBA_USER /containers/structure/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -a -y \
    && rm -f /tmp/environment.yml

WORKDIR /work
ENTRYPOINT []
