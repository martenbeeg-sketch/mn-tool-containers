FROM eclipse-temurin:17-jdk-jammy@sha256:723151f3fc88ca2060153ee08ab8dbbea7983d6ed6f2622fe440acf178737c94 AS builder

ARG P2RANK_COMMIT=d8c8e0d870f79a36b7dbb176edf8c019b6cc789c
ARG P2RANK_REPOSITORY=https://github.com/rdk/p2rank.git

WORKDIR /src/p2rank
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && git init . \
    && git remote add origin "${P2RANK_REPOSITORY}" \
    && git fetch --depth 1 origin "${P2RANK_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && ./gradlew --no-daemon clean assemble

FROM eclipse-temurin:17-jre-jammy@sha256:475d8e96b4b2bfe08999e5e854755c773af1581acdf959a4545d88f0696a2339

ARG P2RANK_COMMIT=d8c8e0d870f79a36b7dbb176edf8c019b6cc789c

LABEL org.opencontainers.image.title="mn-p2rank"
LABEL org.opencontainers.image.description="Pinned CPU P2Rank pocket-detection container for mn-ligand"
LABEL org.opencontainers.image.source="https://github.com/rdk/p2rank"
LABEL org.opencontainers.image.revision="${P2RANK_COMMIT}"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=builder /src/p2rank/distro /opt/p2rank
RUN chmod +x /opt/p2rank/prank \
    && ln -s /opt/p2rank/prank /usr/local/bin/prank

WORKDIR /work
ENTRYPOINT ["/opt/p2rank/prank"]
CMD ["help"]
