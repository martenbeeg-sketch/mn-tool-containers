# MN tool containers

Central Dockerfile repository for the local `mn-protein-design` and `mn-ligand`
apps. Images are built on the host and stay in its local Docker image store; no
registry login or container upload is needed.

## Checkout layout

Keep the three repositories beside one another:

```text
git-projects/
  mn-protein-design/
  ovo-ligand/
  mn-tool-containers/
```

The container repo owns Dockerfiles. Compose files in each app repo keep the
correct build context, because some images copy code, wrappers, or pinned tool
checkouts from that app. Clone the app repositories you plan to use before
building. The protein and ligand apps share the Biohub ESM, PeSTo, AlphaFast,
and Boltz-2 images.

## Build images

Run a full build for either app:

```bash
./build.sh mn-protein-design
./build.sh mn-ligand
```

Build selected Compose services from the app repository when needed:

```bash
docker compose -f ../mn-protein-design/docker-compose.yml build biohub-esm
docker compose -f ../ovo-ligand/docker-compose.yml build gromacs
```

BindCraft 2 is pinned to its upstream source commit and built with its dedicated
script:

```bash
./scripts/build-bindcraft2.sh
```

`./scripts/migrate-local-tags.sh` adds canonical `mn-*` tags to already-built
legacy images on this computer. It keeps the old tags in place and skips images
that already have a canonical tag.

## Image names

App manifests and Compose files use local names in the form `mn-<tool>:<tag>`.
GPU tags identify the CUDA family when that affects compatibility; versioned
tools include their upstream version or source revision. See [images.yaml](images.yaml)
for the canonical name, app, build context, and any legacy tag.

Two mn-ligand wrappers currently use locally supplied base images:
`mn-boltzina-base:latest` and `mn-plip-base:latest`. The wrapper Dockerfiles
remain here, while the base images must be built from their upstream setup or
loaded from a local archive before building those wrappers. The migration script
can retag the matching legacy base images if they are already installed.

Model weights, reference files, user data, and simulation results are not stored
in this repository or baked into these images unless an image's documented build
option explicitly requests it.

## Hardware checks

CUDA tags document the container runtime used by the recipe. They do not by
themselves certify a GPU. Run the smoke commands documented for each tool on the
target computer, especially for RTX 5090 / Blackwell support.

## Licensing

These are build recipes for third-party research tools. Review the upstream
licenses and model weight terms before publishing this repository or distributing
images. This repository does not include model weights.
