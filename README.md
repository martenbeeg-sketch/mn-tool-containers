# MN tool containers

Central Dockerfile repository for the local `mn-protein-design` and `mn-ligand`
apps. Tool source repositories are fetched at pinned commits during image
builds; they do not need to be cloned beside the apps. Build patches and small
app adapters live here. App-specific runtime data, model weights, and caches
remain separate inputs.

## Checkout layout

Keep the three repositories beside one another:

```text
git-projects/
  mn-protein-design/
  ovo-ligand/
  mn-tool-containers/
```

The container repo owns Dockerfiles, pinned source revisions, and local source
patches. App Compose files use this repository as their Docker build context and
pass a filtered app context only for launchers and environment files. Protein
and ligand apps share PeSTo, AlphaFast, and Boltz-2 images. Protein Design uses
a versioned Biohub ESM image; existing ligand images keep their older Biohub
base image.

Builds fetch upstream source at pinned revisions. The Protein Design Biohub
ESM image, `mn-biohub-esm:3.4.1-cu128`, installs pinned PyPI releases of
PyTorch and `esm` and fetches the matching binder-design tutorial from a pinned
public Biohub ESM commit. It does not require a Biohub source checkout or
private `Biohub/transformers` access. `build.sh` reuses an existing versioned
image by default and rebuilds it only when `MN_REBUILD_BIOHUB_ESM=1` is set.
Transfer the image separately when installing on another computer (`docker
save` / `docker load`). Scoring
wrappers and selected ligand images extend separately licensed base images; see
the requirements below.

## Build images

Run a full build for either app:

```bash
./build.sh mn-protein-design
./build.sh mn-ligand
```

Check the pinned recipes and app Compose contexts with:

```bash
python -m pytest -q
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

Some recipes extend locally supplied base images. Protein Design scoring
wrappers need `mn-python-structure:latest`, `mn-bindcraft:latest`, and
`mn-ipsae:latest`; Boltzina and PLIP retain their upstream base-image
requirements. These bases contain separately licensed tools and can be loaded
from image archives. The wrappers and pinned app source are in this repository.

Model weights, reference files, user data, and simulation results are not stored
in this repository or baked into these images unless an image's documented build
option explicitly requests it.
The Protein Design app keeps its public benchmark CSV under the managed app
workdir and fetches it from a pinned source revision on first use.

## Hardware checks

CUDA tags document the container runtime used by the recipe. They do not by
themselves certify a GPU. Run the smoke commands documented for each tool on the
target computer, especially for RTX 5090 / Blackwell support.

## Licensing

These are build recipes for third-party research tools. Review the upstream
licenses and model weight terms before publishing this repository or distributing
images. This repository does not include model weights.
