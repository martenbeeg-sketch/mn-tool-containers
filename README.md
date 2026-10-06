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

For a reusable process to benchmark separate `off`, `fast`, and `big` runtime
modes on the dual RTX 4090 host, see the
[model-container optimization playbook](OPTIMIZATION_PLAYBOOK.md).

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

`./build.sh mn-cofolding` builds the stock AF3 3.1.14 image and the single
active optimized image, `mn-cofolding-af3-memory:3.1.14-cu13`. MN Cofolding defaults to
`mn-cofolding-af3-memory:3.1.14-cu13`; set `MN_COFOLDING_AF3_IMAGE` to select
the stock image explicitly. `off`, `exact`, `fast`, and `big` select runtime behavior
within the chosen image, not different weights. `off` uses stock AF3 model
kernels; `exact` fuses the bfloat16, 128-channel pair GLU with the kit's
Pallas kernel that preserves the stock GLU operation's arithmetic for official
AlphaFold 3 weights. Exactness is operator-level and does not guarantee
bitwise-identical full predictions. `fast` enables guarded FlashPairformer
triangle kernels. In the default memory image, `big` uses 256-row pair-transition shards
and processes diffusion samples one at a time. Official AlphaFold 3 `big` also
row-shards the trunk pair representation across GPUs 0 and 1. This is a
reviewed 3.1.14 port of the kit's rowpair recipe with an exact source pin;
diffusion, confidence, and output use the stock 3.1.14 implementation. Other
AF3-family models remain on GPU 0. Superseded AF3 candidates are preserved under
`protein-design/cofolding-af3/archive/` for provenance, but are not active
build targets. The stock Dockerfile and active image tag remain preserved.
These images share unchanged layers on the same computer.
On another computer, transfer the selected image with `docker save` and
`docker load`, or transfer the pinned patch source tree to build it there.
Guarded kernels fall back to stock operations when a GPU architecture has no
supported tile policy. Exact mode refuses a changed AF3 source or JAX/Tokamax
pin rather than silently using another implementation.
The active recipe has separate host profiles under
`protein-design/cofolding-af3/profiles/` for RTX 4090 (`sm_89`) and RTX 5090
(`sm_120`). The 5090 profile is a deployment target, not a claim of runtime
validation; source it only on the corresponding host and benchmark it there.

ESMFold2 uses a separate optimized image, built with `./build.sh esmfold2-opt`.
It packages the pinned full Biohub ESMFold2 model and its ESM-C 6B language
model runtime, with CUDA extensions compiled for Ada SM89 and Blackwell SM120.
MN Cofolding mounts the verified full-model and ESM-C weights from
`/mnt/db/reference_files/biohub-esm` into the cache, so the image contains no
weights and does not duplicate the existing 25 GB checkpoint set. The app
offers the kit's `off`, `exact`, `fast`, and `big` modes; only `big` reserves
GPUs 0 and 1. RTX 4090 execution is tested locally. The RTX 5090 build target
is included, but the local host has no 5090 for runtime validation; its kit
kernels remain marked uncertified until tested on that card. The custom
multi-architecture wheels are built from the pinned sources because the kit's
prebuilt wheels only cover its documented H100/A100 stacks. On the 4090/5090
small shared-memory classes, the image records the kit's T3 tile-table and
T6/T10 transition fallbacks as class-specific exclusions; the kit then uses
stock tiles and its exact T1 transition path while keeping the remaining mode
levers active. Fast mode also selects a 32-key tile for the fused MSA pair
attention on these classes; the kit's default 64-key tile exceeds the RTX 4090
per-block shared-memory limit. This keeps the fused path active and is reported
in the run counters.

Chai-1 uses its own pinned `chai_lab 0.6.1` optimization-kit image, built with
`./build.sh chai1-opt`. It keeps the eight stock Chai checkpoint files outside
the image under `/mnt/db/reference_files/chai1`, where MN Cofolding mounts them
at `/reference/chai1`; the image does not bundle weights. The app offers the
kit's `off`, `exact`, `fast`, and `big` modes. The RTX 4090 configuration is
tested locally; Chai's `big` mode is a single-GPU memory path and does not shard
across the two 4090s. An RTX 5090 config is included for deployment, but it has
not been runtime-validated on this host.

Boltz-2 keeps its original `mn-boltz2:cu128` stock image beside the single
`mn-boltz2-opt:2.2.1-cu130` image, which packages the upstream optimization kit with
selectable `off`, `fast`, and `big` modes (and its additional `exact` mode).
Its RTX 4090 and RTX 5090 host profiles live under `shared/boltz2/profiles/`;
they select the same image and keep the hardware target separate from the
recipe.
Build it from this repository with:

```bash
./build.sh boltz2-opt
```

The image accepts existing `boltz predict INPUT --out_dir OUT ...` commands.
Set `MN_BOLTZ2_MODE=off|fast|big` in the container environment; `off` is the
default. The old apps' result readers expect `boltz_results_*/predictions/`,
so optimized predictions are mirrored there while the kit's per-seed output
and worker log remain under `by_seed/` and `*_worker.log`.

For a direct run, mount the existing Boltz cache and a working directory:

```bash
docker run --rm --gpus all \
  -v /mnt/db/reference_files/boltz_models:/cache \
  -v "$PWD/run":/work \
  -e BOLTZ_CACHE=/cache -e MN_BOLTZ2_MODE=fast \
  mn-boltz2-opt:2.2.1-cu130 predict /work/input.yaml --out_dir /work/output \
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --accelerator gpu --override
```

The image also exposes the kit commands, such as `check --mode fast` and
`pred --mode big --input INPUT --out_dir OUT`. Optimized kernels compile on
first use; a writable persistent Triton cache avoids recompilation between
runs. App calls reuse a GPU/stack-keyed Triton cache under
`$BOLTZ_CACHE/boltz2_opt_jit/`. The kit's stated floor is compute capability 8.0 and driver 580 or newer.
Optimized runs allow safe per-kernel fallbacks by default: if a kernel is
unsupported or fails its gate, Boltz uses the stock operation and records the
decision in `pred_worker.log`. Set `MN_BOLTZ2_ALLOW_PARTIAL=0` to make any
partial optimization return a nonzero status instead.
It publishes measurements for A100 and H100; RTX 4090/5090 mode performance
has to be measured on those cards. This image uses the kit's pinned Python 3.11
and PyTorch CUDA 13.0 environment; the original image remains on Python 3.10 and
PyTorch CUDA 12.8.

The guards apply per engine and channel width, so a prediction can mix fused
and stock blocks. On RTX 4090 (compute capability 8.9), chunked tiles now cover
256- and 512-channel pair blocks. Non-power-of-two widths such as 384 stay on
stock because Pallas Triton cannot lower those channel extents. Boltz-2's
32-wide attention head is supported. Chai-1 and RoseTTAFold3 attention remain
stock because their projections have different semantics; RoseTTAFold3
triangle multiplication keeps its length normalization in the fused path.
Protenix2 uses fused triangle multiplication, while its 256-channel attention
stays stock after an end-to-end smoke prediction shifted pTM from 0.67 to 0.47
with that attention fused. Its smaller attention blocks still fuse. The
expanded kernels have prediction smoke coverage; this does not claim that
`fast` or `big` is faster for every engine or sequence length.

RF3 keeps `mn-foundry:cu128` as the default stock path. The separate
`mn-rf3-opt:0.2.1-dev13-cu130` candidate is built from the RoseTTAFold3 kit's
pinned Dockerfile with `./build.sh rf3-opt`. The app can select the kit's
`off`, `exact`, `fast`, or `big` mode with `RF3_OPT_MODE` after selecting the
candidate through `RF3_CLI_IMAGE`. Multi-GPU sharding is available only in
`big` mode through `RF3_OPT_N_GPU`. The kit stack differs from the existing
Foundry image's pins, so performance and output equivalence still need to be
measured on the target RTX 4090 workload.

Keep the original tool Dockerfile and image tag, and promote only one tested
optimized recipe per engine. Keep superseded candidates under that engine's
`archive/` directory. Hardware-specific tuning belongs in the engine's
`profiles/` directory, so an RTX 5090 profile can evolve without creating a
second active Dockerfile.

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

The RF3 optimizer image is built directly from the pinned kit checkout with
`./build.sh rf3-opt`; it uses the kit's Foundry 0.2.1.dev13 stack and keeps the
existing `mn-foundry:cu128` image available for stock runs.

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
