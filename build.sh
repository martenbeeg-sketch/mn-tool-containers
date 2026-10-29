#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS="$(cd "$ROOT/.." && pwd)"
PROTEIN_REPO="$PROJECTS/mn-protein-design"
LIGAND_REPO="$PROJECTS/ovo-ligand"
AF3_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
BOLTZ2_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
PROTENIX_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
RF3_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
COLABFOLD_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
AF2IG_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
ESMFOLD2_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
CHAI1_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
ATLASFOLD_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
OPENFOLD3_OB0_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
OPENDDE_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
AF3_TORCH_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
PROTENIX_V1_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
BOLTZGEN_OPT_TOOLS="$PROJECTS/tools-to-implement/uplifting-biomolecular-modeling"
RFANTIBODY_TOOLS="$PROJECTS/tools-to-implement/RFantibody"

usage() {
  printf 'Usage: %s {mn-protein-design|mn-ligand|mn-cofolding|bioemu2|boltz2-opt|rf3-opt|boltz2-opt-batch|boltz2-opt-batch2|rf3-opt-batch|af3-torch-opt-batch|af3-torch-opt-batch2|af3-torch-opt-batch3|alphafold3-batch|openfold3-batch|esmfold2-opt-batch|esmfold2-fast-opt-batch|chai1-opt-batch|atlasfold-opt-batch|atlasfold-opt-batch2|colabfold-batch|colabfold-batch2|colabfold-batch3|colabfold-opt-batch|colabfold-opt-batch2|colabfold-opt-batch3|openfold3-ob0-batch|openfold3-ob0-batch2|opendde-opt-batch|opendde-opt-batch2|opendde-opt-batch3|protenix-v1-opt-batch|protenix-v1-opt-batch2|protenix-v2-opt-batch|protenix-v2-opt-batch2|protenix-v2-opt-sm120|boltzgen-opt-batch|boltzgen-opt-batch2|boltzgen-opt-batch3|alphafast-batch|boltz2-stock-batch|biohub-esm-batch|openfold3-p2-batch|openfold3-p2-batch2|pxdesign-v05-batch|pxdesign-v05-batch2|pxdesign-v05-batch3|pxdesign-v05-batch4|pxdesign-v05-batch5|pxdesign-v05-batch6|protenix-v2-stock-batch|protenix-v2-stock-batch2|foundry-rf3-batch|foundry-rf3-batch2|batch-api|colabfold-opt|af2ig-opt|esmfold-v1|esmfold2-opt|chai1-opt|atlasfold-opt|openfold3-ob0|opendde-opt|protenix-v1-opt|af3-torch-opt|boltzgen-opt|rfantibody|genie3|pxdesign}\n' "$0" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

APP="$1"
if [[ "$APP" == "mn-protein-design" || "$APP" == "mn-ligand" ]]; then
  for repo in "$PROTEIN_REPO" "$LIGAND_REPO"; do
    if [[ ! -d "$repo/.git" ]]; then
      printf 'Required sibling checkout is missing: %s\n' "$repo" >&2
      exit 1
    fi
  done
fi

compose() {
  docker compose -f "$1/docker-compose.yml" build "${@:2}"
}

require_images() {
  local image
  for image in "$@"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      printf 'Required base image is missing: %s\n' "$image" >&2
      printf 'Load or build that licensed base image, then rerun this command.\n' >&2
      exit 1
    fi
  done
}

build_batch_api_overlay() {
  local base_image="$1" engine="$2" output_image="$3" base_image_id
  require_images "$base_image"
  base_image_id="$(docker image inspect --format '{{.Id}}' "$base_image")"
  docker buildx build --load \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "BASE_IMAGE_ID=$base_image_id" \
    --build-arg "BATCH_ENGINE=$engine" \
    -f "$ROOT/protein-design/batch-api/Dockerfile" \
    -t "$output_image" \
    "$ROOT"
}

build_biohub_esm() {
  if [[ "${MN_REBUILD_BIOHUB_ESM:-0}" != "1" ]] && docker image inspect mn-biohub-esm:3.4.1-cu128 >/dev/null 2>&1; then
    printf 'Using existing mn-biohub-esm:3.4.1-cu128 image. Transfer this image separately when installing on another computer.\n'
    printf 'Set MN_REBUILD_BIOHUB_ESM=1 to rebuild it from pinned public PyPI packages and Biohub ESM source.\n'
    return
  fi
  compose "$PROTEIN_REPO" biohub-esm
}

build_protenix_v1_opt() {
  if [[ ! -f "$PROTENIX_V1_OPT_TOOLS/protenix_v1/environment/Dockerfile" \
     || ! -f "$PROTENIX_V1_OPT_TOOLS/protenix_v1/run.sh" \
     || ! -d "$PROTENIX_V1_OPT_TOOLS/common/opt_core/opt_core" \
     || ! -f "$ROOT/protein-design/protenix-v1-opt/profiles/rtx4090.env" \
     || ! -f "$ROOT/protein-design/protenix-v1-opt/profiles/rtx5090.env" ]]; then
    printf 'Required Protenix V1 optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$PROTENIX_V1_OPT_TOOLS" >&2
    exit 1
  fi
  protenix_v1_build_context="$(mktemp -d)"
  trap 'rm -rf "$protenix_v1_build_context"' EXIT
  mkdir -p "$protenix_v1_build_context/common" "$protenix_v1_build_context/_jitcache"
  cp -a "$PROTENIX_V1_OPT_TOOLS/protenix_v1" "$protenix_v1_build_context/protenix_v1"
  cp -a "$PROTENIX_V1_OPT_TOOLS/common/opt_core" "$protenix_v1_build_context/common/opt_core"
  cp "$ROOT/protein-design/protenix-v1-opt/profiles/rtx4090.env" "$protenix_v1_build_context/protenix_v1/configs/rtx4090.env"
  cp "$ROOT/protein-design/protenix-v1-opt/profiles/rtx5090.env" "$protenix_v1_build_context/protenix_v1/configs/rtx5090.env"
  if [[ -d "$PROTENIX_V1_OPT_TOOLS/_jitcache" ]]; then
    cp -a "$PROTENIX_V1_OPT_TOOLS/_jitcache/." "$protenix_v1_build_context/_jitcache/"
  fi
  if ! compgen -G "$protenix_v1_build_context/_jitcache/protenix_v1-*-jit.tar" >/dev/null; then
    tar -cf "$protenix_v1_build_context/_jitcache/protenix_v1-empty-jit.tar" --files-from /dev/null
  fi
  docker buildx build --load \
    --build-arg BUILD_JOBS="${MN_PROTENIX_V1_BUILD_JOBS:-0}" \
    -f "$protenix_v1_build_context/protenix_v1/environment/Dockerfile" \
    -t mn-protenix-v1-kit:1.1.0-cu130 \
    "$protenix_v1_build_context"
  docker buildx build --load \
    --build-arg BASE_IMAGE=mn-protenix-v1-kit:1.1.0-cu130 \
    -f "$ROOT/protein-design/protenix-v1-opt/Dockerfile" \
    -t mn-protenix-v1-opt:1.1.0-cu130 \
    "$ROOT"
}

build_boltzgen_opt() {
  if ! docker image inspect mn-boltzgen:a07fea1-test >/dev/null 2>&1; then
    printf 'Required stock image is missing: mn-boltzgen:a07fea1-test. Build or load that image first.\n' >&2
    exit 1
  fi
  if [[ ! -f "$BOLTZGEN_OPT_TOOLS/boltzgen/run.sh" \
     || ! -d "$BOLTZGEN_OPT_TOOLS/boltzgen/opt/boltzgen_opt" \
     || ! -d "$BOLTZGEN_OPT_TOOLS/common/opt_core/opt_core" \
     || ! -f "$ROOT/protein-design/boltzgen/optimized/Dockerfile" \
     || ! -f "$ROOT/protein-design/boltzgen/profiles/rtx4090.env" \
     || ! -f "$ROOT/protein-design/boltzgen/profiles/rtx5090.env" ]]; then
    printf 'Required BoltzGen kit or optimized recipe is missing. Check out uplifting-biomolecular-modeling and the BoltzGen optimized recipe.\n' >&2
    exit 1
  fi
  boltzgen_build_context="$(mktemp -d)"
  trap 'rm -rf "$boltzgen_build_context"' EXIT
  mkdir -p "$boltzgen_build_context/boltzgen" "$boltzgen_build_context/common" \
    "$boltzgen_build_context/protein-design/boltzgen/optimized"
  cp -a "$BOLTZGEN_OPT_TOOLS/boltzgen/." "$boltzgen_build_context/boltzgen/"
  cp -a "$BOLTZGEN_OPT_TOOLS/common/opt_core" "$boltzgen_build_context/common/opt_core"
  cp "$ROOT/protein-design/boltzgen/optimized/Dockerfile" \
    "$boltzgen_build_context/protein-design/boltzgen/optimized/Dockerfile"
  cp "$ROOT/protein-design/boltzgen/optimized/boltzgen-write-all-samples.patch" \
    "$boltzgen_build_context/protein-design/boltzgen/optimized/boltzgen-write-all-samples.patch"
  cp "$ROOT/protein-design/boltzgen/optimized/boltzgen-a3m-refolding.patch" \
    "$boltzgen_build_context/protein-design/boltzgen/optimized/boltzgen-a3m-refolding.patch"
  cp "$ROOT/protein-design/boltzgen/profiles/rtx4090.env" \
    "$boltzgen_build_context/boltzgen/configs/rtx4090.env"
  cp "$ROOT/protein-design/boltzgen/profiles/rtx5090.env" \
    "$boltzgen_build_context/boltzgen/configs/rtx5090.env"
  boltzgen_opt_kit_rev="$(git -C "$BOLTZGEN_OPT_TOOLS" rev-parse HEAD 2>/dev/null || printf unknown)"
  docker buildx build --load \
    --build-arg BASE_IMAGE=mn-boltzgen:a07fea1-test \
    --build-arg BOLTZGEN_SOURCE_COMMIT=a07fea1153111ef453aaea7a15697ee05de9291f \
    --build-arg BOLTZGEN_OPT_KIT_REV="$boltzgen_opt_kit_rev" \
    -f "$boltzgen_build_context/protein-design/boltzgen/optimized/Dockerfile" \
    -t mn-boltzgen-opt:a07fea1-cu130 \
    "$boltzgen_build_context"
}

build_rfantibody() {
  local image_tag="$1"
  local dockerfile_name="$2"
  local source_dir="${RFANTIBODY_SOURCE_DIR:-$RFANTIBODY_TOOLS}"
  MN_RFAB_BUILD_CONTEXT="$(mktemp -d)"
  trap 'rm -rf "${MN_RFAB_BUILD_CONTEXT:-}"' EXIT
  if [[ -f "$source_dir/pyproject.toml" && -f "$source_dir/uv.lock" ]]; then
    tar -C "$source_dir" --exclude=.git -cf - . | tar -C "$MN_RFAB_BUILD_CONTEXT" -xf -
  else
    source_dir="$MN_RFAB_BUILD_CONTEXT/source"
    bash "$ROOT/scripts/fetch-pinned-source.sh" \
      https://github.com/RosettaCommons/RFantibody.git \
      3e5a5bb1cb98befde9fc049ae3ff58cabb491f67 "$source_dir"
    cp -a "$source_dir/." "$MN_RFAB_BUILD_CONTEXT/"
    rm -rf "$source_dir"
  fi
  cp "$ROOT/protein-design/rfantibody/$dockerfile_name" "$MN_RFAB_BUILD_CONTEXT/Dockerfile"
  docker buildx build --load \
    -t "$image_tag" \
    -f "$MN_RFAB_BUILD_CONTEXT/Dockerfile" \
    "$MN_RFAB_BUILD_CONTEXT"
}

case "$APP" in
  bioemu2)
    docker buildx build --load \
      --build-arg BIOEMU2_COMMIT=f99e0d899a2548ee6d9d54b6854b2bc35b6652d5 \
      -f "$ROOT/protein-design/bioemu2/Dockerfile" \
      -t mn-bioemu2:0.1.0rc0-f99e0d8-cu130 \
      "$ROOT"
    ;;
  boltzgen-opt)
    build_boltzgen_opt
    ;;
  protenix-v1-opt)
    build_protenix_v1_opt
    ;;
  colabfold-opt)
    if [[ ! -f "$COLABFOLD_OPT_TOOLS/colabfold/environment/Dockerfile" \
       || ! -d "$COLABFOLD_OPT_TOOLS/colabfold/opt/colabfold_opt" \
       || ! -d "$COLABFOLD_OPT_TOOLS/common/opt_core" ]]; then
      printf 'Required ColabFold optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$COLABFOLD_OPT_TOOLS" >&2
      exit 1
    fi
    colabfold_build_context="$(mktemp -d)"
    trap 'rm -rf "$colabfold_build_context"' EXIT
    mkdir -p "$colabfold_build_context/common" "$colabfold_build_context/_jitcache"
    cp -a "$COLABFOLD_OPT_TOOLS/colabfold" "$colabfold_build_context/colabfold"
    cp -a "$COLABFOLD_OPT_TOOLS/common/opt_core" "$colabfold_build_context/common/opt_core"
    if [[ -d "$COLABFOLD_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$COLABFOLD_OPT_TOOLS/_jitcache/." "$colabfold_build_context/_jitcache/"
    fi
    if ! compgen -G "$colabfold_build_context/_jitcache/colabfold-*-jit.tar" >/dev/null; then
      tar -cf "$colabfold_build_context/_jitcache/colabfold-empty-jit.tar" --files-from /dev/null
    fi
    docker buildx build --load \
      -f "$colabfold_build_context/colabfold/environment/Dockerfile" \
      -t mn-colabfold-opt-kit:1.6.1-jax0.5.3-cu124 \
      "$colabfold_build_context"
    docker buildx build --load \
      --build-arg BASE_IMAGE=mn-colabfold-opt-kit:1.6.1-jax0.5.3-cu124 \
      -f "$ROOT/protein-design/colabfold-opt/Dockerfile" \
      -t mn-colabfold-opt:1.6.1-jax0.5.3-cu124 \
      "$ROOT"
    ;;
  af2ig-opt)
    if [[ ! -f "$AF2IG_OPT_TOOLS/af2ig/environment/Dockerfile" \
       || ! -d "$AF2IG_OPT_TOOLS/af2ig/opt/af2ig_opt" \
       || ! -d "$AF2IG_OPT_TOOLS/common/opt_core" ]]; then
      printf 'Required AF2-IG optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$AF2IG_OPT_TOOLS" >&2
      exit 1
    fi
    af2ig_build_context="$(mktemp -d)"
    trap 'rm -rf "$af2ig_build_context"' EXIT
    mkdir -p "$af2ig_build_context/common" "$af2ig_build_context/_jitcache"
    cp -a "$AF2IG_OPT_TOOLS/af2ig" "$af2ig_build_context/af2ig"
    cp -a "$AF2IG_OPT_TOOLS/common/opt_core" "$af2ig_build_context/common/opt_core"
    if [[ -d "$AF2IG_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$AF2IG_OPT_TOOLS/_jitcache/." "$af2ig_build_context/_jitcache/"
    fi
    if ! compgen -G "$af2ig_build_context/_jitcache/af2ig-*-jit.tar" >/dev/null; then
      tar -cf "$af2ig_build_context/_jitcache/af2ig-empty-jit.tar" --files-from /dev/null
    fi
    docker buildx build --load \
      -f "$af2ig_build_context/af2ig/environment/Dockerfile" \
      -t mn-af2ig-opt:cafa3853-jax0.5.3-cu124 \
      "$af2ig_build_context"
    ;;
  esmfold-v1)
    esmfold_fair_esm="$ROOT/protein-design/esmfold-v1/fair-esm-2b369911.tar.gz"
    if [[ ! -f "$esmfold_fair_esm" ]]; then
      printf 'Pinned fair-esm source archive is missing: %s\n' "$esmfold_fair_esm" >&2
      printf 'Restore the pinned fair-esm source bundle in this recipe, then rerun this command.\n' >&2
      exit 1
    fi
    esmfold_build_context="$(mktemp -d)"
    trap 'rm -rf "$esmfold_build_context"' EXIT
    mkdir -p "$esmfold_build_context/protein-design/esmfold-v1"
    cp "$esmfold_fair_esm" "$esmfold_build_context/protein-design/esmfold-v1/fair-esm-2b369911.tar.gz"
    docker buildx build --load \
      -f "$ROOT/protein-design/esmfold-v1/Dockerfile" \
      -t mn-esmfold:2.0.1-cu128-sm89 \
      "$esmfold_build_context"
    ;;
  esmfold2-opt)
    if [[ ! -f "$ESMFOLD2_OPT_TOOLS/esmfold2/environment/Dockerfile" \
       || ! -d "$ESMFOLD2_OPT_TOOLS/esmfold2/opt/esmfold2_opt" \
       || ! -d "$ESMFOLD2_OPT_TOOLS/common/opt_core" ]]; then
      printf 'Required ESMFold2 optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$ESMFOLD2_OPT_TOOLS" >&2
      exit 1
    fi
    esmfold2_build_context="$(mktemp -d)"
    esmfold2_wrapper_context="$(mktemp -d)"
    trap 'rm -rf "$esmfold2_build_context" "$esmfold2_wrapper_context"' EXIT
    mkdir -p "$esmfold2_build_context/common" "$esmfold2_build_context/_jitcache"
    cp -a "$ESMFOLD2_OPT_TOOLS/esmfold2" "$esmfold2_build_context/esmfold2"
    cp -a "$ESMFOLD2_OPT_TOOLS/common/opt_core" "$esmfold2_build_context/common/opt_core"
    if [[ -d "$ESMFOLD2_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$ESMFOLD2_OPT_TOOLS/_jitcache/." "$esmfold2_build_context/_jitcache/"
    fi
    if ! compgen -G "$esmfold2_build_context/_jitcache/esmfold2-*-jit.tar" >/dev/null; then
      tar -cf "$esmfold2_build_context/_jitcache/esmfold2-empty-jit.tar" --files-from /dev/null
    fi
    mkdir -p "$esmfold2_wrapper_context/protein-design/esmfold2-opt"
    cp "$ROOT/protein-design/esmfold2-opt/Dockerfile" "$esmfold2_wrapper_context/protein-design/esmfold2-opt/Dockerfile"
    cp "$ROOT/protein-design/esmfold2-opt/mn-esmfold2-runner" "$esmfold2_wrapper_context/protein-design/esmfold2-opt/mn-esmfold2-runner"
    python "$ROOT/protein-design/esmfold2-opt/prepare_build_context.py" \
      "$esmfold2_build_context/esmfold2" "$esmfold2_build_context/common/opt_core" \
      "$esmfold2_wrapper_context/protein-design/esmfold2-opt/registry.py"
    test -f "$esmfold2_wrapper_context/protein-design/esmfold2-opt/ef2_msa_v2.py"
    docker buildx build --load \
      --build-arg STACK=img_mn_rtx4090_rtx5090 \
      --build-arg BUILD_JOBS="${MN_ESMFOLD2_BUILD_JOBS:-0}" \
      -f "$esmfold2_build_context/esmfold2/environment/Dockerfile" \
      -t mn-esmfold2-opt-kit:0.1.0-cu130 \
      "$esmfold2_build_context"
    docker buildx build --load \
      --build-arg BASE_IMAGE=mn-esmfold2-opt-kit:0.1.0-cu130 \
      -f "$esmfold2_wrapper_context/protein-design/esmfold2-opt/Dockerfile" \
      -t mn-esmfold2-opt:0.1.0-cu130 \
      "$esmfold2_wrapper_context"
    ;;
  chai1-opt)
    if [[ ! -f "$CHAI1_OPT_TOOLS/chai1/environment/Dockerfile" \
       || ! -d "$CHAI1_OPT_TOOLS/chai1/opt/chai1_opt" \
       || ! -d "$CHAI1_OPT_TOOLS/common/opt_core" ]]; then
      printf 'Required Chai-1 optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$CHAI1_OPT_TOOLS" >&2
      exit 1
    fi
    chai1_build_context="$(mktemp -d)"
    chai1_wrapper_context="$(mktemp -d)"
    trap 'rm -rf "$chai1_build_context" "$chai1_wrapper_context"' EXIT
    mkdir -p "$chai1_build_context/common" "$chai1_build_context/_jitcache"
    cp -a "$CHAI1_OPT_TOOLS/chai1" "$chai1_build_context/chai1"
    cp -a "$CHAI1_OPT_TOOLS/common/opt_core" "$chai1_build_context/common/opt_core"
    python "$ROOT/protein-design/chai1-opt/prepare_build_context.py" \
      "$chai1_build_context/chai1" "$chai1_build_context/common/opt_core"
    if [[ -d "$CHAI1_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$CHAI1_OPT_TOOLS/_jitcache/." "$chai1_build_context/_jitcache/"
    fi
    if ! compgen -G "$chai1_build_context/_jitcache/chai1-*-jit.tar" >/dev/null; then
      tar -cf "$chai1_build_context/_jitcache/chai1-empty-jit.tar" --files-from /dev/null
    fi
    mkdir -p "$chai1_wrapper_context/protein-design/chai1-opt"
    cp "$ROOT/protein-design/chai1-opt/Dockerfile" "$chai1_wrapper_context/protein-design/chai1-opt/Dockerfile"
    cp "$ROOT/protein-design/chai1-opt/mn-chai1-runner" "$chai1_wrapper_context/protein-design/chai1-opt/mn-chai1-runner"
    cp "$ROOT/protein-design/chai1-opt/mn_chai1_inputs.py" "$chai1_wrapper_context/protein-design/chai1-opt/mn_chai1_inputs.py"
    docker buildx build --load \
      -f "$chai1_build_context/chai1/environment/Dockerfile" \
      -t mn-chai1-opt-kit:0.6.1-cu130 \
      "$chai1_build_context"
    docker buildx build --load \
      --build-arg BASE_IMAGE=mn-chai1-opt-kit:0.6.1-cu130 \
      -f "$chai1_wrapper_context/protein-design/chai1-opt/Dockerfile" \
      -t mn-chai1-opt:0.6.1-cu130 \
      "$chai1_wrapper_context"
    ;;
  atlasfold-opt)
    if [[ ! -f "$ATLASFOLD_OPT_TOOLS/atlasfold/run.sh" \
       || ! -d "$ATLASFOLD_OPT_TOOLS/atlasfold/opt/atlasfold_opt" \
       || ! -d "$ATLASFOLD_OPT_TOOLS/common/opt_core/opt_core" ]]; then
      printf 'Required AtlasFold optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$ATLASFOLD_OPT_TOOLS" >&2
      exit 1
    fi
    atlasfold_build_context="$(mktemp -d)"
    cleanup_atlasfold_context() {
      python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$atlasfold_build_context"
    }
    trap cleanup_atlasfold_context EXIT
    mkdir -p "$atlasfold_build_context/atlasfold" "$atlasfold_build_context/common" "$atlasfold_build_context/protein-design/atlasfold-opt"
    cp -a "$ATLASFOLD_OPT_TOOLS/atlasfold/." "$atlasfold_build_context/atlasfold/"
    cp -a "$ATLASFOLD_OPT_TOOLS/common/opt_core" "$atlasfold_build_context/common/opt_core"
    cp "$ROOT/protein-design/atlasfold-opt/Dockerfile" "$atlasfold_build_context/protein-design/atlasfold-opt/Dockerfile"
    cp "$ROOT/protein-design/atlasfold-opt/mn-atlasfold-runner" "$atlasfold_build_context/protein-design/atlasfold-opt/mn-atlasfold-runner"
    python3 "$ROOT/protein-design/atlasfold-opt/prepare_build_context.py" \
      "$atlasfold_build_context/atlasfold" "$atlasfold_build_context/common/opt_core"
    atlasfold_kit_rev="$(git -C "$ATLASFOLD_OPT_TOOLS" rev-parse HEAD 2>/dev/null || printf unknown)"
    docker buildx build --load \
      --build-arg ATLASFOLD_KIT_REV="$atlasfold_kit_rev" \
      -f "$atlasfold_build_context/protein-design/atlasfold-opt/Dockerfile" \
      -t mn-atlasfold-opt:1.0.0-cu128 \
      "$atlasfold_build_context"
    ;;
  openfold3-ob0)
    if [[ ! -f "$OPENFOLD3_OB0_OPT_TOOLS/openfold3_ob0/environment/Dockerfile" \
       || ! -f "$OPENFOLD3_OB0_OPT_TOOLS/openfold3_ob0/run.sh" \
       || ! -d "$OPENFOLD3_OB0_OPT_TOOLS/common/opt_core/opt_core" ]]; then
      printf 'Required OpenFold3 OpenBind-0 optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$OPENFOLD3_OB0_OPT_TOOLS" >&2
      exit 1
    fi
    openfold3_build_context="$(mktemp -d)"
    trap 'rm -rf "$openfold3_build_context"' EXIT
    mkdir -p "$openfold3_build_context/common" "$openfold3_build_context/_jitcache" \
      "$openfold3_build_context/protein-design/openfold3/optimized"
    cp -a "$OPENFOLD3_OB0_OPT_TOOLS/openfold3_ob0" "$openfold3_build_context/openfold3_ob0"
    cp -a "$OPENFOLD3_OB0_OPT_TOOLS/common/opt_core" "$openfold3_build_context/common/opt_core"
    cp "$ROOT/protein-design/openfold3/optimized/Dockerfile" \
      "$openfold3_build_context/protein-design/openfold3/optimized/Dockerfile"
    python3 "$ROOT/protein-design/openfold3/optimized/prepare_build_context.py" \
      "$openfold3_build_context/openfold3_ob0" "$openfold3_build_context/common/opt_core"
    if [[ -d "$OPENFOLD3_OB0_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$OPENFOLD3_OB0_OPT_TOOLS/_jitcache/." "$openfold3_build_context/_jitcache/"
    fi
    if ! compgen -G "$openfold3_build_context/_jitcache/openfold3_ob0-*-jit.tar" >/dev/null; then
      tar -cf "$openfold3_build_context/_jitcache/openfold3_ob0-empty-jit.tar" --files-from /dev/null
    fi
    openfold3_kit_rev="$(git -C "$OPENFOLD3_OB0_OPT_TOOLS" rev-parse HEAD 2>/dev/null || printf unknown)"
    docker buildx build --load \
      --build-arg BUILD_JOBS="${MN_OPENFOLD3_BUILD_JOBS:-0}" \
      --build-arg OPENFOLD3_OB0_KIT_REV="$openfold3_kit_rev" \
      -f "$openfold3_build_context/protein-design/openfold3/optimized/Dockerfile" \
      -t mn-openfold3-native-opt:0.5.0-ob0-cu128 \
      "$openfold3_build_context"
    ;;
  opendde-opt)
    if [[ ! -f "$OPENDDE_OPT_TOOLS/opendde/environment/Dockerfile" \
       || ! -f "$OPENDDE_OPT_TOOLS/opendde/run.sh" \
       || ! -d "$OPENDDE_OPT_TOOLS/opendde/opt/opendde_opt" \
       || ! -d "$OPENDDE_OPT_TOOLS/common/opt_core/opt_core" ]]; then
      printf 'Required OpenDDE optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$OPENDDE_OPT_TOOLS" >&2
      exit 1
    fi
    opendde_build_context="$(mktemp -d)"
    trap 'rm -rf "$opendde_build_context"' EXIT
    mkdir -p "$opendde_build_context/common" "$opendde_build_context/_jitcache"
    cp -a "$OPENDDE_OPT_TOOLS/opendde" "$opendde_build_context/opendde"
    cp -a "$OPENDDE_OPT_TOOLS/common/opt_core" "$opendde_build_context/common/opt_core"
    python3 "$ROOT/protein-design/opendde/optimized/prepare_build_context.py" \
      "$opendde_build_context/opendde" "$opendde_build_context/common/opt_core"
    if [[ -d "$OPENDDE_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$OPENDDE_OPT_TOOLS/_jitcache/." "$opendde_build_context/_jitcache/"
    fi
    if ! compgen -G "$opendde_build_context/_jitcache/opendde-*-jit.tar" >/dev/null; then
      tar -cf "$opendde_build_context/_jitcache/opendde-empty-jit.tar" --files-from /dev/null
    fi
    docker buildx build --load \
      -f "$opendde_build_context/opendde/environment/Dockerfile" \
      -t mn-opendde-opt:1.1.1-cu128 \
      "$opendde_build_context"
    ;;
  af3-torch-opt)
    if [[ ! -f "$AF3_TORCH_OPT_TOOLS/af3_torch/environment/Dockerfile" \
       || ! -d "$AF3_TORCH_OPT_TOOLS/af3_torch/opt/af3_torch_opt" \
       || ! -d "$AF3_TORCH_OPT_TOOLS/common/opt_core/opt_core" ]]; then
      printf 'Required AF3-Torch optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$AF3_TORCH_OPT_TOOLS" >&2
      exit 1
    fi
    af3_torch_build_context="$(mktemp -d)"
    trap 'rm -rf "$af3_torch_build_context"' EXIT
    mkdir -p "$af3_torch_build_context/common" "$af3_torch_build_context/_jitcache"
    cp -a "$AF3_TORCH_OPT_TOOLS/af3_torch" "$af3_torch_build_context/af3_torch"
    cp -a "$AF3_TORCH_OPT_TOOLS/common/opt_core" "$af3_torch_build_context/common/opt_core"
    if [[ -d "$AF3_TORCH_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$AF3_TORCH_OPT_TOOLS/_jitcache/." "$af3_torch_build_context/_jitcache/"
    fi
    if ! compgen -G "$af3_torch_build_context/_jitcache/af3_torch-*-jit.tar" >/dev/null; then
      tar -cf "$af3_torch_build_context/_jitcache/af3_torch-empty-jit.tar" --files-from /dev/null
    fi
    docker buildx build --load \
      --build-arg BUILD_JOBS="${MN_AF3_TORCH_BUILD_JOBS:-0}" \
      -f "$af3_torch_build_context/af3_torch/environment/Dockerfile" \
      -t mn-af3-torch-opt-kit:22bdeed-cu130 \
      "$af3_torch_build_context"
    docker buildx build --load \
      --build-arg BASE_IMAGE=mn-af3-torch-opt-kit:22bdeed-cu130 \
      -f "$ROOT/protein-design/af3-torch-opt/Dockerfile" \
      -t mn-af3-torch-opt:22bdeed-cu130 \
      "$ROOT"
    ;;
  boltz2-opt)
    if [[ ! -f "$BOLTZ2_OPT_TOOLS/boltz2/environment/Dockerfile" || ! -d "$BOLTZ2_OPT_TOOLS/common/opt_core" ]]; then
      printf 'Required Boltz2 optimization source is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$BOLTZ2_OPT_TOOLS" >&2
      exit 1
    fi
    docker buildx build --load \
      --build-context "boltzkit=$BOLTZ2_OPT_TOOLS" \
      -f "$ROOT/shared/boltz2/optimized/Dockerfile" \
      -t mn-boltz2-opt:2.2.1-cu130 \
      "$ROOT"
    ;;
  rf3-opt)
    if [[ ! -f "$RF3_OPT_TOOLS/rosettafold3/environment/Dockerfile" \
       || ! -d "$RF3_OPT_TOOLS/rosettafold3/opt/rosettafold3_opt" \
       || ! -d "$RF3_OPT_TOOLS/common/opt_core" ]]; then
      printf 'Required RF3 optimization kit is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$RF3_OPT_TOOLS" >&2
      exit 1
    fi
    rf3_build_context="$(mktemp -d)"
    trap 'rm -rf "$rf3_build_context"' EXIT
    mkdir -p "$rf3_build_context/common" "$rf3_build_context/_jitcache"
    cp -a "$RF3_OPT_TOOLS/rosettafold3" "$rf3_build_context/rosettafold3"
    cp -a "$RF3_OPT_TOOLS/common/opt_core" "$rf3_build_context/common/opt_core"
    if [[ -d "$RF3_OPT_TOOLS/_jitcache" ]]; then
      cp -a "$RF3_OPT_TOOLS/_jitcache/." "$rf3_build_context/_jitcache/"
    fi
    if ! compgen -G "$rf3_build_context/_jitcache/rosettafold3-*-jit.tar" >/dev/null; then
      tar -cf "$rf3_build_context/_jitcache/rosettafold3-empty-jit.tar" --files-from /dev/null
    fi
    docker buildx build --load \
      -f "$rf3_build_context/rosettafold3/environment/Dockerfile" \
      -t mn-rf3-opt:0.2.1-dev13-cu130 \
      "$rf3_build_context"
    ;;
  boltz2-opt-batch)
    build_batch_api_overlay mn-boltz2-opt:2.2.1-cu130 boltz2 mn-boltz2-opt:2.2.1-cu130-batch1
    ;;
  boltz2-opt-batch2)
    build_batch_api_overlay mn-boltz2-opt:2.2.1-cu130 boltz2 mn-boltz2-opt:2.2.1-cu130-batch2
    ;;
  rf3-opt-batch)
    build_batch_api_overlay mn-rf3-opt:0.2.1-dev13-cu130 rf3_foundry mn-rf3-opt:0.2.1-dev13-cu130-batch1
    ;;
  af3-torch-opt-batch)
    build_batch_api_overlay mn-af3-torch-opt:22bdeed-cu130 af3_torch mn-af3-torch-opt:22bdeed-cu130-batch1
    ;;
  af3-torch-opt-batch2)
    build_batch_api_overlay mn-af3-torch-opt:22bdeed-cu130 af3_torch mn-af3-torch-opt:22bdeed-cu130-batch2
    ;;
  af3-torch-opt-batch3)
    build_batch_api_overlay mn-af3-torch-opt:22bdeed-cu130 af3_torch mn-af3-torch-opt:22bdeed-cu130-batch3
    ;;
  alphafold3-batch)
    build_batch_api_overlay mn-cofolding-af3-memory:3.1.14-cu13 alphafold3 mn-cofolding-af3-memory:3.1.14-cu13-af3-batch1
    ;;
  openfold3-batch)
    build_batch_api_overlay mn-cofolding-af3-memory:3.1.14-cu13 openfold3 mn-cofolding-af3-memory:3.1.14-cu13-openfold3-batch1
    ;;
  esmfold2-opt-batch)
    build_batch_api_overlay mn-esmfold2-opt:0.1.0-cu130 esmfold2 mn-esmfold2-opt:0.1.0-cu130-batch1
    ;;
  esmfold2-fast-opt-batch)
    build_batch_api_overlay mn-esmfold2-opt:0.1.0-cu130 esmfold2_fast mn-esmfold2-opt:0.1.0-cu130-fast-batch1
    ;;
  chai1-opt-batch)
    build_batch_api_overlay mn-chai1-opt:0.6.1-cu130 chai1 mn-chai1-opt:0.6.1-cu130-batch1
    ;;
  atlasfold-opt-batch)
    build_batch_api_overlay mn-atlasfold-opt:1.0.0-cu128 atlasfold mn-atlasfold-opt:1.0.0-cu128-batch1
    ;;
  atlasfold-opt-batch2)
    build_batch_api_overlay mn-atlasfold-opt:1.0.0-cu128 atlasfold mn-atlasfold-opt:1.0.0-cu128-batch2
    ;;
  colabfold-batch)
    build_batch_api_overlay mn-colabfold:1.6.1-cu12 colabfold mn-colabfold:1.6.1-cu12-batch1
    ;;
  colabfold-batch2)
    build_batch_api_overlay mn-colabfold:1.6.1-cu12 colabfold mn-colabfold:1.6.1-cu12-batch2
    ;;
  colabfold-batch3)
    build_batch_api_overlay mn-colabfold:1.6.1-cu12 colabfold mn-colabfold:1.6.1-cu12-batch3
    ;;
  colabfold-opt-batch)
    build_batch_api_overlay mn-colabfold-opt:1.6.1-jax0.5.3-cu124 colabfold_opt mn-colabfold-opt:1.6.1-jax0.5.3-cu124-batch1
    ;;
  colabfold-opt-batch2)
    build_batch_api_overlay mn-colabfold-opt:1.6.1-jax0.5.3-cu124 colabfold_opt mn-colabfold-opt:1.6.1-jax0.5.3-cu124-batch2
    ;;
  colabfold-opt-batch3)
    build_batch_api_overlay mn-colabfold-opt:1.6.1-jax0.5.3-cu124 colabfold_opt mn-colabfold-opt:1.6.1-jax0.5.3-cu124-batch3
    ;;
  openfold3-ob0-batch)
    build_batch_api_overlay mn-openfold3-native-opt:0.5.0-ob0-cu128 openfold3_ob0 mn-openfold3-native-opt:0.5.0-ob0-cu128-batch1
    ;;
  openfold3-ob0-batch2)
    build_batch_api_overlay mn-openfold3-native-opt:0.5.0-ob0-cu128 openfold3_ob0 mn-openfold3-native-opt:0.5.0-ob0-cu128-batch2
    ;;
  opendde-opt-batch)
    build_batch_api_overlay mn-opendde-opt:1.1.1-cu128 opendde_native mn-opendde-opt:1.1.1-cu128-batch1
    ;;
  opendde-opt-batch2)
    build_batch_api_overlay mn-opendde-opt:1.1.1-cu128 opendde_native mn-opendde-opt:1.1.1-cu128-batch2
    ;;
  opendde-opt-batch3)
    build_batch_api_overlay mn-opendde-opt:1.1.1-cu128 opendde_native mn-opendde-opt:1.1.1-cu128-batch3
    ;;
  protenix-v1-opt-batch)
    build_batch_api_overlay mn-protenix-v1-opt:1.1.0-cu130 protenix_v1 mn-protenix-v1-opt:1.1.0-cu130-batch1
    ;;
  protenix-v1-opt-batch2)
    build_batch_api_overlay mn-protenix-v1-opt:1.1.0-cu130 protenix_v1 mn-protenix-v1-opt:1.1.0-cu130-batch2
    ;;
  protenix-v2-opt-batch)
    build_batch_api_overlay mn-protenix-opt:2.0.0-cu128 protenix_v2_opt mn-protenix-opt:2.0.0-cu128-batch1
    ;;
  protenix-v2-opt-sm120)
    bash "$ROOT/protein-design/protenix-opt/blackwell/build.sh"
    ;;
  protenix-v2-opt-batch2)
    build_batch_api_overlay mn-protenix-opt:2.0.0-cu128 protenix_v2_opt mn-protenix-opt:2.0.0-cu128-batch2
    ;;
  boltzgen-opt-batch)
    build_batch_api_overlay mn-boltzgen-opt:a07fea1-cu130 boltzgen_fold_opt_msa mn-boltzgen-opt:a07fea1-cu130-batch1
    ;;
  boltzgen-opt-batch2)
    build_batch_api_overlay mn-boltzgen-opt:a07fea1-cu130 boltzgen_fold_opt_msa mn-boltzgen-opt:a07fea1-cu130-batch2
    ;;
  boltzgen-opt-batch3)
    build_batch_api_overlay mn-boltzgen-opt:a07fea1-cu130 boltzgen_fold_opt_msa mn-boltzgen-opt:a07fea1-cu130-batch3
    ;;
  alphafast-batch)
    build_batch_api_overlay alphafast:cuda126-backup alphafast mn-alphafast:cuda126-backup-batch1
    ;;
  boltz2-stock-batch)
    build_batch_api_overlay mn-boltz2:cu128 boltz2_legacy mn-boltz2:cu128-batch1
    ;;
  biohub-esm-batch)
    build_batch_api_overlay mn-biohub-esm:3.4.1-cu128 esmfold2_biohub mn-biohub-esm:3.4.1-cu128-batch1
    ;;
  openfold3-p2-batch)
    build_batch_api_overlay mn-openfold3:cu13 openfold3_p2 mn-openfold3:cu13-batch1
    ;;
  openfold3-p2-batch2)
    build_batch_api_overlay mn-openfold3:cu13 openfold3_p2 mn-openfold3:cu13-batch2
    ;;
  pxdesign-v05-batch)
    build_batch_api_overlay mn-pxdesign:cu128 pxdesign_v05 mn-pxdesign:cu128-batch1
    ;;
  pxdesign-v05-batch2)
    build_batch_api_overlay mn-pxdesign:cu128 pxdesign_v05 mn-pxdesign:cu128-batch2
    ;;
  pxdesign-v05-batch3)
    build_batch_api_overlay mn-pxdesign:cu128 pxdesign_v05 mn-pxdesign:cu128-batch3
    ;;
  pxdesign-v05-batch4)
    build_batch_api_overlay mn-pxdesign:cu128 pxdesign_v05 mn-pxdesign:cu128-batch4
    ;;
  pxdesign-v05-batch5)
    build_batch_api_overlay mn-pxdesign:cu128 pxdesign_v05 mn-pxdesign:cu128-batch5
    ;;
  pxdesign-v05-batch6)
    build_batch_api_overlay mn-pxdesign:cu128 pxdesign_v05 mn-pxdesign:cu128-batch6
    ;;
  protenix-v2-stock-batch)
    build_batch_api_overlay mn-protenix:cu128 protenix_v2_stock mn-protenix:cu128-batch1
    ;;
  protenix-v2-stock-batch2)
    build_batch_api_overlay mn-protenix:cu128 protenix_v2_stock mn-protenix:cu128-batch2
    ;;
  foundry-rf3-batch)
    build_batch_api_overlay mn-foundry:cu128 rf3_foundry_legacy mn-foundry:cu128-batch1
    ;;
  foundry-rf3-batch2)
    build_batch_api_overlay mn-foundry:cu128 rf3_foundry_legacy mn-foundry:cu128-batch2
    ;;
  batch-api)
    bash "$ROOT/build.sh" boltz2-opt-batch
    bash "$ROOT/build.sh" rf3-opt-batch
    bash "$ROOT/build.sh" af3-torch-opt-batch2
    bash "$ROOT/build.sh" alphafold3-batch
    bash "$ROOT/build.sh" openfold3-batch
    bash "$ROOT/build.sh" esmfold2-opt-batch
    bash "$ROOT/build.sh" esmfold2-fast-opt-batch
    bash "$ROOT/build.sh" chai1-opt-batch
    bash "$ROOT/build.sh" atlasfold-opt-batch2
    bash "$ROOT/build.sh" colabfold-batch
    bash "$ROOT/build.sh" openfold3-ob0-batch2
    bash "$ROOT/build.sh" opendde-opt-batch2
    bash "$ROOT/build.sh" protenix-v1-opt-batch2
    bash "$ROOT/build.sh" protenix-v2-opt-batch2
    bash "$ROOT/build.sh" protenix-v2-opt-sm120
    bash "$ROOT/build.sh" boltzgen-opt-batch3
    bash "$ROOT/build.sh" alphafast-batch
    bash "$ROOT/build.sh" boltz2-stock-batch
    bash "$ROOT/build.sh" biohub-esm-batch
    bash "$ROOT/build.sh" openfold3-p2-batch2
    bash "$ROOT/build.sh" pxdesign-v05-batch6
    bash "$ROOT/build.sh" protenix-v2-stock-batch2
    bash "$ROOT/build.sh" foundry-rf3-batch2
    ;;
  mn-protein-design)
    build_biohub_esm
    compose "$LIGAND_REPO" alphafast boltz2 pesto
    require_images mn-python-structure:latest mn-bindcraft:latest mn-ipsae:latest
    compose "$PROTEIN_REPO" scannet surf2spot masif-seed genie3 pxdesign protenix protpardelle-1c colabfold openfold3 boltzgen proteina-complexa
    if [[ ! -d "$PROTENIX_OPT_TOOLS/protenix_v2/opt/protenix_opt" || ! -d "$PROTENIX_OPT_TOOLS/common/opt_core/opt_core" ]]; then
      printf 'Required Protenix optimization source is missing at %s. Check out uplifting-biomolecular-modeling, then rerun this command.\n' "$PROTENIX_OPT_TOOLS" >&2
      exit 1
    fi
    docker buildx build --load \
      --build-context "protenixkit=$PROTENIX_OPT_TOOLS" \
      -f "$ROOT/protein-design/protenix-opt/Dockerfile" \
      -t mn-protenix-opt:2.0.0-cu128 \
      "$ROOT"
    compose "$PROTEIN_REPO" scoring-python scoring-pyrosetta scoring-ipsae
    ;;
  mn-ligand)
    require_images mn-biohub-esm:cu128
    compose "$LIGAND_REPO" alphafast boltz2 pesto
    compose "$LIGAND_REPO" conditar
    for image in mn-boltzina-base:latest mn-plip-base:latest; do
      if ! docker image inspect "$image" >/dev/null 2>&1; then
        printf 'Required upstream base image is missing: %s\n' "$image" >&2
        printf 'Build or load that base image, then rerun this command.\n' >&2
        exit 1
      fi
    done
    compose "$LIGAND_REPO" structure docking docking-suite openvs fpocket p2rank md gromacs admet nesso omtra pocketxmol flowr-root paopt drugrpg pfm pocketflow lddm pgmg posebusters plip pandamap qc abfe rbfe boltzina
    ;;
  mn-cofolding)
    docker build \
      -f "$ROOT/protein-design/cofolding-af3/Dockerfile" \
      -t mn-cofolding-af3:3.1.14-cu13 \
      "$ROOT"
    if [[ -d "$AF3_OPT_TOOLS/af3_jax/opt/forward/flashpairformer/af3_flashpairformer" \
       && -d "$AF3_OPT_TOOLS/common/opt_core/opt_core/kernels/fpf_pallas" ]]; then
      docker buildx build --load \
        --build-context "af3jax=$AF3_OPT_TOOLS/af3_jax" \
        --build-context "optcore=$AF3_OPT_TOOLS/common" \
        -f "$ROOT/protein-design/cofolding-af3/optimized/Dockerfile" \
        -t mn-cofolding-af3-memory:3.1.14-cu13 \
        "$ROOT"
    else
      printf 'Required AF3 optimization sources are missing at %s. Check out uplifting-biomolecular-modeling, or use the stock image explicitly with MN_COFOLDING_AF3_IMAGE=mn-cofolding-af3:3.1.14-cu13.\n' "$AF3_OPT_TOOLS" >&2
      exit 1
    fi
    build_protenix_v1_opt
    if [[ "${MN_REBUILD_COLABFOLD:-0}" == "1" ]] || ! docker image inspect mn-colabfold:1.6.1-cu12 >/dev/null 2>&1; then
      docker build \
        -f "$ROOT/protein-design/colabfold/Dockerfile" \
        -t mn-colabfold:1.6.1-cu12 \
        "$ROOT"
    else
      printf 'Using existing mn-colabfold:1.6.1-cu12 image. Set MN_REBUILD_COLABFOLD=1 to rebuild it.\n'
    fi
    if [[ "${MN_BUILD_LOCAL_MSA:-0}" == "1" ]]; then
      if [[ ! -d "$LIGAND_REPO" ]]; then
        printf 'Local MMseqs-GPU image source is missing: %s\n' "$LIGAND_REPO" >&2
        exit 1
      fi
      compose "$LIGAND_REPO" alphafast
    fi
    ;;
  rfantibody)
    build_rfantibody mn-rfantibody:3e5a5bb-cu128 Dockerfile
    ;;
  genie3)
    compose "$PROTEIN_REPO" genie3
    ;;
  pxdesign)
    compose "$PROTEIN_REPO" pxdesign
    ;;
  *)
    usage
    exit 2
    ;;
esac
