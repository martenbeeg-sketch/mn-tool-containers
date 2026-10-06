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

usage() {
  printf 'Usage: %s {mn-protein-design|mn-ligand|mn-cofolding|boltz2-opt|rf3-opt|colabfold-opt|af2ig-opt|esmfold2-opt|chai1-opt}\n' "$0" >&2
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

build_biohub_esm() {
  if [[ "${MN_REBUILD_BIOHUB_ESM:-0}" != "1" ]] && docker image inspect mn-biohub-esm:3.4.1-cu128 >/dev/null 2>&1; then
    printf 'Using existing mn-biohub-esm:3.4.1-cu128 image. Transfer this image separately when installing on another computer.\n'
    printf 'Set MN_REBUILD_BIOHUB_ESM=1 to rebuild it from pinned public PyPI packages and Biohub ESM source.\n'
    return
  fi
  compose "$PROTEIN_REPO" biohub-esm
}

case "$APP" in
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
  *)
    usage
    exit 2
    ;;
esac
