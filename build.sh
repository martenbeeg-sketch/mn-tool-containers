#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS="$(cd "$ROOT/.." && pwd)"
PROTEIN_REPO="$PROJECTS/mn-protein-design"
LIGAND_REPO="$PROJECTS/ovo-ligand"

usage() {
  printf 'Usage: %s {mn-protein-design|mn-ligand}\n' "$0" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

APP="$1"
for repo in "$PROTEIN_REPO" "$LIGAND_REPO"; do
  if [[ ! -d "$repo/.git" ]]; then
    printf 'Required sibling checkout is missing: %s\n' "$repo" >&2
    exit 1
  fi
done

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
  mn-protein-design)
    build_biohub_esm
    compose "$LIGAND_REPO" alphafast boltz2 pesto
    require_images mn-python-structure:latest mn-bindcraft:latest mn-ipsae:latest
    compose "$PROTEIN_REPO" scannet surf2spot masif-seed genie3 pxdesign protenix protpardelle-1c colabfold openfold3 boltzgen proteina-complexa
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
  *)
    usage
    exit 2
    ;;
esac
