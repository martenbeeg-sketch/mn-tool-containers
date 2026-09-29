#!/usr/bin/env bash
set -euo pipefail

# Docker tags point to the same image ID. This adds canonical mn-* tags while
# retaining every legacy tag already installed on the machine.
declare -a PAIRS=(
  'mnprot-scannet:latest mn-scannet:cpu'
  'mnprot-surf2spot-cu128:latest mn-surf2spot:cu128'
  'mnprot-pesto-cu128:latest mn-pesto:cu128'
  'masif_seed:latest mn-masif-seed:cpu'
  'mnprot-genie3-cu128:latest mn-genie3:cu128'
  'mnprot-pxdesign-cu128:latest mn-pxdesign:cu128'
  'mnprot-protenix-cu128:latest mn-protenix:cu128'
  'mnprot-protpardelle-1c-cu128:latest mn-protpardelle-1c:cu128'
  'mnprot-biohub-esm-cu128:latest mn-biohub-esm:cu128'
  'mnprot-colabfold-cuda12:1.6.1 mn-colabfold:1.6.1-cu12'
  'mnprot-openfold3-cu13:latest mn-openfold3:cu13'
  'mnprot-bindcraft2:4a56313 mn-bindcraft2:4a56313-cu13'
  'alphafast:latest mn-alphafast:cu128'
  'ovoex-boltz2:latest mn-boltz2:cu128'
  'ovo-ligandmpnn:latest mn-ligandmpnn:latest'
  'ovoex-foundry-cu128:latest mn-foundry:cu128'
  'ovo-bindcraft:latest mn-bindcraft:latest'
  'ovo-rfdiffusion:latest mn-rfdiffusion:latest'
  'boltzgen:latest mn-boltzgen:latest'
  'ovoex-proteina-complexa:latest mn-proteina-complexa:latest'
  'ovoex-ipsae:latest mn-ipsae:latest'
  'ovo-colabdesign:latest mn-colabdesign:latest'
  'ovo-python-structure:latest mn-python-structure:latest'
  'ovolig-structure:latest mn-structure:latest'
  'ovolig-docking:latest mn-docking:latest'
  'avgu-docking-suite-cuda:latest mn-docking-suite:cu128'
  'openvs:local mn-openvs:local'
  'ovolig-fpocket:latest mn-fpocket:4bb0d84'
  'ovolig-p2rank:latest mn-p2rank:d8c8e0d'
  'ovolig-md-cu128:latest mn-md:cu128'
  'ovolig-gromacs-cu128:latest mn-gromacs:2026.3-cu128'
  'ovolig-admet:latest mn-admet:latest'
  'ovolig-boltzina-cu128:latest mn-boltzina:cu128'
  'ovolig-nesso-cu128:latest mn-nesso:1.0.0-cu128'
  'ovolig-omtra-cu128:latest mn-omtra:cu128-745127d'
  'ovolig-pocketxmol-cu128:latest mn-pocketxmol:cu128-65488cf'
  'ovolig-flowr-root-cu128:latest mn-flowr-root:cu128-b2263e2'
  'ovolig-conditar-cu128:latest mn-conditar:cu128-4294d286'
  'ovolig-paopt-cu128:latest mn-paopt:cu128-4294d286'
  'ovolig-drugrpg-cu128:latest mn-drugrpg:cu128-6fa0e41'
  'ovolig-pfm-cu128:latest mn-pfm:cu128-33be6c1'
  'ovolig-pocketflow-cu128:latest mn-pocketflow:cu128-a31a5a0'
  'ovolig-lddm-cu128:latest mn-lddm:cu128-f254fb4'
  'ovolig-pgmg-cu128:latest mn-pgmg:cu128-85fb712'
  'ovolig-posebusters:latest mn-posebusters:1a5f26a'
  'ovolig-plip:latest mn-plip:latest'
  'ovolig-pandamap:latest mn-pandamap:0007347'
  'ovolig-qc:latest mn-qc:latest'
  'ovolig-abfe-cu128:latest mn-abfe:cu128'
  'ovolig-rbfe-cu128:latest mn-rbfe:cu128'
  'boltzina:latest mn-boltzina-base:latest'
  'local-plip:latest mn-plip-base:latest'
)

for pair in "${PAIRS[@]}"; do
  read -r old_image new_image <<< "$pair"
  if docker image inspect "$new_image" >/dev/null 2>&1; then
    printf 'Already tagged: %s\n' "$new_image"
  elif docker image inspect "$old_image" >/dev/null 2>&1; then
    docker image tag "$old_image" "$new_image"
    printf 'Tagged %s as %s\n' "$old_image" "$new_image"
  else
    printf 'Source image unavailable; skipped %s\n' "$old_image"
  fi
done
