#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
  tags=(cu128 cu128-sm120fix)
else
  tags=("$@")
fi

for tag in "${tags[@]}"; do
  base_tag="mn-docking-suite:${tag}"
  backup_tag="mn-docking-suite:${tag}-before-molscrub-versions"
  candidate_tag="mn-docking-suite:${tag}-molscrub-versions-candidate"
  base_id="$(docker image inspect --format '{{.Id}}' "${base_tag}")"
  if ! docker image inspect "${backup_tag}" >/dev/null 2>&1; then
    docker tag "${base_id}" "${backup_tag}"
  fi

  docker build --pull=false \
    --build-arg "BASE_IMAGE=${backup_tag}" \
    --file "${ROOT}/Dockerfile.molscrub-overlay" \
    --tag "${candidate_tag}" \
    "${ROOT}"

  base_udp_sha="$(docker run --rm --entrypoint bash "${base_id}" -lc 'sha256sum /opt/unidock/UniDock-Pro/build/udp | cut -d" " -f1')"
  candidate_udp_sha="$(docker run --rm --entrypoint bash "${candidate_tag}" -lc 'sha256sum /opt/unidock/UniDock-Pro/build/udp | cut -d" " -f1')"
  if [[ "${candidate_udp_sha}" != "${base_udp_sha}" ]]; then
    printf 'UniDock-Pro binary changed while adding MolScrub to %s\n' "${base_tag}" >&2
    exit 1
  fi

  docker run --rm --entrypoint bash "${candidate_tag}" -lc '
    set -euo pipefail
    for version in 0.1.0 0.3.0; do
      venv="/opt/molscrub/venvs/${version}"
      test -x "${venv}/bin/scrub.py"
      test "$("${venv}/bin/python" -c "from importlib.metadata import version; print(version(\"molscrub\"))")" = "${version}"
      "${venv}/bin/scrub.py" --help >/dev/null
    done
  '

  docker tag "${candidate_tag}" "${base_tag}"
  docker image rm "${candidate_tag}" >/dev/null
  printf 'Updated %s with MolScrub 0.1.0 and 0.3.0; previous image kept as %s\n' \
    "${base_tag}" "${backup_tag}"
done
