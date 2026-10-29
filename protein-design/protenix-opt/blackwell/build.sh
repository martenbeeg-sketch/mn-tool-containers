#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
base_image="${PROTENIX_SM120_BASE_IMAGE:-mn-protenix-opt:2.0.0-cu128-batch2}"
output_image="${PROTENIX_SM120_IMAGE:-mn-protenix-opt:2.0.0-cu128-sm89-sm120-batch3}"
if [[ "$base_image" == "$output_image" ]]; then
  echo 'Choose a new output tag; preserve the base image.' >&2
  exit 1
fi
base_id="$(docker image inspect --format '{{.Id}}' "$base_image")"
docker build --network=none --progress=plain \
  --build-arg "BASE_IMAGE=$base_image" --build-arg "BASE_IMAGE_ID=$base_id" \
  -f protein-design/protenix-opt/Dockerfile.sm120 -t "$output_image" .
docker run --rm --network none --entrypoint cat "$output_image" /opt/mn-protenix-fixes/build-manifest.json
