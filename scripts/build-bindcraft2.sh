#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_URL="https://github.com/PacesaLab/BindCraft2.git"
SOURCE_COMMIT="4a56313e96afc13c443d88427281cf2169a1c9ca"
IMAGE_TAG="${1:-mn-bindcraft2:4a56313-cu13}"
shift || true
PLATFORM="${BINDCRAFT2_PLATFORM:-linux/amd64}"
BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/mn-bindcraft2.XXXXXX")"
trap 'rm -rf "$BUILD_CONTEXT"' EXIT

git clone --no-checkout --filter=blob:none "$SOURCE_URL" "$BUILD_CONTEXT/source"
git -C "$BUILD_CONTEXT/source" checkout --detach "$SOURCE_COMMIT"

docker build \
  --platform "$PLATFORM" \
  --tag "$IMAGE_TAG" \
  -f "$ROOT/protein-design/bindcraft2/Dockerfile" \
  "$@" \
  "$BUILD_CONTEXT/source"

printf 'Built %s from BindCraft2 commit %s (%s).\n' "$IMAGE_TAG" "$SOURCE_COMMIT" "$PLATFORM"
