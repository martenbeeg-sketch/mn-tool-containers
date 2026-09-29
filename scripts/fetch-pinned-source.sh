#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: %s <git-url> <commit> <destination> [patch-name]\n' "$0" >&2
  exit 2
fi

SOURCE_URL="$1"
SOURCE_COMMIT="$2"
SOURCE_DIR="$3"
PATCH_NAME="${4:-}"
PATCH_DIR="${MN_TOOL_PATCH_DIR:-/opt/mn-tool-patches}"

mkdir -p "$(dirname "$SOURCE_DIR")"
git init -q "$SOURCE_DIR"
git -C "$SOURCE_DIR" remote add origin "$SOURCE_URL"
git -C "$SOURCE_DIR" fetch --quiet --depth 1 origin "$SOURCE_COMMIT"
git -C "$SOURCE_DIR" checkout --quiet --detach FETCH_HEAD

if [[ -n "$PATCH_NAME" ]]; then
  PATCH_PATH="$PATCH_DIR/$PATCH_NAME"
  if [[ ! -f "$PATCH_PATH" ]]; then
    printf 'Pinned source patch is missing: %s\n' "$PATCH_PATH" >&2
    exit 1
  fi
  git -C "$SOURCE_DIR" apply --whitespace=nowarn "$PATCH_PATH"
fi

rm -rf "$SOURCE_DIR/.git"
printf 'Fetched %s at %s into %s\n' "$SOURCE_URL" "$SOURCE_COMMIT" "$SOURCE_DIR"
