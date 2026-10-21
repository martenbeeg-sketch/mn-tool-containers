#!/usr/bin/env python3
"""Stage local Ada and Blackwell profiles without modifying the pinned kit."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_build_context.py OPENDDE_COPY OPT_CORE_COPY")
    kit = Path(sys.argv[1]).resolve()
    core = Path(sys.argv[2]).resolve()
    if not (kit / "run.sh").is_file() or not (core / "opt_core").is_dir():
        raise SystemExit("expected the pinned OpenDDE kit and shared opt_core in the staging context")
    a100_path = kit / "configs" / "a100.env"
    a100 = a100_path.read_text(encoding="utf-8")
    marker = "export MODEL_OPT="
    if a100.count(marker) != 1:
        raise SystemExit("could not derive the local GPU profiles from the pinned A100 configuration")
    profile_body = a100[a100.index(marker) :]
    target_line = "export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}"
    if profile_body.count(target_line) != 1:
        raise SystemExit("could not find the A100 target label in the pinned profile")

    profiles = {
        "rtx4090": ("RTX4090", "8.9"),
        "rtx5090": ("RTX5090", "12.0"),
    }
    for name, (gpu, capability) in profiles.items():
        body = profile_body.replace(
            target_line,
            f"export MODEL_OPT_TARGET_GPU=${{MODEL_OPT_TARGET_GPU:-{gpu}}}",
            1,
        )
        heading = (
            f"# Local deployment profile for {gpu} (compute capability {capability}).\n"
            "# The pinned kit does not ship a tuned tile table for these cards. Providers inspect the running GPU;\n"
            "# unsupported rows fall back by name to stock operations.\n"
        )
        (kit / "configs" / f"{name}.env").write_text(heading + body, encoding="utf-8")

    dockerfile_path = kit / "environment" / "Dockerfile"
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    replacements = {
        "FROM nvidia/cuda:12.6.3-devel-ubuntu24.04": "FROM nvidia/cuda:12.8.1-devel-ubuntu24.04",
        "grep -v -E '^(#|opendde==)' /tmp/requirements.lock > /tmp/stack.txt":
        "grep -v -E '^(#|opendde==|torch[[:space:]@=<>!~]|nvidia-)' /tmp/requirements.lock > /tmp/stack.txt",
    }
    for old, new in replacements.items():
        if dockerfile.count(old) != 1:
            raise SystemExit(f"could not apply expected RTX 50-series Dockerfile patch: {old}")
        dockerfile = dockerfile.replace(old, new, 1)

    slash = chr(92)
    old_torch_lock_step = "&& python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt " + slash
    new_torch_steps = (
        "&& python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "
        "--extra-index-url https://pypi.org/simple 'torch==2.7.1+cu128' " + slash + "\n"
        " && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt " + slash
    )
    if dockerfile.count(old_torch_lock_step) != 1:
        raise SystemExit("could not find the pinned CUDA 12.6 Torch lock install to replace")
    dockerfile = dockerfile.replace(old_torch_lock_step, new_torch_steps, 1)
    dockerfile_path.write_text(dockerfile, encoding="utf-8")


if __name__ == "__main__":
    main()
