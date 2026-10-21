#!/usr/bin/env python3
"""Prepare a private AtlasFold kit context with explicit consumer-GPU profiles."""
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected one match for {old!r} in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_build_context.py ATLASFOLD_COPY OPT_CORE_COPY")
    kit = Path(sys.argv[1]).resolve()
    core = Path(sys.argv[2]).resolve()

    arch = core / "opt_core" / "arch.py"
    replace_once(
        arch,
        '    "sm80": {"arch": "ampere", "cards": ("A100",)},\n'
        '    "sm90": {"arch": "hopper", "cards": ("H100", "H200")},\n'
        '    "sm100": {"arch": "blackwell", "cards": ("B200",)},\n'
        '    "sm103": {"arch": "blackwell_ultra", "cards": ("B300",)},',
        '    "sm80": {"arch": "ampere", "cards": ("A100",)},\n'
        '    "sm89": {"arch": "ada", "cards": ("RTX 4090",)},\n'
        '    "sm90": {"arch": "hopper", "cards": ("H100", "H200")},\n'
        '    "sm100": {"arch": "blackwell", "cards": ("B200",)},\n'
        '    "sm103": {"arch": "blackwell_ultra", "cards": ("B300",)},\n'
        '    "sm120": {"arch": "blackwell", "cards": ("RTX 5090",)},',
    )

    h100 = (kit / "configs" / "h100.env").read_text(encoding="utf-8")
    target_line = 'export AFO_TARGET_GPU="H100(sm90)"'
    for filename, target in (("rtx4090.env", "RTX 4090(sm89)"), ("rtx5090.env", "RTX 5090(sm120)")):
        if h100.count(target_line) != 1:
            raise SystemExit(f"could not make {filename} from the pinned h100 profile")
        profile = h100.replace(target_line, f'export AFO_TARGET_GPU="{target}"', 1)
        note = (
            "# AtlasFold kit profile for the RTX 4090. Runtime-tested with this image; optimization rows remain individually reported.\n"
            if filename == "rtx4090.env"
            else "# Deployment-only profile: AtlasFold's pinned CUDA 12.8 stack has not been tested on RTX 5090.\n"
        )
        (kit / "configs" / filename).write_text(note + profile, encoding="utf-8")


if __name__ == "__main__":
    main()
