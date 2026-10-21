#!/usr/bin/env python3
"""Stage the pinned OpenFold3 kit for the local Ada and Blackwell profiles."""
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
        raise SystemExit("usage: prepare_build_context.py OPENFOLD3_OB0_COPY OPT_CORE_COPY")
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
        '    "sm89": {"arch": "ada", "cards": ("RTX4090",)},\n'
        '    "sm90": {"arch": "hopper", "cards": ("H100", "H200")},\n'
        '    "sm100": {"arch": "blackwell", "cards": ("B200",)},\n'
        '    "sm103": {"arch": "blackwell_ultra", "cards": ("B300",)},\n'
        '    "sm120": {"arch": "blackwell", "cards": ("RTX5090",)},',
    )

    a100 = (kit / "configs" / "a100.env").read_text(encoding="utf-8")
    target_line = 'export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}'
    for filename, card, note in (
        ("rtx4090.env", "RTX4090", "# RTX 4090 Ada profile (sm_89); validation pending.\n"),
        ("rtx5090.env", "RTX5090", "# Deployment-only until tested on a real RTX 5090 (sm_120).\n"),
    ):
        if a100.count(target_line) != 1:
            raise SystemExit("could not derive the local GPU profile from the pinned A100 configuration")
        profile = a100.replace(target_line, f'export MODEL_OPT_TARGET_GPU=${{MODEL_OPT_TARGET_GPU:-{card}}}', 1)
        (kit / "configs" / filename).write_text(note + profile, encoding="utf-8")


if __name__ == "__main__":
    main()
