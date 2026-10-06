#!/usr/bin/env python3
"""Make an isolated Chai-1 kit build context with RTX consumer configs."""
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"expected one match for {old!r} in {path}")
    path.write_text(text.replace(old, new))


def main() -> None:
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

    h100 = (kit / "configs" / "h100.env").read_text()
    target_line = 'export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-H100}'
    for filename, target in (("rtx4090.env", "RTX 4090"), ("rtx5090.env", "RTX 5090")):
        config = h100.replace(
            target_line,
            f'export MODEL_OPT_TARGET_GPU=${{MODEL_OPT_TARGET_GPU:-{target}}}',
            1,
        )
        if config == h100:
            raise SystemExit("could not make local RTX config from configs/h100.env")
        (kit / "configs" / filename).write_text(
            "# Consumer GPU deployment target; optimization levers remain uncertified until tested on this hardware.\n" + config
        )


if __name__ == "__main__":
    main()
