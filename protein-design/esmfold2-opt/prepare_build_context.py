#!/usr/bin/env python3
"""Apply the RTX adapter to a disposable copy of the upstream ESMFold2 kit."""
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
    dockerfile = kit / "environment" / "Dockerfile"

    lines = dockerfile.read_text().splitlines(keepends=True)
    case_start = next((i for i, line in enumerate(lines) if line.startswith(" case \"$STACK\" in")), None)
    if case_start is None:
        raise SystemExit(f"could not find STACK case table in {dockerfile}")
    build_start = next(
        (i for i in range(case_start, len(lines)) if 'if [ "$WHEELS_FROM" = build ]; then' in lines[i]),
        None,
    )
    if build_start is None:
        raise SystemExit(f"could not find WHEELS_FROM build branch in {dockerfile}")
    lines[case_start : build_start + 1] = [
        ' case "$STACK" in \\\n',
        '   img_ef2_fa) TORCH_CUDA_ARCH_LIST="9.0"; NVTE_ARCHS="90"; FLASH_ARCHS="90"; XFORMERS_ARCHS="9.0"; PINS_WHEELS="image wheels" ;; \\\n',
        '   img_esmfold2_a100) TORCH_CUDA_ARCH_LIST="8.0;9.0"; NVTE_ARCHS="80;90"; FLASH_ARCHS="80;90"; XFORMERS_ARCHS="8.0;9.0"; PINS_WHEELS="image_variants a100 wheels" ;; \\\n',
        '   img_mn_rtx4090_rtx5090) TORCH_CUDA_ARCH_LIST="8.9;12.0"; NVTE_ARCHS="89;120"; FLASH_ARCHS="80;120"; XFORMERS_ARCHS="9.0"; PINS_WHEELS="custom" ;; \\\n',
        '   *) echo "esmfold2: unsupported STACK=$STACK" >&2; exit 2 ;; \\\n',
        ' esac; export TORCH_CUDA_ARCH_LIST; \\\n',
        ' if [ "$WHEELS_FROM" = build ]; then \\\n',
    ]
    dockerfile.write_text("".join(lines))

    replace_once(
        dockerfile,
        "BUILD_VERSION=0.0.35+03b91d7.d20260904 FORCE_CUDA=1 XFORMERS_BUILD_TYPE=Release",
        'BUILD_VERSION=0.0.35+03b91d7.d20260904 FORCE_CUDA=1 XFORMERS_BUILD_TYPE=Release TORCH_CUDA_ARCH_LIST="$XFORMERS_ARCHS"',
    )
    replace_once(dockerfile, 'NVTE_CUDA_ARCHS="$SM_ARCHS"', 'NVTE_CUDA_ARCHS="$NVTE_ARCHS"')
    replace_once(dockerfile, 'FLASH_ATTN_CUDA_ARCHS="$SM_ARCHS"', 'FLASH_ATTN_CUDA_ARCHS="$FLASH_ARCHS"')

    lines = dockerfile.read_text().splitlines(keepends=True)
    pin_line = next((i for i, line in enumerate(lines) if "python -I -c" in line and "stock/PINS.json" in line), None)
    if pin_line is None or pin_line == 0 or "else" not in lines[pin_line - 1]:
        raise SystemExit(f"could not locate prebuilt wheel branch in {dockerfile}")
    lines.insert(
        pin_line,
        '   if [ "$STACK" = img_mn_rtx4090_rtx5090 ]; then echo "prebuilt wheels are not pinned for the RTX 4090/5090 multi-arch build; use WHEELS_FROM=build" >&2; exit 1; fi; \\\n',
    )
    dockerfile.write_text("".join(lines))

    arch = core / "opt_core" / "arch.py"
    replace_once(
        arch,
        'SM_CLASSES = {                      # sm class -> architecture word and the card labels of that class (labels only; detection never uses them)\n'
        '    "sm80": {"arch": "ampere", "cards": ("A100",)},\n'
        '    "sm90": {"arch": "hopper", "cards": ("H100", "H200")},\n'
        '    "sm100": {"arch": "blackwell", "cards": ("B200",)},\n'
        '    "sm103": {"arch": "blackwell_ultra", "cards": ("B300",)},\n'
        '}',
        'SM_CLASSES = {                      # sm class -> architecture word and the card labels of that class (labels only; detection never uses them)\n'
        '    "sm80": {"arch": "ampere", "cards": ("A100",)},\n'
        '    "sm89": {"arch": "ada", "cards": ("RTX 4090",)},\n'
        '    "sm90": {"arch": "hopper", "cards": ("H100", "H200")},\n'
        '    "sm100": {"arch": "blackwell", "cards": ("B200",)},\n'
        '    "sm103": {"arch": "blackwell_ultra", "cards": ("B300",)},\n'
        '    "sm120": {"arch": "blackwell", "cards": ("RTX 5090",)},\n'
        '}',
    )

    modes = kit / "opt" / "esmfold2_opt" / "modes.py"
    replace_once(
        modes,
        '(("h100", "h200", "gh200"), "9.0"), (("a100", "a800"), "8.0"), (("b200", "gb200"), "10.0"), (("b300", "gb300"), "10.3"),\n'
        '                      (("l40", "rtx 6000 ada", "l4"), "8.9"), (("a10", "a40", "rtx a6000"), "8.6")):',
        '(("h100", "h200", "gh200"), "9.0"), (("a100", "a800"), "8.0"), (("b200", "gb200"), "10.0"), (("b300", "gb300"), "10.3"),\n'
        '                      (("rtx 5090", "geforce rtx 5090"), "12.0"), (("rtx 4090", "geforce rtx 4090"), "8.9"),\n'
        '                      (("l40", "rtx 6000 ada", "l4"), "8.9"), (("a10", "a40", "rtx a6000"), "8.6")):',
    )

    cards = kit / "opt" / "esmfold2_opt" / "cards.py"
    replace_once(cards, 'for sm in ("sm80", "sm100", "sm103")', 'for sm in ("sm80", "sm89", "sm100", "sm103", "sm120")')

    h100 = (kit / "configs" / "h100.env").read_text()
    target_line = 'export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-H100}'
    for filename, target in (("rtx4090.env", "RTX 4090"), ("rtx5090.env", "RTX 5090")):
        config = h100.replace(target_line, f'export MODEL_OPT_TARGET_GPU=${{MODEL_OPT_TARGET_GPU:-{target}}}', 1)
        if config == h100:
            raise SystemExit("could not make local RTX configuration from configs/h100.env")
        (kit / "configs" / filename).write_text(
            "# Local ESMFold2 deployment target. RTX-class kernels remain marked uncertified until hardware validation.\n" + config
        )

    if len(sys.argv) > 3:
        # The kit's W4 device policy already replaces T6/T10 with its exact T1
        # transition and retains stock Triton tiles on <=101 KB cards. Reflect
        # those declared fallbacks in class resolution so the fail-closed mode
        # gate does not reject the entire mode on Ada/Blackwell consumer GPUs.
        registry = kit / "opt" / "esmfold2_opt" / "registry.py"
        text = registry.read_text()
        small_memory_classes = 'classes=("8.0", "9.0", "10.0", "10.3")'
        for name, old in (
            ("t3", '    "t3": Lever("t3", FIELD_W4, "ef2_w4.py", "re-tuned Triton tile table (transition + stage-5 GEMM)", "forward", "bitwise", "T2", ("w4", "tiles")),'),
            ("t6", '    "t6": Lever("t6", FIELD_W4, "ef2_w4.py", "row-block transition kernel", "forward", "bitwise", "T2", ("w4", "transition_rowblock")),'),
            ("t10", '    "t10": Lever("t10", FIELD_W4, "ef2_w4.py", "one-kernel fused pair transition (supersedes t6)", "forward", "T2", "T2", ("w4", "transition_fused")),'),
        ):
            if text.count(old) != 1:
                raise SystemExit(f"expected one registry row for {name} in {registry}")
            text = text.replace(old, old[:-2] + f", {small_memory_classes}),")
        overlay = Path(sys.argv[3]).resolve()
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_text(text)

        # The kit's fused PWA bias/softmax tile (BJ=64, 4 warps, 3 stages)
        # requests 108 KiB of shared memory on its MSA call. RTX 4090 reports
        # a 99 KiB per-block limit, so retain the fused kernel but halve its
        # key tile on Ada/Blackwell consumer targets. The reduction order is
        # still a T2/tolerance path; exact mode does not enable this lever.
        msa_driver = kit / "opt" / "forward" / "fast_inference" / "driver" / "ef2_msa_v2.py"
        msa_text = msa_driver.read_text()
        old = '    cfg = cfg or _M16_CFG\n    B, L, M, K = m.shape\n'
        new = (
            '    cfg = cfg or _M16_CFG\n'
            '    if m.device.type == "cuda" and torch.cuda.get_device_capability(m.device) in ((8, 9), (12, 0)) and cfg["BJ"] > 32:\n'
            '        cfg = dict(cfg)\n'
            '        cfg["BJ"] = 32\n'
            '        STATS["m16_bj32_small_mem"] += 1\n'
            '    B, L, M, K = m.shape\n'
        )
        if msa_text.count(old) != 1:
            raise SystemExit(f"expected one M16 config point in {msa_driver}")
        (overlay.parent / "ef2_msa_v2.py").write_text(msa_text.replace(old, new))


if __name__ == "__main__":
    main()
