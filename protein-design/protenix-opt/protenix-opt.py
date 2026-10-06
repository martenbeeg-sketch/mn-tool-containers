#!/usr/bin/env python3
"""Selectable Protenix entrypoint for the off / exact / fast / big modes."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys


KIT_ROOT = "/opt/protenix-opt/protenix_v2"
KIT_PYTHON = os.path.join(KIT_ROOT, "opt")
MODE = os.environ.get("MN_PROTENIX_OPT_MODE", "off").strip().lower() or "off"
VALID_MODES = {"off", "exact", "fast", "big"}


def refuse(message: str, code: int = 3) -> "NoReturn":
    print(f"[protenix-opt] NOT ACTIVE: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def gpu_caps() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def cache_root(caps: list[str]) -> str:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "unknown"
    version, _, local = torch_version.partition("+")
    cuda = local.removeprefix("cu") if local.startswith("cu") else "unknown"
    cc = "-".join(cap.replace(".", "") for cap in caps) or "unknown"
    return os.path.join("/opt/protenix-jit", f"torch{version}-cu{cuda}-sm{cc}")


def set_cache_environment(env: dict[str, str], caps: list[str]) -> None:
    root = cache_root(caps)
    env.setdefault("MODEL_OPT_JIT_ROOT", root)
    env.setdefault("TRITON_CACHE_DIR", os.path.join(root, "triton"))
    env.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(root, "torch_extensions"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(root, "inductor"))
    env.setdefault("CUEQ_TRITON_CACHE_DIR", os.path.join(root, "cueq"))
    env.setdefault("PROTENIX_OPT_CACHE_DIR", os.path.join(root, "weights"))
    env.setdefault("INFOPT_FASTLN_BUILD_DIR", os.path.join(root, "fastln_stream_build"))
    for key in (
        "MODEL_OPT_JIT_ROOT",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUEQ_TRITON_CACHE_DIR",
        "PROTENIX_OPT_CACHE_DIR",
        "INFOPT_FASTLN_BUILD_DIR",
    ):
        os.makedirs(env[key], exist_ok=True)


def add_4090_ablations(env: dict[str, str], caps: list[str], mode: str) -> list[str]:
    """Disable kit levers for which this kit has no RTX 4090 (sm_89) cells."""
    if not caps or any(cap != "8.9" for cap in caps):
        return []
    # The kit refuses ablations of levers outside the selected mode.  The
    # block path and prebuilt LayerNorm are unsupported on sm_89 in every
    # optimized mode; the K2B-specific levers apply only to fast/big.
    required_by_mode = {
        "exact": ["blk2_block_path", "fastln_prebuilt"],
        "fast": ["blk2_block_path", "blk2_chunked_k2b", "k2b_flash_triattention", "fastln_prebuilt"],
        "big": ["blk2_block_path", "blk2_chunked_k2b", "k2b_flash_triattention", "fastln_prebuilt"],
    }
    required = required_by_mode.get(mode, [])
    selected = [value.strip() for value in env.get("MODEL_OPT_LEVERS_OFF", "").split(",") if value.strip()]
    for name in required:
        if name not in selected:
            selected.append(name)
    env["MODEL_OPT_LEVERS_OFF"] = ",".join(selected)
    return required


def take_option(args: list[str], option: str) -> tuple[list[str], list[str]]:
    """Return (option values, argv without the option), accepting --x V and --x=V."""
    values: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == option:
            if i + 1 >= len(args):
                refuse(f"{option} needs a value", 2)
            values.append(args[i + 1])
            i += 2
        elif arg.startswith(option + "="):
            values.append(arg.split("=", 1)[1])
            i += 1
        else:
            rest.append(arg)
            i += 1
    return values, rest


args = list(sys.argv[1:])
if args and args[0] == "protenix":
    args.pop(0)
if not args:
    args = ["--help"]

if MODE not in VALID_MODES:
    refuse(f"unsupported PROTENIX_OPT_MODE={MODE!r}; choose off, exact, fast, or big", 2)

try:
    requested_n_gpu = int(os.environ.get("MN_PROTENIX_OPT_N_GPU", "1"))
except ValueError:
    refuse("PROTENIX_OPT_N_GPU must be a positive integer", 2)
if requested_n_gpu < 1:
    refuse("PROTENIX_OPT_N_GPU must be a positive integer", 2)
if requested_n_gpu > 1 and MODE != "big":
    refuse("multiple GPUs are supported only by PROTENIX_OPT_MODE=big")

protenix = shutil.which("protenix")
if not protenix:
    refuse("stock protenix executable is missing", 127)

if MODE == "off":
    if requested_n_gpu != 1:
        refuse("PROTENIX_OPT_N_GPU must be 1 in off mode")
    os.environ.pop("MN_PROTENIX_OPT_MODE", None)
    os.environ.pop("MN_PROTENIX_OPT_N_GPU", None)
    for name in list(os.environ):
        if name.startswith(("PROTENIX_OPT", "PTX_", "FPF_", "INFOPT_", "PF_", "MODEL_OPT")) or name == "CUEQ_TRITON_CACHE_DIR":
            os.environ.pop(name, None)
    print("[protenix-opt] ACTIVE mode=off implementation=stock", file=sys.stderr, flush=True)
    os.execve(protenix, [protenix, *args], os.environ.copy())

cli_modes, args = take_option(args, "--mode")
if cli_modes and any(value.strip().lower() != MODE for value in cli_modes):
    refuse(f"CLI --mode conflicts with PROTENIX_OPT_MODE={MODE}", 2)
cli_gpu_counts, args = take_option(args, "--n_gpu")
if cli_gpu_counts:
    try:
        parsed_counts = {int(value) for value in cli_gpu_counts}
    except ValueError:
        refuse("--n_gpu values must be positive integers", 2)
    if len(parsed_counts) != 1 or next(iter(parsed_counts)) != requested_n_gpu:
        refuse(f"CLI --n_gpu conflicts with PROTENIX_OPT_N_GPU={requested_n_gpu}", 2)

if args and args[0] not in {"pred", "check", "warm"} and args[0] not in {"--help", "-h", "help"}:
    args.insert(0, "pred")
if args and args[0] == "warm" and requested_n_gpu > 1:
    refuse("warm accepts single-GPU mode only; use check or pred for big multi-GPU")

child_env = os.environ.copy()
child_env.pop("MN_PROTENIX_OPT_MODE", None)
child_env.pop("MN_PROTENIX_OPT_N_GPU", None)
child_env.pop("PROTENIX_OPT_MODE", None)  # the kit reads PROTENIX_OPT, not this wrapper's setting
child_env.pop("PROTENIX_OPT_N_GPU", None)  # the shell consumed this wrapper setting before Python started
child_env.setdefault("PROTENIX_ROOT_DIR", "/ref/protenix")
child_env["PROTENIX_OPT_HOME"] = KIT_PYTHON
child_env["MODEL_OPT"] = KIT_ROOT
path_parts = [part for part in child_env.get("PYTHONPATH", "").split(os.pathsep) if part]
if os.path.realpath(KIT_PYTHON) not in {os.path.realpath(part) for part in path_parts}:
    path_parts.insert(0, KIT_PYTHON)
child_env["PYTHONPATH"] = os.pathsep.join(path_parts)

caps = gpu_caps()
ablated = add_4090_ablations(child_env, caps, MODE)
set_cache_environment(child_env, caps)
print(
    f"[protenix-opt] ACTIVE mode={MODE} implementation=protenix_opt.cli "
    f"n_gpu={requested_n_gpu} gpu_cc={','.join(caps) or 'unknown'} "
    f"ablated={','.join(ablated) or 'none'}",
    file=sys.stderr,
    flush=True,
)
kit_args = ["-m", "protenix_opt", *args]
if args and args[0] in {"pred", "check", "warm"}:
    command = args[0]
    kit_args = ["-m", "protenix_opt", command, "--mode", MODE]
    if command in {"pred", "check"} and requested_n_gpu > 1:
        kit_args.extend(["--n_gpu", str(requested_n_gpu)])
    kit_args.extend(args[1:])
elif args:
    kit_args = ["-m", "protenix_opt", *args]

os.execve(sys.executable, [sys.executable, *kit_args], child_env)
