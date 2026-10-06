#!/usr/bin/env python3
"""Compatibility entrypoint for stock Boltz CLI callers and kit mode commands."""

from __future__ import annotations

import json
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path


PREFIX = "[mn-boltz2-opt]"
MODE_NAMES = {"off", "exact", "fast", "big"}
FLAG_OPTIONS = {
    "--write_full_pae", "--write_full_pde", "--override", "--use_msa_server",
    "--use_potentials", "--affinity_mw_correction", "--subsample_msa",
    "--no_kernels", "--write_embeddings",
}
VALUE_OPTIONS = {
    "--checkpoint", "--recycling_steps", "--sampling_steps", "--diffusion_samples",
    "--max_parallel_samples", "--step_scale", "--output_format", "--num_workers",
    "--seed", "--msa_server_url", "--msa_pairing_strategy", "--method",
    "--preprocessing-threads", "--sampling_steps_affinity", "--diffusion_samples_affinity",
    "--affinity_checkpoint", "--max_msa_seqs", "--num_subsampled_msa",
}


def _help() -> None:
    print(
        "Usage: docker run ... mn-boltz2-opt:2.2.1-cu130 predict INPUT --out_dir OUT [Boltz predict options]\n"
        "\nModes (set MN_BOLTZ2_MODE=off|fast|big; default: off):\n"
        "  off   stock Boltz 2.2.1 prediction\n"
        "  fast  optimized prediction; small documented numeric differences\n"
        "  big   lower-memory optimized prediction\n"
        "\nThe image also exposes the optimization kit directly, for example:\n"
        "  docker run ... mn-boltz2-opt:2.2.1-cu130 check --mode fast\n"
        "  docker run ... mn-boltz2-opt:2.2.1-cu130 pred --mode big --input INPUT --out_dir OUT"
    )


def _mode(environment: dict[str, str]) -> str:
    selected = (environment.get("MN_BOLTZ2_MODE") or environment.get("BOLTZ2_OPT") or "off").strip().lower()
    if selected not in MODE_NAMES:
        raise ValueError(f"{PREFIX} unsupported mode {selected!r}; use off, exact, fast, or big")
    return selected


def _configure_jit_cache(environment: dict[str, str], mode: str) -> None:
    """Keep compiled kernels between one-shot app containers, keyed by stack and GPU."""
    if mode == "off" or environment.get("TRITON_CACHE_DIR"):
        return
    cache_root = environment.get("BOLTZ_CACHE")
    if not cache_root:
        return
    try:
        import torch

        capability = torch.cuda.get_device_capability(0)
        cuda_version = (torch.version.cuda or "").replace(".", "")
        torch_version = importlib.metadata.version("torch")
        if not cuda_version or not capability:
            return
        key = f"torch{torch_version}-cu{cuda_version}-sm{capability[0]}{capability[1]}"
        path = Path(cache_root) / "boltz2_opt_jit" / key / "triton"
        path.mkdir(parents=True, exist_ok=True)
        environment["TRITON_CACHE_DIR"] = str(path)
    except (ImportError, OSError, RuntimeError):
        # The toolkit still has Triton's own per-container fallback when the cache
        # mount is read-only or no CUDA device is visible.
        return


def _parse_legacy(argv: list[str], mode: str, environment: dict[str, str]) -> tuple[list[str], str, Path, str, bool]:
    if len(argv) < 2:
        raise ValueError("predict requires an input YAML/FASTA path")
    input_path = argv[1]
    output_dir: str | None = None
    cache_path: str | None = None
    model = "boltz2"
    accelerator = "gpu"
    n_gpu = environment.get("MN_BOLTZ2_N_GPU", "1")
    forwarded: list[str] = []
    stock_extra: list[str] = []
    output_format = "mmcif"
    i = 2
    while i < len(argv):
        token = argv[i]
        if token == "--":
            stock_extra.extend(argv[i + 1 :])
            break
        key, equal, inline_value = token.partition("=")
        if key in {"--out_dir", "--cache", "--model", "--accelerator", "--n_gpu"}:
            if equal:
                value = inline_value
            else:
                i += 1
                if i >= len(argv):
                    raise ValueError(f"{key} requires a value")
                value = argv[i]
            if key == "--out_dir":
                output_dir = value
            elif key == "--cache":
                cache_path = value
            elif key == "--model":
                model = value
            elif key == "--accelerator":
                accelerator = value
            else:
                n_gpu = value
        elif key in FLAG_OPTIONS:
            if equal:
                raise ValueError(f"{key} does not take a value")
            forwarded.append(key)
        elif key in VALUE_OPTIONS:
            if equal:
                value = inline_value
            else:
                i += 1
                if i >= len(argv):
                    raise ValueError(f"{key} requires a value")
                value = argv[i]
            forwarded.extend((key, value))
            if key == "--output_format":
                output_format = value
        else:
            if mode == "off":
                stock_extra.append(token)
            else:
                raise ValueError(f"optimized mode does not support Boltz option {token!r}; use --mode off to pass upstream-only options")
        i += 1

    if model != "boltz2":
        raise ValueError(f"this image contains Boltz2 2.2.1; unsupported --model {model!r}")
    if mode != "off" and accelerator.lower() not in {"gpu", "cuda"}:
        raise ValueError(f"mode {mode} requires --accelerator gpu")
    if output_dir is None:
        raise ValueError("predict requires --out_dir so results can be collected consistently")
    try:
        n_gpu_count = int(n_gpu)
    except ValueError as exc:
        raise ValueError("MN_BOLTZ2_N_GPU/--n_gpu must be an integer") from exc
    if n_gpu_count < 1:
        raise ValueError("--n_gpu must be at least 1")
    if mode != "big" and n_gpu_count != 1:
        raise ValueError("--n_gpu greater than 1 is supported only in big mode")

    if cache_path:
        environment["BOLTZ_CACHE"] = cache_path
    environment.pop("BOLTZ2_OPT", None)
    if mode == "off":
        stock_extra[:0] = ["--model", model, "--accelerator", accelerator]
    out = Path(output_dir)
    command = [sys.executable, "-m", "boltz2_opt", "pred", "--mode", mode]
    if mode == "big":
        command.extend(("--n_gpu", str(n_gpu_count)))
    # Partial optimization is safe here: rejected kernels fall back to the
    # upstream Boltz operation, and the worker keeps the per-lever report in
    # pred_worker.log. Without this switch the kit exits 3 after writing valid
    # structures whenever a GPU-specific optimization is correctly declined.
    allow_partial = environment.get("MN_BOLTZ2_ALLOW_PARTIAL", "1").strip().lower()
    if mode != "off" and allow_partial not in {"0", "false", "no", "off"}:
        command.append("--allow-partial")
    command.extend(("--input", input_path, "--out_dir", str(out)))
    command.extend(forwarded)
    if mode == "off" and stock_extra:
        command.append("--")
        command.extend(stock_extra)
    return command, input_path, out, output_format, mode != "off", n_gpu_count


def _normalize_outputs(out_dir: Path, input_path: str, output_format: str, mode: str, n_gpu: int) -> None:
    """Mirror worker outputs into Boltz's usual boltz_results_*/predictions layout."""
    source_root = out_dir / "by_seed"
    if not source_root.is_dir():
        raise RuntimeError(f"optimized run returned successfully but did not create {source_root}")
    input_name = Path(input_path.rstrip("/" )).stem
    prediction_root = out_dir / f"boltz_results_{input_name}" / "predictions"
    records = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not records:
        raise RuntimeError("optimized run returned successfully without any per-record outputs")
    for record in records:
        seed_dirs = sorted(path for path in record.iterdir() if path.is_dir() and path.name.startswith("s"))
        if not seed_dirs:
            raise RuntimeError(f"optimized run produced no seed output for {record.name}")
        # Legacy application workflows make one Boltz call per input/replicate. Preserve every seed if the CLI kit is used with several.
        for seed_dir in seed_dirs:
            seed_label = seed_dir.name[1:]
            target_name = record.name if len(seed_dirs) == 1 else f"{record.name}_seed_{seed_label}"
            target = prediction_root / target_name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            for artifact in seed_dir.iterdir():
                if artifact.is_file():
                    shutil.copy2(artifact, target / artifact.name)
    metadata = {
        "mode": mode,
        "input": input_path,
        "output_format": output_format,
        "n_gpu": n_gpu,
        "allow_partial_optimization": mode != "off" and os.environ.get("MN_BOLTZ2_ALLOW_PARTIAL", "1").strip().lower() not in {"0", "false", "no", "off"},
        "worker_log": str(out_dir / "pred_worker.log"),
        "normalized_prediction_root": str(prediction_root),
    }
    (out_dir / "mn_boltz2_opt.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ.copy()
    try:
        mode = _mode(environment)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if not argv or argv == ["--help"] or argv == ["-h"]:
        _help()
        return 0
    # Keep the optimization kit's management commands available in the same image.
    if argv[0] == "install":
        if len(argv) == 3 and argv[1] == "--weights":
            weight_dir = argv[2]
        elif len(argv) == 2:
            weight_dir = argv[1]
        else:
            print(f"{PREFIX} usage: install --weights CACHE_DIR", file=sys.stderr)
            return 2
        return subprocess.run(
            [sys.executable, "-m", "boltz2_opt.weights", weight_dir], env=environment, check=False
        ).returncode
    if argv[0] in {"pred", "check", "warm"}:
        direct_mode = mode
        if "--mode" in argv:
            index = argv.index("--mode")
            if index + 1 < len(argv):
                direct_mode = argv[index + 1].strip().lower()
        _configure_jit_cache(environment, direct_mode)
        environment.pop("BOLTZ2_OPT", None)
        direct = [sys.executable, "-m", "boltz2_opt", *argv]
        return subprocess.run(direct, env=environment, check=False).returncode
    if argv[0] == "predict" and len(argv) > 1 and argv[1] in {"--help", "-h"}:
        environment.pop("BOLTZ2_OPT", None)
        return subprocess.run(["boltz", *argv], env=environment, check=False).returncode
    if argv[0] != "predict":
        print(f"{PREFIX} expected 'predict' or a kit command (pred/check/warm); see --help", file=sys.stderr)
        return 2
    try:
        command, input_path, output_dir, output_format, optimized, n_gpu = _parse_legacy(argv, mode, environment)
    except ValueError as error:
        print(f"{PREFIX} {error}", file=sys.stderr)
        return 2
    print(f"{PREFIX} mode={mode} input={input_path} output={output_dir}", flush=True)
    _configure_jit_cache(environment, mode)
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode != 0:
        return result.returncode
    if optimized:
        try:
            _normalize_outputs(output_dir, input_path, output_format, mode, n_gpu)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"{PREFIX} output normalization failed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
