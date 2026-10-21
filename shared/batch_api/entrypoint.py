#!/usr/bin/env python3
from __future__ import annotations

"""Common image-side CLI. Model-specific behavior lives in ``adapters.py``."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_batch_api import API_VERSION
from mn_batch_api.adapters import capabilities, run_batch
from mn_batch_api.contract import BatchRequestError, load_request, write_result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage() -> None:
    print(
        "Usage: mn-predict batch --manifest /work/batch.json --output-dir /work/output\n"
        "       mn-predict capabilities [--json]\n"
        "\nBatch manifest API: mn-predict/v1. The manifest holds the mode, GPU count,\n"
        "sample count, seed, engine options, and one input path per unique item ID."
    )


def _engine() -> str:
    engine = os.environ.get("MN_BATCH_ENGINE", "").strip()
    if engine not in {
        "rf3_foundry", "boltz2", "af3_torch", "af3_jax", "alphafold3", "openfold3",
        "esmfold2", "esmfold2_fast", "chai1", "atlasfold", "colabfold", "colabfold_opt",
        "openfold3_ob0", "opendde_native", "protenix_v1",
        "boltz2_legacy", "alphafast", "esmfold2_biohub", "openfold3_p2",
        "pxdesign_v05", "protenix_v2_opt", "protenix_v2_stock",
        "rf3_foundry_legacy", "boltzgen_fold_opt_msa",
    }:
        raise RuntimeError(f"MN_BATCH_ENGINE must identify a supported pilot image; got {engine!r}")
    return engine


def _batch(argv: list[str], engine: str) -> int:
    manifest: Path | None = None
    output_dir: Path | None = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in {"--manifest", "--output-dir"}:
            if i + 1 >= len(argv):
                raise BatchRequestError(f"{token} requires a value")
            value = Path(argv[i + 1]).expanduser()
            if token == "--manifest":
                manifest = value
            else:
                output_dir = value
            i += 2
            continue
        if token.startswith("--manifest="):
            manifest = Path(token.partition("=")[2]).expanduser()
        elif token.startswith("--output-dir="):
            output_dir = Path(token.partition("=")[2]).expanduser()
        else:
            raise BatchRequestError(f"unexpected batch option {token!r}")
        i += 1
    if manifest is None or output_dir is None:
        raise BatchRequestError("batch requires --manifest and --output-dir")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = load_request(manifest.resolve(), expected_engine=engine)
    started = time.monotonic()
    started_at = _now()
    try:
        item_results, execution = run_batch(request, output_dir)
    except Exception as exc:
        item_results = [
            {
                "id": item.item_id,
                "status": "failed",
                "outputs": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            for item in request.items
        ]
        execution = {"return_code": 1, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.monotonic() - started
    successes = sum(row.get("status") == "completed" for row in item_results)
    status = "completed" if successes == len(item_results) else ("partial" if successes else "failed")
    result: dict[str, Any] = {
        "api_version": API_VERSION,
        "batch_id": request.batch_id,
        "engine": request.engine,
        "mode": request.mode,
        "gpu_count": request.gpu_count,
        "samples_per_item": request.samples_per_item,
        "seed": request.seed,
        "status": status,
        "started_at": started_at,
        "elapsed_seconds": round(elapsed, 3),
        "items": item_results,
        "execution": execution,
    }
    write_result(output_dir / "mn-batch-result.json", result)
    print(json.dumps({key: result[key] for key in ("batch_id", "engine", "mode", "status", "elapsed_seconds")}, sort_keys=True), flush=True)
    return 0 if status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Docker callers can use either `batch ...` or the normalized
    # `mn-predict batch ...` spelling with the same image ENTRYPOINT.
    if args and args[0] == "mn-predict":
        args = args[1:]
    try:
        engine = _engine()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args or args[0] in {"--help", "-h"}:
        if args and engine == "boltz2":
            return subprocess.run(
                [sys.executable, "/usr/local/bin/mn-boltz2-entrypoint.py", *args], check=False
            ).returncode
        _usage()
        return 0
    if args[0] == "capabilities":
        payload = capabilities(engine)
        if len(args) == 2 and args[1] == "--json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif len(args) == 1:
            _usage()
        else:
            print("capabilities accepts only --json", file=sys.stderr)
            return 2
        return 0
    if args[0] == "batch":
        try:
            return _batch(args[1:], engine)
        except (BatchRequestError, OSError, ValueError) as exc:
            print(f"mn-predict: {exc}", file=sys.stderr)
            return 2
    if engine == "rf3_foundry" and args[0] in {"install", "pred", "check", "warm"}:
        args = ["bash", "/kit/rosettafold3/run.sh", *args]
    if engine == "boltz2":
        legacy = "/usr/local/bin/mn-boltz2-entrypoint.py"
        return subprocess.run([sys.executable, legacy, *args], check=False).returncode
    if engine == "af3_torch":
        return subprocess.run(["bash", "/kit/af3_torch/run.sh", *args], check=False).returncode
    if engine in {"af3_jax", "alphafold3", "openfold3"}:
        return subprocess.run(
            [sys.executable, "/usr/local/bin/cofolding-af3-runner.py", *args], check=False
        ).returncode
    return subprocess.run(args, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
