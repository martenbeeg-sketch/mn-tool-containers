#!/usr/bin/env python3
"""Studio adapter for BioEmu-2, preserving the upstream CLI for direct use."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


WORK_ROOT = Path("/work").resolve()


def _work_path(value: str) -> Path:
    path = (WORK_ROOT / value).resolve()
    if path != WORK_ROOT and WORK_ROOT not in path.parents:
        raise ValueError(f"Input/output path escapes the job directory: {value}")
    return path


def _pdb_to_cif(source: Path, destination: Path) -> Path:
    import gemmi

    structure = gemmi.read_structure(str(source))
    if not structure or not any(len(model) for model in structure):
        raise ValueError(f"No coordinate models found in {source.name}.")
    structure.setup_entities()
    structure.assign_label_seq_id()
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(destination))
    return destination


def _common_args(request: dict) -> list[str]:
    args = [
        "--num-samples", str(int(request["num_samples"])),
        "--out", str(_work_path(request["output_dir"])),
        "--sampler", str(request.get("sampler", "edm")),
        "--physicality-filter", str(request.get("physicality_filter", "high_quality")),
        "--precision", str(request.get("precision", "bf16-mixed")),
        "--matmul-precision", str(request.get("matmul_precision", "high")),
    ]
    if request.get("batch_size") is not None:
        args.extend(["--batch-size", str(int(request["batch_size"]))])
    if request.get("seed") is not None:
        args.extend(["--seed", str(int(request["seed"]))])
    if request.get("num_steps") is not None:
        args.extend(["--num-steps", str(int(request["num_steps"]))])
    if request.get("compile_denoiser"):
        args.append("--compile-denoiser")
    return args


def run_studio_request(request_path: Path) -> int:
    request = json.loads(request_path.read_text())
    mode = str(request["mode"])
    output_root = _work_path(str(request["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    records = []

    if mode == "propagate":
        structures = request.get("structure_paths") or []
        if not structures:
            raise ValueError("Propagate mode requires at least one starting structure.")
        for index, relative_path in enumerate(structures, start=1):
            source = _work_path(str(relative_path))
            if not source.is_file():
                raise FileNotFoundError(f"Starting structure not found: {relative_path}")
            cif_path = source
            if source.suffix.lower() not in {".cif", ".mmcif"}:
                cif_path = _pdb_to_cif(source, _work_path(f"artifacts/input/converted_{index:03d}.cif"))
            stem = source.stem.replace(" ", "_")[:80] or f"complex_{index:03d}"
            output_dir = output_root / f"{index:03d}_{stem}"
            args = [
                "propagate", "--x0-cif", str(cif_path), "--num-samples", str(int(request["num_samples"])),
                "--lag-time-ns", str(float(request.get("lag_time_ns", 1.0))),
                "--out", str(output_dir), "--sampler", str(request.get("sampler", "edm")),
                "--physicality-filter", str(request.get("physicality_filter", "high_quality")),
                "--precision", str(request.get("precision", "bf16-mixed")),
                "--matmul-precision", str(request.get("matmul_precision", "high")),
            ]
            if request.get("batch_size") is not None:
                args.extend(["--batch-size", str(int(request["batch_size"]))])
            if request.get("seed") is not None:
                args.extend(["--seed", str(int(request["seed"]) + index - 1)])
            if request.get("num_steps") is not None:
                args.extend(["--num-steps", str(int(request["num_steps"]))])
            if request.get("compile_denoiser"):
                args.append("--compile-denoiser")
            print(f"BioEmu-2: propagating {source.name} ({index}/{len(structures)})", flush=True)
            result = subprocess.run(["bioemu2", *args], cwd=WORK_ROOT, check=False)
            if result.returncode:
                return int(result.returncode)
            records.append({"input": str(source.relative_to(WORK_ROOT)), "output": str(output_dir.relative_to(WORK_ROOT))})
    elif mode in {"sample-equilibrium", "predict-structure"}:
        sequences = request.get("protein_sequences") or []
        if not sequences:
            raise ValueError("Sequence mode requires at least one protein sequence.")
        entities = []
        for index, sequence in enumerate(sequences):
            normalized = "".join(str(sequence).split()).upper()
            if not normalized or any(residue not in "ACDEFGHIKLMNPQRSTVWYX" for residue in normalized):
                raise ValueError(f"Sequence {index + 1} contains unsupported amino-acid characters.")
            entities.append({"entity_type": "protein", "sequence": normalized, "asym_ids": [chr(65 + index)]})
        system_path = _work_path("artifacts/input/system.yaml")
        system_path.parent.mkdir(parents=True, exist_ok=True)
        system_path.write_text(yaml.safe_dump({"entities": entities}, sort_keys=False))
        args = [mode, "--system", str(system_path), *_common_args(request)]
        print(f"BioEmu-2: {mode} for {len(sequences)} protein chain(s)", flush=True)
        result = subprocess.run(["bioemu2", *args], cwd=WORK_ROOT, check=False)
        if result.returncode:
            return int(result.returncode)
        records.append({"input": "sequence system", "output": str(output_root.relative_to(WORK_ROOT))})
    else:
        raise ValueError(f"Unsupported BioEmu-2 mode: {mode}")

    summary = {"mode": mode, "records": records, "num_samples_per_system": int(request["num_samples"])}
    (output_root / "studio_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "--studio-request":
        os.execvp("bioemu2", ["bioemu2", *sys.argv[1:]])
    if len(sys.argv) != 3:
        raise SystemExit("Usage: --studio-request /work/request.json")
    return run_studio_request(_work_path(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
