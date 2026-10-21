#!/usr/bin/env python3
"""Run the AtlasFold-M kit profiles through MN Cofolding and publish FoldBench rows.

The default run matches the existing five-target FoldBench panel: seeds 0–2,
five samples per seed, ten recycles, 200 diffusion steps, deterministic kit
execution, single-sequence input, and GPU 0 only. Profiles run by mode and
ascending total length; a prediction failure stops only that mode's sweep.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


APP = Path(os.environ.get("MN_COFOLDING_SOURCE", "/home/user/programs/git-projects/mn-cofolding")).resolve()
sys.path.insert(0, str(APP))

from mn_cofolding.jobs import JobStore  # noqa: E402
from mn_cofolding.results import parse_attempt_timings, prediction_records  # noqa: E402
from mn_cofolding.runtime import get_settings  # noqa: E402
from mn_cofolding.worker import run_job  # noqa: E402


DEFAULT_ROOT = Path(
    "/mnt/data/RESULTS/mn-cofolding-workdir/workdir/benchmarks/"
    "foldbench-heterodimer-length-20261004"
)
EXPERIMENT = "foldbench-heterodimer-length-2026-10-04"
MODES = ("off", "exact", "fast", "big")
SEEDS = (0, 1, 2)
SAMPLES = 5
RECYCLES = 10
STEPS = 200
GPU_IDS = (0,)
DOCKQ_THRESHOLD = 0.23


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return fallback


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GpuMonitor:
    def __init__(self, path: Path, gpu_id: int = 0, interval: float = 0.5):
        self.path = path
        self.gpu_id = gpu_id
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rows: list[dict[str, int | float]] = []
        self.started = 0.0

    def _sample(self) -> dict[str, int] | None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_id),
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            fields = [item.strip() for item in output.splitlines()[0].split(",")]
            return {
                "gpu_index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "utilization_pct": int(fields[2]),
            }
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            return None

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("elapsed_s", "gpu_index", "memory_used_mib", "utilization_pct"),
            )
            writer.writeheader()
            while not self.stop_event.is_set():
                sample = self._sample()
                if sample is not None:
                    elapsed = round(time.monotonic() - self.started, 3)
                    row = {"elapsed_s": elapsed, **sample}
                    self.rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def metrics(self) -> dict[str, Any]:
        values = [int(row["memory_used_mib"]) for row in self.rows]
        utils = [int(row["utilization_pct"]) for row in self.rows]
        return {
            str(self.gpu_id): {
                "peak_memory_mib": max(values) if values else None,
                "peak_utilization_pct": max(utils) if utils else None,
                "samples": len(values),
                "sampling_interval_seconds": self.interval,
                "scope": "device-wide nvidia-smi memory.used",
            }
        }


def gpu_info() -> dict[str, Any]:
    def query(args: list[str]) -> list[str]:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            return []

    gpu_lines = query(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader"])
    driver = query(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {"gpu_rows": gpu_lines, "driver_version": driver[0].strip() if driver else None}


def image_id(settings: Any) -> str | None:
    try:
        return subprocess.check_output(
            [settings.docker_executable, "image", "inspect", "--format", "{{.Id}}", settings.atlasfold_image],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def benchmark_input(target: dict[str, Any]) -> str:
    chains = target.get("sequences") or []
    if len(chains) < 2:
        raise ValueError(f"{target.get('pdb_id')} is not a multi-chain target")
    return f">{target['pdb_id']}_complex\n{':'.join(chains)}\n"


def job_key(pdb_id: str, mode: str) -> str:
    return f"{EXPERIMENT}:atlasfold:{pdb_id}:{mode}:single-gpu-no-msa"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def save_dashboard_data(root: Path, batch: dict[str, Any], registered: list[dict[str, Any]]) -> None:
    output = root / "atlasfold"
    atomic_json(output / "benchmark_summary.json", batch)
    write_csv(output / "profile_summary.csv", batch.get("profile_summaries", []))
    write_csv(output / "structure_scores.csv", batch.get("structures", []))
    atomic_json(root / "registered_jobs.json", {"experiment": EXPERIMENT, "jobs": registered})


def kit_lever_lines(run_dir: Path) -> dict[str, Any]:
    texts = []
    for name in ("stdout.log", "stderr.log"):
        path = run_dir / name
        if path.is_file():
            texts.append(path.read_text(errors="replace"))
    text = "\n".join(texts)
    return {
        "active": re.findall(r"\[atlasfold-opt\] ACTIVE .*", text),
        "levers": re.findall(r"\[atlasfold-opt\] LEVER .*", text),
        "final": re.findall(r"\[atlasfold-opt\] FINAL .*", text),
        "not_active": re.findall(r"\[atlasfold-opt\] NOT ACTIVE:.*", text),
    }


def run_cold_compile_warmups(root: Path, settings: Any, modes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Run the kit's 640-token warm command once per mode before timed profiles."""
    warm_root = root / "atlasfold" / "warmups"
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_root = warm_root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        command = [
            settings.docker_executable,
            "run",
            "--rm",
            "--name",
            f"mn-atlasfold-warm-{mode}",
            "--gpus",
            '"device=0"',
            "--shm-size=16g",
            "--mount",
            f"type=bind,src={settings.reference_dir.resolve()},dst=/reference,readonly",
            "--mount",
            f"type=bind,src={settings.cache_dir.resolve()},dst=/cache",
            "--env",
            "HOME=/cache/home",
            "--env",
            "XDG_CACHE_HOME=/cache/xdg",
            "--env",
            "ATLASFOLD_WEIGHTS_DIR=/reference/atlasfold",
            "--env",
            "MODEL_OPT_JIT_ROOT=/cache/atlasfold/jit",
            *( ["--env", "MODEL_OPT_LEVERS_OFF=ln_bf16"] if mode != "off" else [] ),
            settings.atlasfold_image,
            "warm",
            "--config",
            "rtx4090",
            "--mode",
            mode,
            "--det",
            "1",
            "--n_gpu",
            "1",
            "--weights",
            "/reference/atlasfold",
            "--multimer",
        ]
        (mode_root / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        monitor = GpuMonitor(mode_root / "gpu_telemetry.csv", 0)
        monitor.start()
        started = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=3 * 60 * 60, check=False)
            stdout, stderr, return_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return_code = -1
            stderr += "\nWarmup exceeded the 3-hour timeout.\n"
        except OSError as exc:
            stdout, stderr, return_code = "", f"{type(exc).__name__}: {exc}", -1
        wall = time.monotonic() - started
        monitor.stop()
        (mode_root / "stdout.log").write_text(stdout, encoding="utf-8", errors="replace")
        (mode_root / "stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
        all_text = stdout + "\n" + stderr
        result = {
            "mode": mode,
            "status": "completed" if return_code == 0 and "exit=0" in all_text else "failed",
            "return_code": return_code,
            "wall_seconds": round(wall, 3),
            "gpu_metrics": monitor.metrics(),
            "command": command,
            "kit_output": [line for line in all_text.splitlines() if "atlasfold-opt" in line],
            "note": "Synthetic multimer warmup; one residue sequence of 640 tokens, 1 recycle, 1 sample, and 2 diffusion steps. Excluded from prediction timing.",
        }
        atomic_json(mode_root / "summary.json", result)
        results[mode] = result
        print(
            f"WARMUP mode={mode} status={result['status']} wall_s={result['wall_seconds']} "
            f"peak_gpu0_mib={result['gpu_metrics'].get('0', {}).get('peak_memory_mib')}",
            flush=True,
        )
    return results


def compare_exact_profiles(root: Path, off_dir: Path, exact_dir: Path, target: dict[str, Any]) -> dict[str, Any]:
    """Compare all atom coordinates for matching deterministic off/exact samples."""
    try:
        import gemmi
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"gemmi missing: {exc}"}

    def collect(run_dir: Path) -> dict[tuple[int, int], dict[tuple[Any, ...], tuple[float, float, float]]]:
        mapping: dict[tuple[int, int], dict[tuple[Any, ...], tuple[float, float, float]]] = {}
        for record in prediction_records(run_dir):
            seed_text = str(record.get("seed") or "")
            seed_match = re.search(r"(\d+)", seed_text)
            if not seed_match or record.get("sample") is None:
                continue
            structure = gemmi.read_structure(str(record["structure_path"]))
            atoms: dict[tuple[Any, ...], tuple[float, float, float]] = {}
            for chain in structure[0]:
                for residue in chain:
                    for atom in residue:
                        key = (chain.name, residue.seqid.num, residue.name, atom.name, str(atom.altloc), atom.element.name)
                        atoms[key] = (atom.pos.x, atom.pos.y, atom.pos.z)
            mapping[(int(seed_match.group(1)), int(record["sample"]))] = atoms
        return mapping

    off = collect(off_dir)
    exact = collect(exact_dir)
    def artifact_hashes(run_dir: Path) -> dict[str, str]:
        output_dir = run_dir / "output" / f"{target['pdb_id']}_complex"
        return {
            path.relative_to(output_dir).as_posix(): sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        }

    off_artifacts = artifact_hashes(off_dir)
    exact_artifacts = artifact_hashes(exact_dir)
    shared_artifacts = set(off_artifacts) & set(exact_artifacts)
    artifact_mismatches = sorted(
        name for name in shared_artifacts if off_artifacts[name] != exact_artifacts[name]
    )
    artifact_file_sets_match = set(off_artifacts) == set(exact_artifacts)
    common = sorted(set(off) & set(exact))
    rows = []
    for key in common:
        left, right = off[key], exact[key]
        atom_keys = sorted(set(left) & set(right))
        differences = [abs(a - b) for atom_key in atom_keys for a, b in zip(left[atom_key], right[atom_key])]
        rows.append(
            {
                "seed": key[0],
                "sample": key[1],
                "off_atoms": len(left),
                "exact_atoms": len(right),
                "matched_atoms": len(atom_keys),
                "max_abs_coordinate_error_angstrom": max(differences, default=None),
                "rms_coordinate_error_angstrom": (
                    (sum(value * value for value in differences) / len(differences)) ** 0.5 if differences else None
                ),
            }
        )
    maxima = [row["max_abs_coordinate_error_angstrom"] for row in rows if row["max_abs_coordinate_error_angstrom"] is not None]
    result = {
        "pdb_id": target["pdb_id"],
        "target_residues": target["total_residues"],
        "status": "compared" if common else "no_matching_outputs",
        "n_samples_compared": len(common),
        "all_atom_keys_match": all(row["off_atoms"] == row["exact_atoms"] == row["matched_atoms"] for row in rows),
        "max_abs_coordinate_error_angstrom": max(maxima, default=None),
        "samples": rows,
        "artifact_files_compared": len(shared_artifacts),
        "artifact_file_sets_match": artifact_file_sets_match,
        "mismatched_artifacts": artifact_mismatches,
        "all_prediction_artifacts_bitwise_identical": artifact_file_sets_match and not artifact_mismatches,
        "note": "Coordinates are compared directly without structural superposition. Every file in the off and exact prediction output directories is also compared by SHA256.",
    }
    exact_path = root / "atlasfold" / "exact_coordinate_comparison.json"
    previous = read_json(exact_path, {"targets": []})
    previous_rows = [row for row in previous.get("targets", []) if row.get("pdb_id") != target["pdb_id"]]
    previous_rows.append(result)
    atomic_json(exact_path, {"targets": previous_rows})
    return result


def import_score_helper(root: Path):
    path = root / "score_benchmarks.py"
    spec = importlib.util.spec_from_file_location("foldbench_score_benchmarks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load scoring helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OST_COMPAT.mkdir(parents=True, exist_ok=True)
    libtiff = sorted((module.OST.parent.parent / "lib").glob("libtiff.so.6*"))
    link = module.OST_COMPAT / "libtiff.so.5"
    if libtiff and not link.exists():
        link.symlink_to(libtiff[0].resolve())
    return module


def confidence_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in ("mean_plddt", "ptm", "iptm", "ranking_score", "max_pae")
        if record.get(key) is not None
    }


def score_predictions(
    score_helper: Any,
    target: dict[str, Any],
    records: list[dict[str, Any]],
    mode: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    reference = Path(target["native_normalized_file"])
    scored = []
    for record in records:
        prediction = Path(record["structure_path"])
        seed_match = re.search(r"seed-(\d+)", str(record.get("seed") or record.get("label")))
        if not seed_match:
            seed_match = re.search(r"Seed (\d+), sample", str(record.get("label") or ""))
        if not seed_match or record.get("sample") is None:
            continue
        seed = int(seed_match.group(1))
        sample = int(record["sample"])
        detail = (
            run_dir
            / "artifacts"
            / "quality"
            / "openstructure"
            / f"{target['pdb_id']}_seed-{seed}_sample-{sample}.json"
        )
        metrics = score_helper.ost_score(prediction, reference, detail)
        confidence = confidence_row(record)
        row = {
            **metrics,
            "pdb_id": target["pdb_id"],
            "total_residues": target["total_residues"],
            "engine": "atlasfold",
            "mode": mode,
            "seed": seed,
            "sample": sample,
            "prediction": str(prediction),
            "confidence": confidence,
            "mean_plddt": confidence.get("mean_plddt"),
            "ptm": confidence.get("ptm"),
            "iptm": confidence.get("iptm"),
            "ranking_score": confidence.get("ranking_score"),
            "success": metrics.get("DockQ") is not None and metrics["DockQ"] >= DOCKQ_THRESHOLD,
            "ost_detail": str(detail),
        }
        scored.append(row)
    return scored


def profile_from_job(
    target: dict[str, Any],
    mode: str,
    job: dict[str, Any],
    wall: float,
    gpu_metrics: dict[str, Any],
    score_helper: Any,
    samples_per_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(job["run_dir"])
    metadata = job["metadata"]
    result = job["result"]
    records = prediction_records(run_dir)
    expected = len(SEEDS) * samples_per_seed
    timings = parse_attempt_timings(run_dir, "af3", model_name="atlasfold", samples_per_seed=samples_per_seed)
    by_seed: dict[int, float] = {}
    phase_by_seed: dict[int, dict[str, float]] = {}
    for attempt in timings:
        match = re.search(r"seed (\d+)", str(attempt.get("Attempt") or ""), re.I)
        if not match:
            continue
        seed = int(match.group(1))
        forward = float(attempt.get("Inference time") or 0)
        by_seed[seed] = by_seed.get(seed, 0.0) + forward
        phase_by_seed[seed] = {
            "forward_seconds": by_seed[seed],
            "output_processing_seconds": float(attempt.get("Output processing time") or 0),
        }
    scored = score_predictions(score_helper, target, records, mode, run_dir)
    dockq = [float(row["DockQ"]) for row in scored if row.get("DockQ") is not None]
    valid = result.get("status") == "completed" and len(records) == expected
    status = "completed" if valid else str(result.get("status") or "failed")
    reasons = []
    if not valid:
        reasons.append(result.get("message") or "expected structure count was not produced")
    if len(records) != expected:
        reasons.append(f"found {len(records)} of {expected} expected structures")
    if len(scored) != len(records):
        reasons.append(f"scored {len(scored)} of {len(records)} structures")
    forward_total = sum(by_seed.values()) if by_seed else None
    by_seed_text = {str(seed): round(value, 4) for seed, value in sorted(by_seed.items())}
    memory = {gpu: metrics.get("peak_memory_mib") for gpu, metrics in gpu_metrics.items()}
    profile = {
        "pdb_id": target["pdb_id"],
        "target_title": target.get("title", target["pdb_id"]),
        "total_residues": target["total_residues"],
        "engine": "atlasfold",
        "mode": mode,
        "run_status": status,
        "reason": "; ".join(reasons),
        "app_status": metadata.get("status"),
        "job_id": metadata.get("job_id"),
        "job_key": job_key(target["pdb_id"], mode),
        "run_dir": str(run_dir),
        "n_structures_found": len(records),
        "expected_structures": expected,
        "n_scored": len(dockq),
        "mean_DockQ": statistics.mean(dockq) if dockq else None,
        "median_DockQ": statistics.median(dockq) if dockq else None,
        "best_DockQ": max(dockq) if dockq else None,
        "successes": sum(value >= DOCKQ_THRESHOLD for value in dockq),
        "success_rate": sum(value >= DOCKQ_THRESHOLD for value in dockq) / len(dockq) if dockq else None,
        "wall_seconds": round(wall, 3),
        "inference_seconds": round(forward_total, 4) if forward_total is not None else None,
        "inference_seconds_by_seed": by_seed_text,
        "forward_display_seconds": round(statistics.median(by_seed.values()), 4) if by_seed else None,
        "forward_per_seed_seconds": round(statistics.median(by_seed.values()), 4) if by_seed else None,
        "per_seed_phases": phase_by_seed,
        "timing_repeats": len(by_seed),
        "timing_scope": "one five-sample prediction batch per seed; median forward time across seeds; separate model load in each app job",
        "gpu_ids": list(GPU_IDS),
        "gpu_count": 1,
        "peak_memory_mib_by_gpu": memory,
        "gpu_metrics": gpu_metrics,
        "mode_levers_off": ["ln_bf16"] if mode != "off" else [],
        "kit_report": kit_lever_lines(run_dir),
        "completed_seeds": len(by_seed),
        "msa_rows_by_chain": None,
    }
    profile["profile_complete"] = (
        status == "completed"
        and len(records) == expected
        and len(dockq) == expected
    )
    return profile, scored


def register_job(root: Path, job: dict[str, Any], target: dict[str, Any], mode: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / "registered_jobs.json"
    payload = read_json(path, {"experiment": EXPERIMENT, "jobs": []})
    rows = payload.get("jobs", []) if isinstance(payload, dict) else []
    rows = [
        row
        for row in rows
        if not (
            isinstance(row, dict)
            and row.get("target") == target["pdb_id"]
            and row.get("engine") == "atlasfold"
            and row.get("mode") == mode
        )
    ]
    run_dir = Path(job["run_dir"])
    rows.append(
        {
            "benchmark_key": job_key(target["pdb_id"], mode),
            "target": target["pdb_id"],
            "engine": "atlasfold",
            "family": "MN Cofolding AtlasFold-M",
            "mode": mode,
            "job_id": job["metadata"].get("job_id"),
            "status": profile.get("app_status"),
            "predictions": profile.get("n_structures_found"),
            "quality_n_scored": profile.get("n_scored"),
            "mean_DockQ": profile.get("mean_DockQ"),
            "quality_scores_csv": str(run_dir / "artifacts" / "quality" / "structure_scores.csv"),
            "run_dir": str(run_dir),
        }
    )
    rows.sort(key=lambda row: (str(row.get("target")), str(row.get("engine")), str(row.get("mode"))))
    return rows


def update_job_quality_artifacts(job: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    root = Path(job["run_dir"]) / "artifacts" / "quality"
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "structure_scores.csv", rows)


def run_mode_job(
    store: JobStore,
    settings: Any,
    target: dict[str, Any],
    mode: str,
    seeds: list[int],
    samples: int,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    input_text = benchmark_input(target)
    created = store.create(
        model_name="atlasfold",
        input_text=input_text,
        input_format="fasta",
        name=f"FoldBench · {target['pdb_id']} · AtlasFold-M · {mode} · single sequence",
        msa_mode="single_sequence",
        msa_source="single_sequence",
        seeds=seeds,
        num_diffusion_samples=samples,
        num_recycles=RECYCLES,
        optimization_mode=mode,
        msa_max_depth=1,
        cpu_threads=4,
        gpu=True,
    )
    job_id = str(created["metadata"]["job_id"])
    store.update_status(
        job_id,
        "queued",
        benchmark_experiment=EXPERIMENT,
        benchmark_key=job_key(target["pdb_id"], mode),
        benchmark_target=target["pdb_id"],
        benchmark_engine="atlasfold",
        benchmark_mode=mode,
    )
    run_dir = Path(created["run_dir"])
    benchmark_artifacts = run_dir / "artifacts" / "benchmark"
    benchmark_artifacts.mkdir(parents=True, exist_ok=True)
    (benchmark_artifacts / "target_manifest.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "target": target,
                "input_sha256": hashlib.sha256(input_text.encode()).hexdigest(),
                "input_condition": "AtlasFold-M with AtlasLM sequence embeddings; no external AA MSA",
                "mode": mode,
                "seeds": seeds,
                "samples_per_seed": samples,
                "num_recycles": RECYCLES,
                "num_steps": STEPS,
                "gpu_ids": list(GPU_IDS),
                "msa_rows_by_chain": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    monitor = GpuMonitor(run_dir / "artifacts" / "gpu_telemetry.csv", 0)
    monitor.start()
    started = time.monotonic()
    deadline = started + 3 * 60 * 60
    while True:
        run_job(job_id, settings=settings)
        current = store.get(job_id)
        state = current["metadata"].get("status")
        if state in {"completed", "failed", "cancelled"}:
            break
        if time.monotonic() >= deadline:
            store.finish(job_id, success=False, return_code=-1, message="Timed out waiting for MN Cofolding scheduler resources")
            current = store.get(job_id)
            break
        time.sleep(5)
    wall = time.monotonic() - started
    monitor.stop()
    return store.get(job_id), wall, monitor.metrics()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--targets", default="", help="comma-separated PDBs; defaults to all targets")
    parser.add_argument("--modes", default=",".join(MODES), help="comma-separated kit modes")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--resume", action="store_true", help="reuse complete profile rows in benchmark_summary.json")
    args = parser.parse_args()

    root = args.results_root.resolve()
    output = root / "atlasfold"
    output.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    weights_root = settings.reference_dir / "atlasfold"
    manifest = read_json(root / "targets.json", {})
    target_rows = sorted(manifest.get("targets", []), key=lambda row: int(row.get("total_residues", 0)))
    target_filter = {item.strip().upper() for item in args.targets.split(",") if item.strip()}
    if target_filter:
        target_rows = [row for row in target_rows if str(row.get("pdb_id", "")).upper() in target_filter]
    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    if any(mode not in MODES for mode in modes):
        raise SystemExit(f"modes must be selected from {', '.join(MODES)}")
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds or args.samples < 1:
        raise SystemExit("provide at least one seed and one sample")
    if not weights_root.is_dir():
        raise SystemExit(f"AtlasFold weights not found: {weights_root}")

    score_helper = import_score_helper(root)
    previous = read_json(output / "benchmark_summary.json", {})
    profiles: list[dict[str, Any]] = list(previous.get("profile_summaries", [])) if args.resume else []
    structures: list[dict[str, Any]] = list(previous.get("structures", [])) if args.resume else []
    recorded_modes = tuple(
        mode
        for mode in MODES
        if mode in modes or any(row.get("mode") == mode for row in profiles)
    )
    registered_file = read_json(root / "registered_jobs.json", {"experiment": EXPERIMENT, "jobs": []})
    registered = list(registered_file.get("jobs", [])) if isinstance(registered_file, dict) else []
    previous_keys = {(row.get("pdb_id"), row.get("mode")): row for row in profiles if isinstance(row, dict)}

    gpu = gpu_info()
    try:
        kit_version = subprocess.check_output(
            [
                settings.docker_executable,
                "image",
                "inspect",
                "--format",
                '{{ index .Config.Labels "org.mn.atlasfold.optimizer-kit-revision" }}',
                settings.atlasfold_image,
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        kit_version = "kit version recorded in image build recipe"

    weights = {
        name: sha256(path)
        for name, path in {
            "atlasfold_m": weights_root / "atlasfold-m-260725" / "weights" / "atlasfold-m-260725.pth",
            "atlaslm_3b": weights_root / "atlaslm-3b-base" / "weights" / "atlaslm_3b_base.pth",
        }.items()
    }
    conditions = {
        "status": "running",
        "experiment": EXPERIMENT,
        "image": settings.atlasfold_image,
        "image_id": image_id(settings),
        "upstream_commit": "992067e67df29b665c501e0d2e9ead9dd4ba9b69",
        "optimizer_kit": "AtlasFold kit 0.2.44",
        "container_build_recipe": "mn-tool-containers/protein-design/atlasfold-opt",
        "weights_sha256": weights,
        "weight_root": str(weights_root),
        "hardware": gpu,
        "gpu_ids": list(GPU_IDS),
        "seeds": seeds,
        "samples_per_seed": args.samples,
        "num_recycles": RECYCLES,
        "num_steps": STEPS,
        "deterministic": True,
        "modes": list(recorded_modes),
        "msa": "No external AA MSA. AtlasFold-M computes sequence embeddings with AtlasLM-3B.",
        "templates": "Not used by the AtlasFold-M prediction command.",
        "timing_note": "App wall time includes worker resource acquisition, Docker startup, model load, inference, and output. Forward time is parsed from kit PHASE lines and summed over seeds. GPU memory is device-wide nvidia-smi sampled every 0.5 seconds on GPU 0.",
        "warmup_note": "A separate kit warm command is run once per mode on a synthetic 640-token multimer; its time and memory are retained but excluded from profiles. Each profile then records three seed-batch forward timings (five samples per seed).",
        "rtx4090_mode_adjustment": {
            "levers_ablated": ["ln_bf16"],
            "reason": "The first RTX 4090 fast warmup recorded 192 fp32_params fallbacks and zero served calls for this exact-class lever; the kit's fail-closed gate refused the mode. Reruns ablate that unused lever by name and retain every other kit gate and lever report.",
        },
        "quality_metric": "OpenStructure DockQ-v2 against native assembly normalized to the two protein chains",
        "success_threshold_DockQ": DOCKQ_THRESHOLD,
        "gpu_by_mode": {mode: [0] for mode in recorded_modes},
        "kit_check": "All four modes passed run.sh check --config rtx4090 with all three pinned weight files present. Optimized runs ablate ln_bf16 by name after its warmup census showed only fp32_params fallback calls.",
        "kit_version_probe": kit_version.strip()[-500:],
        "started_at": now(),
    }
    previous_conditions = previous.get("conditions", {}) if isinstance(previous, dict) else {}
    if args.resume:
        exact_validation = (
            previous_conditions.get("exact_validation", {})
            if isinstance(previous_conditions, dict)
            else {}
        )
        if not exact_validation:
            exact_validation = {}
            for row in profiles:
                comparison = row.get("exact_coordinate_comparison")
                if row.get("mode") != "exact" or not isinstance(comparison, dict):
                    continue
                exact_validation[str(row.get("pdb_id"))] = {
                    "pass": row.get("exact_validation_pass"),
                    "max_abs_coordinate_error_angstrom": comparison.get("max_abs_coordinate_error_angstrom"),
                    "n_samples_compared": comparison.get("n_samples_compared"),
                    "all_prediction_artifacts_bitwise_identical": comparison.get("all_prediction_artifacts_bitwise_identical"),
                    "artifact_files_compared": comparison.get("artifact_files_compared"),
                }
        if exact_validation:
            conditions["exact_validation"] = exact_validation

    batch = {
        "experiment": EXPERIMENT,
        "conditions": conditions,
        "profile_summaries": profiles,
        "structures": structures,
    }
    save_dashboard_data(root, batch, registered)

    warmups = run_cold_compile_warmups(root, settings, modes)
    if args.resume:
        for mode in recorded_modes:
            if mode not in warmups:
                saved_warmup = read_json(output / "warmups" / mode / "summary.json", {})
                if saved_warmup:
                    warmups[mode] = saved_warmup
    conditions["cold_compile_runs"] = warmups
    conditions["warmups_all_completed"] = all(item.get("status") == "completed" for item in warmups.values())
    conditions["updated_at"] = now()
    batch.update({"conditions": conditions})
    save_dashboard_data(root, batch, registered)

    store = JobStore(settings)
    for mode in modes:
        failed_target: str | None = None
        if warmups.get(mode, {}).get("status") != "completed":
            failed_target = "kit warmup"
        for target in target_rows:
            pdb_id = str(target["pdb_id"])
            prior = previous_keys.get((pdb_id, mode))
            expected_levers_off = ["ln_bf16"] if mode != "off" else []
            compatible_resume = (
                prior is not None
                and prior.get("profile_complete")
                and prior.get("mode_levers_off", [] if mode == "off" else None) == expected_levers_off
            )
            if args.resume and compatible_resume:
                print(f"RESUME reuse target={pdb_id} mode={mode}", flush=True)
                continue
            if failed_target:
                skipped_status = "not_supported" if failed_target == "kit warmup" else "skipped"
                profiles = [row for row in profiles if not (row.get("pdb_id") == pdb_id and row.get("mode") == mode)]
                profiles.append(
                    {
                        "pdb_id": pdb_id,
                        "target_title": target.get("title", pdb_id),
                        "total_residues": target.get("total_residues"),
                        "engine": "atlasfold",
                        "mode": mode,
                        "run_status": skipped_status,
                        "reason": (
                            f"Kit warmup failed: {warmups.get(mode, {}).get('return_code')}"
                            if failed_target == "kit warmup"
                            else f"Stopped after earlier {mode} failure on {failed_target}; larger target not tested"
                        ),
                        "n_structures_found": 0,
                        "expected_structures": len(seeds) * args.samples,
                        "n_scored": 0,
                        "gpu_ids": [0],
                        "gpu_count": 1,
                        "peak_memory_mib_by_gpu": {},
                        "profile_complete": False,
                    }
                )
                batch.update({"profile_summaries": profiles, "structures": structures})
                save_dashboard_data(root, batch, registered)
                continue

            print(f"START target={pdb_id} aa={target['total_residues']} mode={mode} seeds={seeds} samples={args.samples} gpu=0", flush=True)
            try:
                job, wall, gpu_metrics = run_mode_job(store, settings, target, mode, seeds, args.samples)
            except Exception as exc:
                failed_target = pdb_id
                print(f"EXCEPTION target={pdb_id} mode={mode} error={type(exc).__name__}: {exc}", flush=True)
                profiles = [row for row in profiles if not (row.get("pdb_id") == pdb_id and row.get("mode") == mode)]
                profiles.append(
                    {
                        "pdb_id": pdb_id,
                        "target_title": target.get("title", pdb_id),
                        "total_residues": target.get("total_residues"),
                        "engine": "atlasfold",
                        "mode": mode,
                        "run_status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "n_structures_found": 0,
                        "expected_structures": len(seeds) * args.samples,
                        "n_scored": 0,
                        "gpu_ids": [0],
                        "gpu_count": 1,
                        "peak_memory_mib_by_gpu": {},
                        "profile_complete": False,
                    }
                )
                batch.update({"profile_summaries": profiles, "structures": structures})
                conditions["updated_at"] = now()
                save_dashboard_data(root, batch, registered)
                continue
            profile, scored = profile_from_job(target, mode, job, wall, gpu_metrics, score_helper, args.samples)
            update_job_quality_artifacts(job, scored)
            profiles = [row for row in profiles if not (row.get("pdb_id") == pdb_id and row.get("mode") == mode)]
            profiles.append(profile)
            profiles.sort(key=lambda row: (next((int(t.get("total_residues", 0)) for t in target_rows if t.get("pdb_id") == row.get("pdb_id")), 0), MODES.index(row.get("mode", "off"))))
            structures = [
                row for row in structures
                if not (row.get("pdb_id") == pdb_id and row.get("engine") == "atlasfold" and row.get("mode") == mode)
            ]
            structures.extend(scored)
            registered = register_job(root, job, target, mode, profile)
            conditions["updated_at"] = now()
            if mode == "exact" and profile.get("run_status") == "completed":
                off_row = next(
                    (row for row in profiles if row.get("pdb_id") == pdb_id and row.get("mode") == "off" and row.get("run_dir")),
                    None,
                )
                if off_row:
                    comparison = compare_exact_profiles(root, Path(off_row["run_dir"]), Path(job["run_dir"]), target)
                    profile["exact_coordinate_comparison"] = comparison
                    profile["exact_validation_pass"] = (
                        comparison.get("status") == "compared"
                        and comparison.get("all_atom_keys_match")
                        and comparison.get("max_abs_coordinate_error_angstrom") == 0
                        and comparison.get("all_prediction_artifacts_bitwise_identical")
                    )
                    conditions.setdefault("exact_validation", {})[pdb_id] = {
                        "pass": profile["exact_validation_pass"],
                        "max_abs_coordinate_error_angstrom": comparison.get("max_abs_coordinate_error_angstrom"),
                        "n_samples_compared": comparison.get("n_samples_compared"),
                        "all_prediction_artifacts_bitwise_identical": comparison.get("all_prediction_artifacts_bitwise_identical"),
                        "artifact_files_compared": comparison.get("artifact_files_compared"),
                    }
            batch.update({"conditions": conditions, "profile_summaries": profiles, "structures": structures})
            save_dashboard_data(root, batch, registered)
            print(
                "FINISH "
                + json.dumps(
                    {
                        "target": pdb_id,
                        "mode": mode,
                        "app_status": profile.get("app_status"),
                        "run_status": profile.get("run_status"),
                        "structures": f"{profile.get('n_structures_found')}/{profile.get('expected_structures')}",
                        "scored": profile.get("n_scored"),
                        "forward_s": profile.get("inference_seconds"),
                        "wall_s": profile.get("wall_seconds"),
                        "peak_gpu0_mib": profile.get("peak_memory_mib_by_gpu", {}).get("0"),
                        "job_id": profile.get("job_id"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if profile.get("run_status") != "completed" or profile.get("n_structures_found") != profile.get("expected_structures"):
                failed_target = pdb_id

    off = {(row.get("pdb_id"), row.get("engine")): row for row in profiles if row.get("mode") == "off"}
    for row in profiles:
        baseline = off.get((row.get("pdb_id"), row.get("engine")))
        base_forward = baseline.get("inference_seconds") if baseline else None
        forward = row.get("inference_seconds")
        base_wall = baseline.get("wall_seconds") if baseline else None
        wall = row.get("wall_seconds")
        row["inference_speedup_vs_off"] = base_forward / forward if base_forward and forward else None
        row["wall_speedup_vs_off"] = base_wall / wall if base_wall and wall else None
    conditions["status"] = "completed" if all(row.get("run_status") in {"completed", "skipped"} for row in profiles) else "partial"
    conditions["finished_at"] = now()
    batch.update({"conditions": conditions, "profile_summaries": profiles, "structures": structures})
    save_dashboard_data(root, batch, registered)
    print(f"BATCH_FINISH status={conditions['status']} profiles={len(profiles)} structures={len(structures)}", flush=True)


if __name__ == "__main__":
    main()
