from __future__ import annotations

"""Image-specific execution behind the shared ``mn-predict/v1`` contract."""

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .contract import BatchRequest, BatchRequestError


def capabilities(engine: str) -> dict[str, Any]:
    if engine in {"boltz2_legacy", "alphafast", "esmfold2_biohub", "openfold3_p2", "pxdesign_v05", "protenix_v2_stock", "rf3_foundry_legacy"}:
        native = {
            "boltz2_legacy": "stock_boltz_yaml_directory_single_worker",
            "alphafast": "alphafold3_input_directory_single_process",
            "esmfold2_biohub": "single_container_model_loaded_once_for_all_requests",
            "openfold3_p2": "single_container_serial_native_openfold3_calls",
            "pxdesign_v05": "single_container_serial_default_runner_calls",
            "protenix_v2_stock": "native_input_directory_single_process",
            "rf3_foundry_legacy": "single_container_serial_native_rf3_fold_calls",
        }[engine]
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off"],
            "batch_execution": native,
            "model_processes_per_batch": "one persistent process" if engine in {"alphafast", "esmfold2_biohub", "protenix_v2_stock"} else "one native call per input item",
            "container_invocations_per_batch": 1,
            "sample_axis": "engine samples or model seeds per input item",
            "output_identity": "input_item_id_and_sample_id",
        }
    if engine == "protenix_v2_opt":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "native_input_directory_single_process",
            "model_processes_per_batch": 1,
            "container_invocations_per_batch": 1,
            "sample_axis": "protenix_samples_per_input",
            "output_identity": "input_item_id_and_output_directory",
        }
    if engine == "boltzgen_fold_opt_msa":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "single_container_single_native_design_call_for_full_input_directory",
            "model_processes_per_batch": 1,
            "container_invocations_per_batch": 1,
            "sample_axis": "diffusion_samples_per_input_item",
            "output_identity": "input_item_id_and_diffusion_sample",
        }
    if engine == "rf3_foundry":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "native_json_item_list_single_model_process",
            "model_processes_per_batch": 1,
            "sample_axis": "diffusion_batch_size",
            "output_identity": "input_item_id_and_seed_sample_directory",
        }
    if engine == "boltz2":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "native_yaml_directory_single_persistent_worker",
            "model_processes_per_batch": 1,
            "sample_axis": "diffusion_samples_per_input",
            "output_identity": "input_item_id_directory",
        }
    if engine == "af3_torch":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "native_json_input_directory_with_per_item_forward_processes",
            "model_processes_per_batch": "one_per_input_item_per_seed",
            "container_invocations_per_batch": 1,
            "sample_axis": "alphafold3_num_diffusion_samples",
            "output_identity": "input_item_id_seed_sample_directory",
        }
    if engine in {"af3_jax", "alphafold3", "openfold3"}:
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "single_container_serial_inputs_with_per_item_runner_calls",
            "model_processes_per_batch": "one_runner_call_per_input_item",
            "container_invocations_per_batch": 1,
            "sample_axis": "alphafold3_num_diffusion_samples",
            "output_identity": "input_item_id_seed_sample_directory",
            "model_variants": [engine] if engine in {"alphafold3", "openfold3"} else ["alphafold3", "openfold3"],
        }
    if engine in {"esmfold2", "esmfold2_fast"}:
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "native_multi_item_json_single_model_process",
            "model_processes_per_batch": 1,
            "sample_axis": "diffusion_samples_per_input",
            "output_identity": "input_item_id_seed_sample_cif",
        }
    if engine == "chai1":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "native_items_manifest_single_container",
            "model_processes_per_batch": 1,
            "sample_axis": "diffusion_samples_per_input",
            "output_identity": "input_item_id_and_seed_sample_directory",
        }
    if engine == "colabfold_opt":
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off", "exact", "fast", "big"],
            "batch_execution": "optimized_input_directory_single_process_per_template_group",
            "model_processes_per_batch": "one per distinct target-only template; shared templates run together",
            "container_invocations_per_batch": 1,
            "sample_axis": "ColabFold model predictions per input",
            "output_identity": "input_item_id_and_output_directory",
        }
    if engine in {"atlasfold", "colabfold", "openfold3_ob0", "opendde_native", "protenix_v1"}:
        native = {
            "atlasfold": "native_multi_record_fasta",
            "colabfold": "native_input_directory",
            "openfold3_ob0": "single_container_per_item_cli_calls",
            "opendde_native": "native_json_item_list",
            "protenix_v1": "native_input_directory",
        }[engine]
        return {
            "api_version": "mn-predict/v1",
            "engine": engine,
            "modes": ["off"] if engine == "colabfold" else ["off", "exact", "fast", "big"],
            "batch_execution": native,
            "model_processes_per_batch": "native batch process" if engine != "openfold3_ob0" else "one runner call per input item",
            "container_invocations_per_batch": 1,
            "sample_axis": "engine_samples_per_input",
            "output_identity": "input_item_id_and_output_directory",
        }
    raise BatchRequestError(f"no batch adapter is registered for engine {engine!r}")


def run_batch(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.engine == "boltz2_legacy":
        return _run_boltz2(request, output_dir, legacy=True)
    if request.engine in {"alphafast", "esmfold2_biohub", "openfold3_p2", "pxdesign_v05", "protenix_v2_opt", "protenix_v2_stock", "boltzgen_fold_opt_msa"}:
        from . import legacy_adapters

        return legacy_adapters.run_batch(request, output_dir)
    if request.engine == "rf3_foundry_legacy":
        from . import legacy_adapters

        return legacy_adapters.run_batch(request, output_dir)
    if request.engine == "rf3_foundry":
        return _run_rf3(request, output_dir)
    if request.engine == "boltz2":
        return _run_boltz2(request, output_dir)
    if request.engine == "af3_torch":
        return _run_af3_torch(request, output_dir)
    if request.engine in {"af3_jax", "alphafold3", "openfold3"}:
        return _run_af3_jax(request, output_dir)
    if request.engine in {"esmfold2", "esmfold2_fast"}:
        return _run_esmfold2(request, output_dir)
    if request.engine == "chai1":
        return _run_chai1(request, output_dir)
    if request.engine == "atlasfold":
        return _run_atlasfold(request, output_dir)
    if request.engine in {"colabfold", "colabfold_opt"}:
        return _run_colabfold(request, output_dir, optimized=request.engine == "colabfold_opt")
    if request.engine == "openfold3_ob0":
        return _run_openfold3_ob0(request, output_dir)
    if request.engine == "opendde_native":
        return _run_opendde_native(request, output_dir)
    if request.engine == "protenix_v1":
        return _run_protenix_v1(request, output_dir)
    raise BatchRequestError(f"no batch adapter is registered for engine {request.engine!r}")


def _item_result(item_id: str, output_files: list[Path], expected: int, elapsed: float) -> dict[str, Any]:
    status = "completed" if len(output_files) == expected else "failed"
    return {
        "id": item_id,
        "status": status,
        "outputs": [str(path) for path in output_files],
        "expected_outputs": expected,
        "output_count": len(output_files),
        "elapsed_seconds": round(elapsed, 3),
        "error": "" if status == "completed" else f"expected {expected} structures, found {len(output_files)}",
    }


def _run_rf3(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {
        "checkpoint", "config", "n_recycles", "num_steps", "allow_partial",
        "raise_if_missing_msa_for_protein_of_length_n",
    }
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported RF3 options: {sorted(unknown)}")
    input_items: list[dict[str, Any]] = []
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"RF3 item {item.item_id!r} must use a JSON input file")
        try:
            payload = json.loads(item.input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchRequestError(f"cannot read RF3 input for {item.item_id!r}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("components"), list) or not payload["components"]:
            raise BatchRequestError(f"RF3 input for {item.item_id!r} must be one named complex object")
        payload = dict(payload)
        payload["name"] = item.item_id
        input_items.append(payload)

    combined_input = output_dir / ".mn-predict-rf3-inputs.json"
    combined_input.write_text(json.dumps(input_items, indent=1) + "\n", encoding="utf-8")
    checkpoint = str(request.options.get("checkpoint") or os.environ.get("ROSETTAFOLD3_OPT_CKPT") or "")
    if not checkpoint:
        raise BatchRequestError("RF3 requires options.checkpoint or ROSETTAFOLD3_OPT_CKPT")
    command = [
        "bash", "/kit/rosettafold3/run.sh", "pred",
        "--config", str(request.options.get("config") or "a100"),
        "--mode", request.mode,
        "--n_gpu", str(request.gpu_count),
        "--input", str(combined_input),
        "--out_dir", str(output_dir),
        "--ckpt", checkpoint,
        "--log", str(output_dir / "rf3.pred.log"),
    ]
    if bool(request.options.get("allow_partial", False)):
        command.append("--allow-partial")
    command.extend(
        [
            f"n_recycles={int(request.options.get('n_recycles', 10))}",
            f"num_steps={int(request.options.get('num_steps', 50))}",
            f"diffusion_batch_size={request.samples_per_item}",
            f"seed={request.seed}",
            f"raise_if_missing_msa_for_protein_of_length_n={int(request.options.get('raise_if_missing_msa_for_protein_of_length_n', 10000))}",
        ]
    )
    started = time.monotonic()
    process = subprocess.run(command, check=False)
    elapsed = time.monotonic() - started
    results: list[dict[str, Any]] = []
    for item in request.items:
        item_root = output_dir / item.item_id
        sample_dirs = sorted(
            path for path in item_root.rglob("*")
            if path.is_dir() and path.name.startswith("seed-") and "_sample-" in path.name
        ) if item_root.is_dir() else []
        output_files = []
        for sample_dir in sample_dirs:
            matches = sorted(sample_dir.glob("*.cif"))
            if len(matches) == 1:
                output_files.extend(matches)
        row = _item_result(item.item_id, output_files, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row["status"] = "failed"
            row["error"] = f"RF3 exited with code {process.returncode}; found {len(output_files)} of {request.samples_per_item} requested structures"
        results.append(row)
    return results, {
        "return_code": process.returncode,
        "command": command,
        "log_path": str(output_dir / "rf3.pred.log"),
        "batch_execution": capabilities("rf3_foundry")["batch_execution"],
    }


def _run_boltz2(
    request: BatchRequest,
    output_dir: Path,
    *,
    legacy: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {
        "cache", "recycling_steps", "sampling_steps", "write_full_pae",
        "output_format", "max_parallel_samples", "num_workers", "step_scale",
        "use_potentials", "use_msa_server", "max_msa_seqs", "method", "seed_count",
    }
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported Boltz2 options: {sorted(unknown)}")
    staged_inputs = output_dir / ".mn-predict-boltz2-inputs"
    if staged_inputs.exists():
        shutil.rmtree(staged_inputs)
    staged_inputs.mkdir(parents=True)
    for item in request.items:
        if item.input_path.suffix.lower() not in {".yaml", ".yml"}:
            raise BatchRequestError(f"Boltz2 item {item.item_id!r} must use a YAML input file")
        destination = staged_inputs / f"{item.item_id}.yaml"
        destination.symlink_to(item.input_path)
    cache_dir = str(request.options.get("cache") or os.environ.get("BOLTZ_CACHE") or "/models")
    seed_count = int(request.options.get("seed_count", 1))
    if not 1 <= seed_count <= 16 or request.seed + seed_count - 1 > 2**31 - 1:
        raise BatchRequestError("seed_count must be 1–16 and the generated seeds must not exceed 2,147,483,647")
    environment = os.environ.copy()
    environment["MN_BOLTZ2_MODE"] = request.mode
    environment["MN_BOLTZ2_N_GPU"] = str(request.gpu_count)
    started = time.monotonic()
    commands = []
    return_codes = []
    prediction_roots = []
    for seed_offset in range(seed_count):
        current_seed = request.seed + seed_offset
        seed_output_dir = output_dir / f"boltz_seed_{current_seed}"
        boltz_args = [
            "predict", str(staged_inputs), "--out_dir", str(seed_output_dir), "--cache", cache_dir,
            "--accelerator", "gpu", "--model", "boltz2",
            "--recycling_steps", str(int(request.options.get("recycling_steps", 10))),
            "--sampling_steps", str(int(request.options.get("sampling_steps", 200))),
            "--diffusion_samples", str(request.samples_per_item), "--seed", str(current_seed),
            "--output_format", str(request.options.get("output_format", "mmcif")),
        ]
        if request.options.get("step_scale") is not None:
            boltz_args.extend(("--step_scale", str(float(request.options["step_scale"]))))
        if request.options.get("max_msa_seqs") is not None:
            boltz_args.extend(("--max_msa_seqs", str(int(request.options["max_msa_seqs"]))))
        if request.options.get("method"):
            boltz_args.extend(("--method", str(request.options["method"])))
        if request.options.get("max_parallel_samples") is not None:
            boltz_args.extend(("--max_parallel_samples", str(int(request.options["max_parallel_samples"]))))
        if request.options.get("num_workers") is not None:
            boltz_args.extend(("--num_workers", str(int(request.options["num_workers"]))))
        if bool(request.options.get("write_full_pae", True)):
            boltz_args.append("--write_full_pae")
        if bool(request.options.get("use_potentials", False)):
            boltz_args.append("--use_potentials")
        if bool(request.options.get("use_msa_server", False)):
            boltz_args.append("--use_msa_server")
        command = (
            ["bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate boltz2 && exec " + shlex.join(["boltz", *boltz_args])]
            if legacy
            else [sys_executable(), "/usr/local/bin/mn-boltz2-entrypoint.py", *boltz_args]
        )
        commands.append(command)
        process = subprocess.run(command, env=environment, check=False)
        return_codes.append(process.returncode)
        input_name = staged_inputs.stem
        prediction_roots.append(seed_output_dir / f"boltz_results_{input_name}" / "predictions")
    elapsed = time.monotonic() - started
    results: list[dict[str, Any]] = []
    for item in request.items:
        outputs = []
        for root in prediction_roots:
            item_dir = root / item.item_id
            if item_dir.is_dir():
                outputs.extend([*item_dir.glob("*.cif"), *item_dir.glob("*.pdb")])
        outputs = sorted(outputs)
        expected_outputs = request.samples_per_item * seed_count
        row = _item_result(item.item_id, outputs, expected_outputs, elapsed / len(request.items))
        if any(return_code != 0 for return_code in return_codes):
            row["status"] = "failed"
            row["error"] = f"Boltz2 exited with code(s) {return_codes}; found {len(outputs)} of {expected_outputs} requested structures"
        results.append(row)
    return results, {
        "return_code": max(return_codes, default=0),
        "commands": commands,
        "prediction_roots": [str(path) for path in prediction_roots],
        "seeds": [request.seed + index for index in range(seed_count)],
        "batch_execution": capabilities("boltz2_legacy" if legacy else "boltz2")["batch_execution"],
    }


def _read_item_json_value(item_path: Path, item_id: str) -> Any:
    try:
        payload = json.loads(item_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchRequestError(f"cannot read native input for {item_id!r}: {exc}") from exc
    return payload


def _read_item_json(item_path: Path, item_id: str) -> dict[str, Any]:
    payload = _read_item_json_value(item_path, item_id)
    if not isinstance(payload, dict):
        raise BatchRequestError(f"native input for {item_id!r} must be a JSON object")
    return payload


def _batch_item_reference(path_text: str, item_path: Path, marker: str, item_id: str, label: str) -> str:
    source = Path(path_text)
    if source.is_absolute() and source.is_file():
        return str(source.resolve())
    if marker in path_text:
        relative = path_text.split(marker, 1)[1]
        resolved = (item_path.parent / relative).resolve()
    else:
        resolved = (item_path.parent / path_text).resolve()
    if not resolved.is_file():
        raise BatchRequestError(f"{label} for {item_id!r} is missing after batch path mapping: {resolved}")
    return str(resolved)


def _rebase_protein_chain_paths(
    payload: Any,
    item_path: Path,
    item_id: str,
    *,
    marker: str,
    engine: str,
) -> None:
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if not isinstance(row, dict):
            continue
        sequences = row.get("sequences")
        if not isinstance(sequences, list):
            continue
        for entity in sequences:
            protein = entity.get("proteinChain") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            for key in ("unpairedMsaPath", "pairedMsaPath", "templatesPath"):
                value = protein.get(key)
                if isinstance(value, str) and value.strip():
                    protein[key] = _batch_item_reference(
                        value, item_path, marker, item_id, f"{engine} {key}"
                    )


def _rebase_openfold3_query(query: dict[str, Any], item_path: Path, item_id: str, engine: str) -> None:
    queries = query.get("queries")
    if not isinstance(queries, dict):
        raise BatchRequestError(f"{engine} query for {item_id!r} must contain a queries object")
    for query_payload in queries.values():
        if not isinstance(query_payload, dict):
            raise BatchRequestError(f"{engine} query for {item_id!r} contains an invalid query")
        chains = query_payload.get("chains")
        if not isinstance(chains, list):
            raise BatchRequestError(f"{engine} query for {item_id!r} has no chains list")
        for chain in chains:
            if not isinstance(chain, dict):
                raise BatchRequestError(f"{engine} query for {item_id!r} contains an invalid chain")
            for msa_key in ("main_msa_file_paths", "paired_msa_file_paths"):
                msa_paths = chain.get(msa_key)
                if not msa_paths:
                    continue
                was_list = isinstance(msa_paths, list)
                if isinstance(msa_paths, str):
                    msa_paths = [msa_paths]
                if not isinstance(msa_paths, list):
                    raise BatchRequestError(f"{engine} {msa_key} for {item_id!r} must be a path or list")
                rebased = [
                    _batch_item_reference(
                        str(path), item_path, "/input/openfold3_native/", item_id, f"{engine} {msa_key}"
                    )
                    for path in msa_paths
                ]
                chain[msa_key] = rebased if was_list or len(rebased) != 1 else rebased[0]


def _gpu_profile(fallback: str = "rtx4090") -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback
    name = result.stdout.strip().lower() if result.returncode == 0 else ""
    if "5090" in name:
        return "rtx5090"
    if "4090" in name:
        return "rtx4090"
    return fallback


def _matching_structures(
    root: Path,
    item_id: str,
    *,
    exclude_path_parts: set[str] | frozenset[str] = frozenset(),
) -> list[Path]:
    all_files = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cif", ".mmcif", ".pdb"}
        and not (set(path.parts) & exclude_path_parts)
    ) if root.is_dir() else []
    matching = [path for path in all_files if item_id in path.name or item_id in path.parts]
    return matching or (all_files if len(all_files) and not item_id else [])


def _run_esmfold2(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "variant", "num_loops", "num_sampling_steps", "msa_max_depth", "backend"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported ESMFold2 options: {sorted(unknown)}")
    variant = str(request.options.get("variant") or ("fast" if request.engine == "esmfold2_fast" else "full_msa"))
    if request.engine == "esmfold2_fast" and variant != "fast":
        raise BatchRequestError("ESMFold2-Fast only accepts variant='fast'")
    if request.engine == "esmfold2" and variant not in {"full_msa", "full_nomsa"}:
        raise BatchRequestError("ESMFold2 variant must be 'full_msa' or 'full_nomsa'")
    items: list[dict[str, Any]] = []
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"ESMFold2 item {item.item_id!r} must use a native JSON input")
        payload = _read_item_json(item.input_path, item.item_id)
        if not isinstance(payload.get("sequences"), list):
            raise BatchRequestError(f"ESMFold2 input {item.item_id!r} must contain a sequences list")
        payload["id"] = item.item_id
        payload["seeds"] = [request.seed]
        payload.pop("name", None)
        for chain in payload["sequences"]:
            if isinstance(chain, dict) and isinstance(chain.get("msa"), str):
                msa_path = Path(chain["msa"])
                if not msa_path.is_absolute():
                    chain["msa"] = str((item.input_path.parent / msa_path).resolve())
        items.append(payload)
    staged_input = output_dir / ".mn-predict-esmfold2-inputs.json"
    staged_input.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    config = str(request.options.get("config") or _gpu_profile())
    command = [
        "/usr/local/bin/mn-esmfold2-runner", "pred",
        "--config", config,
        "--variant", variant,
        "--mode", request.mode,
        "--n_gpu", str(request.gpu_count),
        "--input", str(staged_input),
        "--out_dir", str(output_dir),
        "--seeds", str(request.seed),
        "--num_loops", str(int(request.options.get("num_loops", 10))),
        "--num_sampling_steps", str(int(request.options.get("num_sampling_steps", 200))),
        "--num_diffusion_samples", str(request.samples_per_item),
        "--msa_max_depth", str(int(request.options.get("msa_max_depth", 1024))),
        "--det", "1",
    ]
    if request.options.get("backend") is not None:
        command.extend(("--backend", str(request.options["backend"])))
    environment = os.environ.copy()
    environment.update({
        "HOME": "/cache/home",
        "XDG_CACHE_HOME": "/cache/xdg",
        "HF_HOME": "/cache/huggingface/esmfold2",
        "MODEL_OPT_JIT_ROOT": "/cache/esmfold2/jit",
        "MN_ESMFOLD2_MODE": request.mode,
        "MN_ESMFOLD2_N_GPU": str(request.gpu_count),
    })
    log_path = output_dir / "esmfold2.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    results = []
    for item in request.items:
        outputs = sorted((output_dir / "cif_all").glob(f"{item.item_id}__*.cif"))
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row.update(status="failed", error=f"ESMFold2 exited with code {process.returncode}; found {len(outputs)} structures")
        results.append(row)
    return results, {"return_code": process.returncode, "command": command, "log_path": str(log_path), "batch_execution": capabilities(request.engine)["batch_execution"]}


def _run_chai1(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "msa_depth", "num_trunk_recycles", "num_trunk_samples", "low_memory"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported Chai-1 options: {sorted(unknown)}")
    stage = output_dir / ".mn-predict-chai1"
    if stage.exists():
        shutil.rmtree(stage)
    fasta_dir = stage / "fasta"
    msa_dir = stage / "msas"
    fasta_dir.mkdir(parents=True)
    msa_dir.mkdir()
    items: list[dict[str, Any]] = []
    for item in request.items:
        payload = _read_item_json(item.input_path, item.item_id)
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or len(raw_items) != 1 or not isinstance(raw_items[0], dict):
            raise BatchRequestError(f"Chai-1 input {item.item_id!r} must contain exactly one item")
        raw = raw_items[0]
        fasta = Path(str(raw.get("fasta") or ""))
        if not fasta.is_absolute():
            fasta = (item.input_path.parent / fasta).resolve()
        if not fasta.is_file():
            raise BatchRequestError(f"Chai-1 FASTA is missing for {item.item_id!r}: {fasta}")
        fasta_target = fasta_dir / f"{item.item_id}.fasta"
        shutil.copy2(fasta, fasta_target)
        items.append({"id": item.item_id, "fasta": str(fasta_target), "seeds": [request.seed]})
        raw_msa_dir = payload.get("msa_dir")
        if raw_msa_dir:
            source_msa_dir = Path(str(raw_msa_dir))
            if not source_msa_dir.is_absolute():
                source_msa_dir = (item.input_path.parent / source_msa_dir).resolve()
            if not source_msa_dir.is_dir():
                raise BatchRequestError(f"Chai-1 MSA directory is missing for {item.item_id!r}")
            for source in sorted(source_msa_dir.glob("*.a3m")):
                target = msa_dir / f"{item.item_id}__{source.name}"
                shutil.copy2(source, target)
    staged_input = stage / "items.json"
    payload = {"items": items, "msa_dir": str(msa_dir)} if any(msa_dir.iterdir()) else {"items": items}
    staged_input.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = str(request.options.get("config") or _gpu_profile())
    command = [
        "/usr/local/bin/mn-chai1-runner", "pred",
        "--config", config,
        "--mode", request.mode,
        "--n_gpu", str(request.gpu_count),
        "--input", str(staged_input),
        "--out_dir", str(output_dir),
        "--tag", "chai1",
        "--seeds", str(request.seed),
        "--num-trunk-recycles", str(int(request.options.get("num_trunk_recycles", 3))),
        "--num-diffn-timesteps", "200",
        "--num-diffn-samples", str(request.samples_per_item),
        "--num-trunk-samples", str(int(request.options.get("num_trunk_samples", 1))),
        "--recycle-msa-subsample", "0",
        "--no-use-msa-server", "--no-use-templates-server",
        "--det", "1",
        "--msa-depth", str(int(request.options.get("msa_depth", 1024))),
    ]
    if bool(request.options.get("low_memory", True)):
        command.append("--low-memory")
    environment = os.environ.copy()
    environment.update({
        "CHAI_DOWNLOADS_DIR": "/reference/chai1",
        "MODEL_OPT_JIT_ROOT": "/cache/chai1/jit",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/chai1/inductor",
        "CUDA_CACHE_PATH": "/cache/chai1/cuda",
        "HOME": "/cache/home",
        "XDG_CACHE_HOME": "/cache/xdg",
    })
    log_path = output_dir / "chai1.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    results = []
    for item in request.items:
        outputs = _matching_structures(output_dir, item.item_id)
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row.update(status="failed", error=f"Chai-1 exited with code {process.returncode}; found {len(outputs)} structures")
        results.append(row)
    return results, {"return_code": process.returncode, "command": command, "log_path": str(log_path), "batch_execution": capabilities("chai1")["batch_execution"]}


def _run_atlasfold(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "num_recycles", "num_steps", "seed"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported AtlasFold options: {sorted(unknown)}")
    fasta = output_dir / ".mn-predict-atlasfold.fasta"
    records = []
    for item in request.items:
        if item.input_path.suffix.lower() not in {".fa", ".fasta", ".faa"}:
            raise BatchRequestError(f"AtlasFold item {item.item_id!r} must be a multimer FASTA record")
        text = item.input_path.read_text(encoding="utf-8").strip()
        if not text.startswith(">"):
            raise BatchRequestError(f"AtlasFold FASTA for {item.item_id!r} has no header")
        _, sequence = text.split("\n", 1)
        records.extend((f">{item.item_id}", "".join(sequence.split())))
    fasta.write_text("\n".join(records) + "\n", encoding="utf-8")
    config = str(request.options.get("config") or _gpu_profile())
    command = [
        "/usr/local/bin/mn-atlasfold-runner", "pred",
        "--config", config,
        "--mode", request.mode,
        "--det", "1",
        "--n_gpu", str(request.gpu_count),
        "--weights", "/reference/atlasfold",
        "--", "multimer",
        "--input-fasta", str(fasta),
        "--out-dir", str(output_dir),
        "--num-recycles", str(int(request.options.get("num_recycles", 10))),
        "--num-samples", str(request.samples_per_item),
        "--num-steps", str(int(request.options.get("num_steps", 200))),
        "--seed", str(int(request.options.get("seed", request.seed))),
        "--format", "cif", "--save-confidence-arrays", "--overwrite",
    ]
    environment = os.environ.copy()
    environment.update({"ATLASFOLD_WEIGHTS_DIR": "/reference/atlasfold", "MODEL_OPT_JIT_ROOT": "/cache/atlasfold/jit", "HOME": "/cache/home", "XDG_CACHE_HOME": "/cache/xdg"})
    if request.mode != "off":
        # The pinned 4090 profile's own FoldBench disables this one BF16 lever:
        # its inputs are FP32, so it refuses every call and the kit exits 3.
        environment["MODEL_OPT_LEVERS_OFF"] = "ln_bf16"
    log_path = output_dir / "atlasfold.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    results = []
    for item in request.items:
        # AtlasFold also writes a ranked_model.cif beside the requested samples.
        # FoldBench scores the explicit diffusion samples only.
        outputs = [
            path for path in _matching_structures(output_dir, item.item_id)
            if "_sample-" in path.name
        ]
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row.update(status="failed", error=f"AtlasFold exited with code {process.returncode}; found {len(outputs)} structures")
        results.append(row)
    return results, {
        "return_code": process.returncode,
        "command": command,
        "log_path": str(log_path),
        "runtime_overrides": {"MODEL_OPT_LEVERS_OFF": "ln_bf16"} if request.mode != "off" else {},
        "batch_execution": capabilities("atlasfold")["batch_execution"],
    }


def _run_openfold3_ob0(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "runner_yamls", "checkpoint", "num_recycles", "use_msa_server", "use_templates"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported OpenFold3 OB0 options: {sorted(unknown)}")
    runner_yamls = request.options.get("runner_yamls")
    if not isinstance(runner_yamls, dict):
        raise BatchRequestError("OpenFold3 OB0 requires options.runner_yamls keyed by item id")
    config = str(request.options.get("config") or _gpu_profile())
    checkpoint = str(request.options.get("checkpoint") or "/reference/openfold3/of3-ob-2025-06-30-174k.pt")
    results = []
    commands = []
    return_codes = []
    started_batch = time.monotonic()
    for item in request.items:
        runner_yaml = str(runner_yamls.get(item.item_id) or "")
        if not runner_yaml:
            raise BatchRequestError(f"OpenFold3 OB0 is missing runner YAML for {item.item_id!r}")
        item_output = output_dir / item.item_id
        item_output.mkdir(parents=True, exist_ok=True)
        query = _read_item_json(item.input_path, item.item_id)
        _rebase_openfold3_query(query, item.input_path, item.item_id, "OpenFold3 OB0")
        staged_query = output_dir / f".{item.item_id}.openfold3-query.json"
        staged_query.write_text(json.dumps(query, indent=2) + "\n", encoding="utf-8")
        command = [
            "bash", "/kit/openfold3_ob0/run.sh", "pred",
            "--config", config, "--mode", request.mode,
            "--n_gpu", str(request.gpu_count), "--det", "1", "--ckpt", checkpoint,
            "--runner-yaml", runner_yaml, "--query-json", str(staged_query),
            "--output-dir", str(item_output),
            "--num-diffusion-samples", str(request.samples_per_item),
            "--use-msa-server", "true" if bool(request.options.get("use_msa_server")) else "false",
            "--use-templates", "true" if bool(request.options.get("use_templates")) else "false",
        ]
        commands.append(command)
        started = time.monotonic()
        process = subprocess.run(command, check=False)
        elapsed = time.monotonic() - started
        return_codes.append(process.returncode)
        outputs = _matching_structures(item_output, item.item_id)
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed)
        if process.returncode != 0:
            row.update(status="failed", error=f"OpenFold3 OB0 exited with code {process.returncode}; found {len(outputs)} structures")
        results.append(row)
    return results, {"return_code": max(return_codes, default=0), "commands": commands, "batch_execution": capabilities("openfold3_ob0")["batch_execution"], "elapsed_seconds": round(time.monotonic() - started_batch, 3)}


def _run_opendde_native(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "model_name", "use_msa", "use_template", "cycle", "step", "dtype", "allow_partial"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported OpenDDE options: {sorted(unknown)}")
    items = []
    for item in request.items:
        payload = _read_item_json_value(item.input_path, item.item_id)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
            rows = payload["items"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise BatchRequestError(f"OpenDDE input for {item.item_id!r} must be a JSON array or object")
        _rebase_protein_chain_paths(
            payload, item.input_path, item.item_id,
            marker="/input/opendde_native/", engine="OpenDDE",
        )
        for row in rows:
            if not isinstance(row, dict):
                raise BatchRequestError(f"OpenDDE input for {item.item_id!r} contains a non-object item")
            copy = dict(row)
            copy["name"] = item.item_id
            items.append(copy)
    combined = output_dir / ".mn-predict-opendde-inputs.json"
    combined.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    config = str(request.options.get("config") or _gpu_profile())
    command = [
        "bash", "/kit/opendde/run.sh", "pred",
        "--config", config, "--mode", request.mode,
        "--n_gpu", str(request.gpu_count), "--det", "1",
        "-i", str(combined), "-o", str(output_dir),
        "--seeds", str(request.seed), "--cycle", str(int(request.options.get("cycle", 10))),
        "--step", str(int(request.options.get("step", 200))), "--sample", str(request.samples_per_item),
        "--dtype", str(request.options.get("dtype", "bf16")),
        "--model_name", str(request.options.get("model_name", "opendde_v1")),
        "--use_msa", "true" if bool(request.options.get("use_msa", True)) else "false",
        "--use_template", "true" if bool(request.options.get("use_template", False)) else "false",
        "--need_atom_confidence", "true",
    ]
    if request.mode != "off" or bool(request.options.get("allow_partial", False)):
        command.append("--allow-partial")
    environment = os.environ.copy()
    environment.update({"OPENDDE_ROOT_DIR": "/reference/opendde", "MODEL_OPT_JIT_ROOT": "/cache/opendde/jit", "HOME": "/cache/home", "XDG_CACHE_HOME": "/cache/xdg"})
    log_path = output_dir / "opendde.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    results = []
    for item in request.items:
        # Distributed row-pair mode stages a duplicate rank-local copy before
        # gathering the final samples into the public output tree.
        outputs = _matching_structures(
            output_dir,
            item.item_id,
            exclude_path_parts={".rowpair"},
        )
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row.update(status="failed", error=f"OpenDDE exited with code {process.returncode}; found {len(outputs)} structures")
        results.append(row)
    return results, {"return_code": process.returncode, "command": command, "log_path": str(log_path), "batch_execution": capabilities("opendde_native")["batch_execution"]}


def _run_protenix_v1(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "cycle", "step", "dtype", "model_name", "use_msa", "use_template"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported Protenix V1 options: {sorted(unknown)}")
    staged = output_dir / ".mn-predict-protenix-v1-inputs"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir()
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"Protenix V1 item {item.item_id!r} must use JSON input")
        payload = _read_item_json_value(item.input_path, item.item_id)
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            payload[0]["name"] = item.item_id
        elif isinstance(payload, dict):
            payload["name"] = item.item_id
        _rebase_protein_chain_paths(
            payload, item.input_path, item.item_id,
            marker="/input/protenix_v1/", engine="Protenix V1",
        )
        (staged / f"{item.item_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = str(request.options.get("config") or _gpu_profile())
    command = [
        "bash", "/kit/protenix_v1/run.sh", "pred",
        "--config", config, "--mode", request.mode,
        "--n_gpu", str(request.gpu_count), "--det", "1",
        "--input", str(staged), "--out_dir", str(output_dir),
        "--seeds", str(request.seed), "--cycle", str(int(request.options.get("cycle", 10))),
        "--step", str(int(request.options.get("step", 200))), "--sample", str(request.samples_per_item),
        "--dtype", str(request.options.get("dtype", "bf16")),
        "--model_name", str(request.options.get("model_name", "protenix_base_default_v1.0.0")),
        "--use_msa", "true" if bool(request.options.get("use_msa", True)) else "false",
        "--use_template", "true" if bool(request.options.get("use_template", False)) else "false",
        "--need_atom_confidence", "true",
    ]
    if request.mode != "off":
        command.append("--allow-partial")
    environment = os.environ.copy()
    environment.update({"PROTENIX_ROOT_DIR": "/reference/protenix", "MODEL_OPT_JIT_ROOT": "/cache/protenix_v1/jit", "HOME": "/cache/home", "XDG_CACHE_HOME": "/cache/xdg"})
    log_path = output_dir / "protenix_v1.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    results = []
    for item in request.items:
        outputs = _matching_structures(output_dir, item.item_id)
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row.update(status="failed", error=f"Protenix V1 exited with code {process.returncode}; found {len(outputs)} structures")
        results.append(row)
    return results, {"return_code": process.returncode, "command": command, "log_path": str(log_path), "batch_execution": capabilities("protenix_v1")["batch_execution"]}


def _run_colabfold(
    request: BatchRequest,
    output_dir: Path,
    *,
    optimized: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not optimized and request.mode != "off":
        raise BatchRequestError("The stock ColabFold image only supports mode='off'")
    allowed_options = {"model_type", "msa_mode", "num_recycle", "template_path"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported ColabFold options: {sorted(unknown)}")
    staged = output_dir / ".mn-predict-colabfold-inputs"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir()
    grouped: dict[str, dict[str, Any]] = {}
    global_template = str(request.options.get("template_path") or "").strip()
    work_root = Path("/work").resolve()
    for item in request.items:
        if item.input_path.suffix.lower() not in {".a3m", ".fasta", ".fa", ".faa"}:
            raise BatchRequestError(f"ColabFold item {item.item_id!r} must be A3M or FASTA")
        template_source = item.template_path or (Path(global_template).resolve() if global_template else None)
        template_sha256 = ""
        if template_source is not None:
            source = Path(template_source).resolve()
            if not source.is_relative_to(work_root):
                raise BatchRequestError("ColabFold template paths must point inside the mounted /work directory")
            if not source.is_file() or source.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
                raise BatchRequestError(f"ColabFold target-only template is missing or unsupported: {source}")
            import hashlib

            template_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        group_key = template_sha256 or "no-template"
        group = grouped.setdefault(group_key, {
            "key": group_key,
            "items": [],
            "template_source": str(template_source) if template_source is not None else "",
            "template_sha256": template_sha256,
            "input_dir": staged / f"group-{group_key}",
        })
        group["items"].append(item)
        group["input_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.input_path, group["input_dir"] / f"{item.item_id}{item.input_path.suffix.lower()}")

    group_output_root = output_dir / ".mn-predict-colabfold-group-output"
    if group_output_root.exists():
        shutil.rmtree(group_output_root)
    group_output_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({"HOME": "/cache/home", "XDG_CACHE_HOME": "/cache/xdg"})
    if optimized:
        environment["COLABFOLD_OPT_DATA_DIR"] = "/cache"
        environment["COLABFOLD_OPT"] = request.mode
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        detected_gpu = gpu_query.stdout.strip().splitlines()[0] if gpu_query.returncode == 0 and gpu_query.stdout.strip() else ""
        if detected_gpu:
            environment["MODEL_OPT_TARGET_GPU"] = detected_gpu
        if request.mode == "big":
            environment["COLABFOLD_OPT_N_GPU"] = str(request.gpu_count)
        else:
            environment.pop("COLABFOLD_OPT_N_GPU", None)
    else:
        environment.pop("COLABFOLD_OPT", None)
        environment.pop("COLABFOLD_OPT_N_GPU", None)

    started = time.monotonic()
    item_outputs: dict[str, list[Path]] = {item.item_id: [] for item in request.items}
    item_return_codes: dict[str, int] = {}
    commands: list[list[str]] = []
    log_paths: list[str] = []
    group_metadata: list[dict[str, Any]] = []
    item_elapsed: dict[str, float] = {}
    overall_return_code = 0
    for group in grouped.values():
        group_items = group["items"]
        group_output = group_output_root / group["key"]
        group_output.mkdir(parents=True, exist_ok=True)
        template_dir: Path | None = None
        template_pdb_id = ""
        if group["template_source"]:
            template_dir = group["input_dir"] / "target-only-templates"
            template_dir.mkdir()
            source = Path(group["template_source"])
            # AlphaFold's custom-template parser derives each template hit as
            # ``<pdb filename>_<chain>`` and requires the first token to be a
            # four-character PDB ID. The frozen input filenames are SHA-256
            # hashes, which produce invalid hit names such as
            # ``636f..._A``. Give the single template in this SHA-group a
            # deterministic valid PDB ID while leaving its bytes and chain IDs
            # untouched. Each distinct template is run in its own group, so
            # the short ID cannot collide within a ColabFold invocation.
            template_pdb_id = f"0{group['template_sha256'][:3]}"
            template_name = f"{template_pdb_id}{source.suffix.lower()}"
            shutil.copy2(source, template_dir / template_name)
        if optimized:
            command = ["bash", "run.sh", "pred", "--config", "a100", "--mode", request.mode]
            if request.mode == "big":
                command.extend(("--n_gpu", str(request.gpu_count)))
            command.extend((
                str(group["input_dir"]),
                str(group_output),
                "--model-type", str(request.options.get("model_type", "alphafold2_multimer_v3")),
                "--num-models", str(request.samples_per_item),
                "--num-seeds", "1",
                "--random-seed", str(request.seed),
                "--num-ensemble", "1",
                "--data", "/cache",
                "--msa-mode", str(request.options.get("msa_mode", "single_sequence")),
            ))
        else:
            command = [
                "colabfold_batch", str(group["input_dir"]), str(group_output),
                "--data", "/cache",
                "--model-type", str(request.options.get("model_type", "alphafold2_multimer_v3")),
                "--msa-mode", str(request.options.get("msa_mode", "single_sequence")),
                "--num-models", str(request.samples_per_item),
                "--random-seed", str(request.seed),
                "--num-seeds", "1",
            ]
        if template_dir is not None:
            command.extend(("--templates", "--custom-template-path", str(template_dir)))
        if request.options.get("num_recycle") is not None:
            command.extend(("--num-recycle", str(int(request.options["num_recycle"]))))
        log_path = output_dir / f"colabfold-{group['key']}.pred.log"
        log_paths.append(str(log_path))
        commands.append(command)
        group_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd="/kit/colabfold" if optimized else None,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        group_elapsed = time.monotonic() - group_started
        if process.returncode != 0 and overall_return_code == 0:
            overall_return_code = process.returncode
        group_item_ids = []
        for item in group_items:
            item_return_codes[item.item_id] = int(process.returncode)
            item_elapsed[item.item_id] = group_elapsed / max(1, len(group_items))
            group_item_ids.append(item.item_id)
            outputs = sorted(
                path for path in group_output.glob(f"{item.item_id}*")
                if path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
            )
            for source in outputs:
                destination = output_dir / source.name
                if destination.exists():
                    destination.unlink()
                shutil.move(str(source), str(destination))
                item_outputs[item.item_id].append(destination)
        group_metadata.append({
            "template_sha256": group["template_sha256"],
            "template_path": group["template_source"],
            "item_ids": group_item_ids,
            "return_code": int(process.returncode),
            "elapsed_seconds": round(group_elapsed, 3),
            "log_path": str(log_path),
            "template_pdb_id": template_pdb_id,
        })

    elapsed = time.monotonic() - started
    results = []
    for item in request.items:
        row = _item_result(item.item_id, sorted(item_outputs[item.item_id]), request.samples_per_item, item_elapsed.get(item.item_id, 0.0))
        if item_return_codes.get(item.item_id, 1) != 0:
            row.update(status="failed", error=f"ColabFold exited with code {item_return_codes[item.item_id]}; found {len(item_outputs[item.item_id])} structures")
        results.append(row)
    return results, {
        "return_code": overall_return_code,
        "item_return_codes": item_return_codes,
        "commands": commands,
        "log_paths": log_paths,
        "elapsed_seconds": round(elapsed, 3),
        "batch_execution": capabilities("colabfold_opt" if optimized else "colabfold")["batch_execution"],
        "template_path": global_template,
        "templates_enabled": any(group["template_source"] for group in grouped.values()),
        "template_groups": group_metadata,
    }


def _run_af3_torch(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_options = {"config", "num_recycles", "diffusion_steps", "allow_partial"}
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported AF3-Torch options: {sorted(unknown)}")
    staged_inputs = output_dir / ".mn-predict-af3-torch-inputs"
    if staged_inputs.exists():
        shutil.rmtree(staged_inputs)
    staged_inputs.mkdir(parents=True)
    template_files_staged = 0
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"AF3-Torch item {item.item_id!r} must use a JSON input file")
        try:
            payload = json.loads(item.input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchRequestError(f"cannot read AF3-Torch input for {item.item_id!r}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), list) or not payload["sequences"]:
            raise BatchRequestError(f"AF3-Torch input for {item.item_id!r} must be one AlphaFold 3 complex object")
        payload = dict(payload)
        payload["name"] = item.item_id
        payload["modelSeeds"] = [request.seed]
        for entity_index, entity in enumerate(payload["sequences"]):
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            templates = protein.get("templates") or []
            if not isinstance(templates, list):
                raise BatchRequestError(f"AF3-Torch templates for {item.item_id!r} must be a list")
            for template_index, template in enumerate(templates):
                if not isinstance(template, dict) or not str(template.get("mmcifPath") or "").strip():
                    continue
                raw_path = Path(str(template["mmcifPath"]).replace("\\", "/"))
                source = raw_path.resolve() if raw_path.is_absolute() else (item.input_path.parent / raw_path).resolve()
                work_root = Path("/work").resolve()
                if not source.is_relative_to(work_root):
                    raise BatchRequestError("AF3-Torch template paths must stay inside the mounted /work directory")
                if not source.is_file() or source.suffix.lower() not in {".cif", ".mmcif"}:
                    raise BatchRequestError(f"AF3-Torch template is missing or unsupported: {source}")
                # AF3 template indexes restart for every protein entity. Keep
                # the entity index in the staged filename so two target
                # chains with a template each cannot overwrite one another.
                relative = (
                    Path("templates")
                    / item.item_id
                    / f"template-{entity_index}-{template_index}{source.suffix.lower()}"
                )
                destination = staged_inputs / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                template["mmcifPath"] = relative.as_posix()
                template_files_staged += 1
        destination = staged_inputs / f"{item.item_id}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    command = [
        "bash", "/kit/af3_torch/run.sh", "pred",
        "--config", str(request.options.get("config", "h100")),
        "--mode", request.mode,
        "--n_gpu", str(request.gpu_count),
        "--input_dir", str(staged_inputs),
        "--output_dir", str(output_dir),
        "--norun_data_pipeline",
        "--num_recycles", str(int(request.options.get("num_recycles", 10))),
        "--num_diffusion_samples", str(request.samples_per_item),
    ]
    if request.options.get("diffusion_steps") is not None:
        command.extend(("--diffusion_steps", str(int(request.options["diffusion_steps"]))))
    if bool(request.options.get("allow_partial", False)):
        command.append("--allow-partial")
    log_path = output_dir / "af3_torch.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    results: list[dict[str, Any]] = []
    for item in request.items:
        item_root = output_dir / item.item_id
        sample_dirs = sorted(
            path for path in item_root.iterdir()
            if path.is_dir() and path.name.startswith("seed-") and "_sample-" in path.name
        ) if item_root.is_dir() else []
        outputs = [
            model
            for sample_dir in sample_dirs
            for model in sorted(sample_dir.glob("*_model.cif"))
        ]
        row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items))
        if process.returncode != 0:
            row["status"] = "failed"
            row["error"] = (
                f"AF3-Torch exited with code {process.returncode}; "
                f"found {len(outputs)} of {request.samples_per_item} requested structures"
            )
        results.append(row)
    return results, {
        "return_code": process.returncode,
        "command": command,
        "log_path": str(log_path),
        "template_files_staged": template_files_staged,
        "batch_execution": capabilities("af3_torch")["batch_execution"],
    }


def _run_af3_jax(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run each independent AF3-JAX input inside one target-batch container."""
    allowed_options = {
        "model", "model_dir", "runner", "num_recycles", "flash_attention_implementation", "xla_flags",
    }
    unknown = set(request.options) - allowed_options
    if unknown:
        raise BatchRequestError(f"unsupported AF3-JAX options: {sorted(unknown)}")
    model = str(request.options.get("model") or "alphafold3")
    if model not in {"alphafold3", "openfold3"}:
        raise BatchRequestError("AF3-JAX model must be 'alphafold3' or 'openfold3'")

    staged_inputs = output_dir / ".mn-predict-af3-jax-inputs"
    if staged_inputs.exists():
        shutil.rmtree(staged_inputs)
    staged_inputs.mkdir(parents=True)
    runner = str(request.options.get("runner") or "/usr/local/bin/cofolding-af3-runner.py")
    if not (runner == "/usr/local/bin/cofolding-af3-runner.py" or runner.startswith("/work/")):
        raise BatchRequestError("AF3-JAX runner must be the pinned image runner or a staged /work override")
    environment = os.environ.copy()
    environment["MN_COFOLDING_AF3_OPT_MODE"] = request.mode
    environment["MN_COFOLDING_AF3_N_GPU"] = str(request.gpu_count)
    environment.setdefault("HOME", "/cache/home")
    environment.setdefault("XDG_CACHE_HOME", "/cache/xdg")
    environment.setdefault("AF3_WEIGHTS_DIR", "/cache/alphafold3/weights")
    environment.setdefault("AF3_CCD_CACHE_DIR", "/cache/alphafold3/ccd")
    environment.setdefault("AF3_MSA_CACHE_DIR", "/msa_cache/mn-cofolding/alphafold3")
    environment.setdefault("AF3_SHARED_MSA_CACHE_DIR", "/msa_cache")
    if request.options.get("xla_flags"):
        environment["XLA_FLAGS"] = str(request.options["xla_flags"])

    log_path = output_dir / "af3_jax.pred.log"
    results: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    return_codes: list[int] = []
    batch_started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for item in request.items:
            if item.input_path.suffix.lower() != ".json":
                raise BatchRequestError(f"AF3-JAX item {item.item_id!r} must use a JSON input file")
            try:
                payload = json.loads(item.input_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BatchRequestError(f"cannot read AF3-JAX input for {item.item_id!r}: {exc}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), list) or not payload["sequences"]:
                raise BatchRequestError(f"AF3-JAX input for {item.item_id!r} must be one AlphaFold 3 JSON complex")
            payload = dict(payload)
            payload["name"] = item.item_id
            payload["modelSeeds"] = [request.seed]
            staged_input = staged_inputs / f"{item.item_id}.json"
            staged_input.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            item_output = output_dir / item.item_id
            item_output.mkdir(parents=True, exist_ok=True)
            command = [
                sys_executable(), runner,
                f"--json_path={staged_input}",
                f"--model={model}",
                "--norun_data_pipeline",
                f"--output_dir={item_output}",
                "--force_output_dir",
                "--cache_dir=/cache/alphafold3",
                f"--num_recycles={int(request.options.get('num_recycles', 10))}",
                f"--num_diffusion_samples={request.samples_per_item}",
                f"--flash_attention_implementation={request.options.get('flash_attention_implementation', 'xla')}",
            ]
            model_dir = request.options.get("model_dir")
            if model_dir:
                command.append(f"--model_dir={model_dir}")
            commands.append(command)
            started = time.monotonic()
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            elapsed = time.monotonic() - started
            return_codes.append(process.returncode)
            outputs = sorted(
                path
                for path in item_output.rglob("*.cif")
                if path.parent.name.startswith("seed-") and "_sample-" in path.parent.name
            )
            row = _item_result(item.item_id, outputs, request.samples_per_item, elapsed)
            if process.returncode != 0:
                row["status"] = "failed"
                row["error"] = (
                    f"AF3-JAX runner exited with code {process.returncode}; "
                    f"found {len(outputs)} of {request.samples_per_item} requested structures"
                )
            results.append(row)
    return results, {
        "return_code": 0 if all(code == 0 for code in return_codes) else 1,
        "item_return_codes": return_codes,
        "commands": commands,
        "log_path": str(log_path),
        "elapsed_seconds": round(time.monotonic() - batch_started, 3),
        "batch_execution": capabilities(request.engine)["batch_execution"],
    }


def sys_executable() -> str:
    import sys

    return sys.executable
