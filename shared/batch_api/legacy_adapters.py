from __future__ import annotations

"""Batch adapters for the historical July images and remaining current profiles."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contract import BatchRequest, BatchRequestError
from .adapters import _rebase_openfold3_query, _rebase_protein_chain_paths


def _read_json(path: Path, item_id: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchRequestError(f"cannot read native input for {item_id!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchRequestError(f"native input for {item_id!r} must be a JSON object")
    return value


def _result(item_id: str, outputs: list[Path], expected: int, elapsed: float, error: str = "") -> dict[str, Any]:
    status = "completed" if len(outputs) == expected and not error else "failed"
    return {
        "id": item_id,
        "status": status,
        "outputs": [str(path.resolve()) for path in outputs],
        "expected_outputs": expected,
        "output_count": len(outputs),
        "elapsed_seconds": round(elapsed, 3),
        "error": error or ("" if status == "completed" else f"expected {expected} structures, found {len(outputs)}"),
    }


def _structures(root: Path, item_id: str) -> list[Path]:
    if not root.is_dir():
        return []
    files = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".cif", ".mmcif", ".pdb"}
    )
    matching = [path for path in files if item_id in path.name or item_id in path.parts]
    return matching


def _as_seeded_af3(item_path: Path, item_id: str, seeds: list[int], destination: Path) -> None:
    payload = _read_json(item_path, item_id)
    if not isinstance(payload.get("sequences"), list) or not payload["sequences"]:
        raise BatchRequestError(f"AlphaFast item {item_id!r} must be an AlphaFold 3 JSON complex")
    payload["name"] = item_id
    payload["modelSeeds"] = seeds
    for entity in payload["sequences"]:
        protein = entity.get("protein") if isinstance(entity, dict) else None
        if not isinstance(protein, dict):
            continue
        msa_path = str(protein.get("unpairedMsaPath") or "").strip()
        if msa_path:
            source = Path(msa_path)
            if not source.is_absolute():
                source = (item_path.parent / source).resolve()
            if source.is_file():
                protein["unpairedMsa"] = source.read_text(encoding="utf-8", errors="replace")
            protein.pop("unpairedMsaPath", None)
        paired_path = str(protein.get("pairedMsaPath") or "").strip()
        if paired_path:
            protein.pop("pairedMsaPath", None)
        protein["pairedMsa"] = ""
        protein.setdefault("templates", [])
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_alphafast(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.mode != "off":
        raise BatchRequestError("The historical AlphaFast image only supports mode='off'")
    allowed = {"model_dir", "num_recycles", "xla_flags"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported AlphaFast options: {sorted(unknown)}")
    staged = output_dir / ".mn-predict-alphafast-inputs"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    seeds = [request.seed + index for index in range(request.samples_per_item)]
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"AlphaFast item {item.item_id!r} must use JSON input")
        _as_seeded_af3(item.input_path, item.item_id, seeds, staged / f"{item.item_id}.json")
    model_dir = str(request.options.get("model_dir") or "/reference/alphafold3")
    command = [
        sys.executable, "/app/alphafold/run_alphafold.py",
        f"--input_dir={staged}", f"--model_dir={model_dir}", "--norun_data_pipeline",
        f"--output_dir={output_dir}", "--force_output_dir",
        f"--num_recycles={int(request.options.get('num_recycles', 10))}",
    ]
    environment = os.environ.copy()
    environment.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    environment.setdefault("XLA_CLIENT_MEM_FRACTION", "0.95")
    if request.options.get("xla_flags"):
        environment["XLA_FLAGS"] = str(request.options["xla_flags"])
    log_path = output_dir / "alphafast.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    rows = []
    for item in request.items:
        outputs: list[Path] = []
        for seed in seeds:
            outputs.extend(sorted((output_dir / item.item_id / f"seed-{seed}_sample-0").glob("*.cif")))
        error = f"AlphaFast exited with code {process.returncode}" if process.returncode else ""
        rows.append(_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items), error))
    return rows, {
        "return_code": process.returncode,
        "command": command,
        "log_path": str(log_path),
        "batch_execution": "alphafold3_input_directory_single_process",
        "sample_seeds": seeds,
    }


def _run_esmfold2_biohub(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.mode != "off":
        raise BatchRequestError("The historical Biohub ESMFold2 image only supports mode='off'")
    allowed = {"num_loops", "num_sampling_steps", "reference_root"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported Biohub ESMFold2 options: {sorted(unknown)}")
    msa_root = output_dir / ".mn-predict-biohub-msas"
    if msa_root.exists():
        shutil.rmtree(msa_root)
    msa_root.mkdir(parents=True)
    requests: list[dict[str, Any]] = []
    request_owner: dict[str, str] = {}
    for item in request.items:
        payload = _read_json(item.input_path, item.item_id)
        sequences = []
        for index, entity in enumerate(payload.get("sequences") or [], start=1):
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                raise BatchRequestError(f"Biohub ESMFold2 item {item.item_id!r} contains a non-protein chain")
            raw_ids = protein.get("id") or [chr(ord("A") + index - 1)]
            chain_id = str(raw_ids[0] if isinstance(raw_ids, list) else raw_ids)
            sequence = "".join(str(protein.get("sequence") or "").split()).upper()
            msa_text = str(protein.get("unpairedMsa") or "")
            msa_path = str(protein.get("unpairedMsaPath") or "").strip()
            if msa_path:
                source = Path(msa_path)
                if not source.is_absolute():
                    source = (item.input_path.parent / source).resolve()
                if source.is_file():
                    msa_text = source.read_text(encoding="utf-8", errors="replace")
            chain: dict[str, Any] = {"id": chain_id, "sequence": sequence}
            if msa_text:
                path = msa_root / f"{item.item_id}_{index:02d}.a3m"
                path.write_text(msa_text, encoding="utf-8")
                chain["msa_path"] = str(path)
            sequences.append(chain)
        for sample in range(request.samples_per_item):
            request_id = f"{item.item_id}__sample-{sample + 1}"
            requests.append({
                "request_id": request_id,
                "sequences": sequences,
                "num_diffusion_samples": 1,
                "seed": request.seed + sample,
                "num_loops": int(request.options.get("num_loops", 3)),
                "num_sampling_steps": int(request.options.get("num_sampling_steps", 32)),
            })
            request_owner[request_id] = item.item_id
    config_path = output_dir / ".mn-predict-biohub-config.json"
    config_path.write_text(json.dumps({
        "work_root": "/work",
        "reference_root": str(request.options.get("reference_root") or "/reference/biohub-esm"),
        "output_dir": str(output_dir),
        "device": "cuda",
        "num_loops": int(request.options.get("num_loops", 3)),
        "num_sampling_steps": int(request.options.get("num_sampling_steps", 32)),
        "seed": request.seed,
        "requests": requests,
    }, indent=2) + "\n", encoding="utf-8")
    runner = Path(__file__).with_name("esmfold2_biohub_runner.py")
    command = [sys.executable, str(runner), "--config", str(config_path)]
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_DISABLE_XET": "1",
        "ESMCFOLD_CCD_PATH": "/reference/biohub-esm/ESMFold2/ccd.pkl",
    })
    log_path = output_dir / "esmfold2_biohub.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    result_manifest = output_dir / "results.json"
    manifest = json.loads(result_manifest.read_text()) if result_manifest.is_file() else {}
    files_by_item: dict[str, list[Path]] = {item.item_id: [] for item in request.items}
    for row in manifest.get("results") or []:
        request_id = str(row.get("request_id") or "")
        owner = request_owner.get(request_id)
        if not owner:
            continue
        relative = str(row.get("complex_path") or "")
        if relative:
            files_by_item[owner].append((Path("/work") / relative).resolve())
    rows = []
    for item in request.items:
        error = f"Biohub ESMFold2 exited with code {process.returncode}" if process.returncode else ""
        rows.append(_result(item.item_id, files_by_item[item.item_id], request.samples_per_item, elapsed / len(request.items), error))
    return rows, {
        "return_code": process.returncode,
        "command": command,
        "log_path": str(log_path),
        "result_manifest": str(result_manifest) if result_manifest.is_file() else "",
        "batch_execution": "single_container_model_loaded_once_for_all_requests",
        "sample_seeds": [request.seed + index for index in range(request.samples_per_item)],
    }


def _run_openfold3_p2(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.mode != "off":
        raise BatchRequestError("The historical OpenFold3 P2 image only supports mode='off'")
    allowed = {"runner_yamls", "checkpoint", "use_msa_server", "num_recycles"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported historical OpenFold3 options: {sorted(unknown)}")
    runner_yamls = request.options.get("runner_yamls")
    if not isinstance(runner_yamls, dict):
        raise BatchRequestError("OpenFold3 P2 requires options.runner_yamls keyed by item ID")
    checkpoint = str(request.options.get("checkpoint") or "/reference/openfold3/of3-p2-155k.pt")
    rows = []
    commands = []
    return_codes = []
    batch_started = time.monotonic()
    for item in request.items:
        runner_yaml = str(runner_yamls.get(item.item_id) or "")
        if not runner_yaml:
            raise BatchRequestError(f"OpenFold3 P2 is missing runner YAML for {item.item_id!r}")
        query = _read_json(item.input_path, item.item_id)
        _rebase_openfold3_query(query, item.input_path, item.item_id, "OpenFold3 P2")
        staged_query = output_dir / f".{item.item_id}.openfold3-p2-query.json"
        staged_query.write_text(json.dumps(query, indent=2) + "\n", encoding="utf-8")
        item_output = output_dir / item.item_id
        item_output.mkdir(parents=True, exist_ok=True)
        command = [
            "run_openfold", "predict", "--runner-yaml", runner_yaml,
            "--query-json", str(staged_query), "--output-dir", str(item_output),
            "--use-msa-server", "true" if bool(request.options.get("use_msa_server")) else "false",
            "--num-diffusion-samples", str(request.samples_per_item),
            "--num-model-seeds", "1", "--inference-ckpt-path", checkpoint,
        ]
        commands.append(command)
        started = time.monotonic()
        process = subprocess.run(command, check=False)
        elapsed = time.monotonic() - started
        return_codes.append(process.returncode)
        outputs = _structures(item_output, item.item_id)
        error = f"OpenFold3 P2 exited with code {process.returncode}" if process.returncode else ""
        rows.append(_result(item.item_id, outputs, request.samples_per_item, elapsed, error))
    return rows, {
        "return_code": 0 if all(code == 0 for code in return_codes) else 1,
        "item_return_codes": return_codes,
        "commands": commands,
        "elapsed_seconds": round(time.monotonic() - batch_started, 3),
        "batch_execution": "single_container_serial_native_openfold3_calls",
    }


def _run_protenix_v2(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stock = request.engine == "protenix_v2_stock"
    if stock and request.mode != "off":
        raise BatchRequestError("The historical Protenix V2 image only supports mode='off'")
    allowed = {"model_name", "cycle", "step", "dtype", "use_msa", "use_template", "use_default_params"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported Protenix V2 options: {sorted(unknown)}")
    staged = output_dir / ".mn-predict-protenix-v2-inputs"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir()
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"Protenix V2 item {item.item_id!r} must use JSON input")
        payload = json.loads(item.input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            payload[0]["name"] = item.item_id
        elif isinstance(payload, dict):
            payload["name"] = item.item_id
        _rebase_protein_chain_paths(
            payload, item.input_path, item.item_id,
            marker="/input/protenix_v1/", engine="Protenix V2",
        )
        (staged / f"{item.item_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args = [
        "pred", "-i", str(staged), "-o", str(output_dir),
        "-s", str(request.seed), "-n", str(request.options.get("model_name") or "protenix_base_default_v2.0.0"),
        "-c", str(int(request.options.get("cycle", 10))),
        "-p", str(int(request.options.get("step", 200))),
        "-e", str(request.samples_per_item), "-d", str(request.options.get("dtype", "bf16")),
        "--use_msa", str(bool(request.options.get("use_msa", True))),
        "--use_template", str(bool(request.options.get("use_template", False))),
        "--use_default_params", str(bool(request.options.get("use_default_params", True))),
        "--need_atom_confidence", "True",
    ]
    environment = os.environ.copy()
    environment.update({
        "PROTENIX_ROOT_DIR": "/reference/protenix",
        "PROTENIX_DATA_ROOT_DIR": "/reference/protenix/release_data/ccd_cache",
        "TOOL_WEIGHTS_ROOT": "/reference/protenix/tool_weights",
        "MODEL_OPT_JIT_ROOT": "/cache/protenix_v2/jit",
        "HOME": "/cache/home",
        "XDG_CACHE_HOME": "/cache/xdg",
    })
    if stock:
        command = ["protenix", *args]
    else:
        environment["PROTENIX_OPT_MODE"] = request.mode
        environment["PROTENIX_OPT_N_GPU"] = str(request.gpu_count)
        command = ["/usr/local/bin/protenix-opt", "protenix", *args]
    log_path = output_dir / "protenix_v2.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    rows = []
    for item in request.items:
        outputs = _structures(output_dir, item.item_id)
        error = f"Protenix V2 exited with code {process.returncode}" if process.returncode else ""
        rows.append(_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items), error))
    return rows, {
        "return_code": process.returncode,
        "command": command,
        "log_path": str(log_path),
        "batch_execution": "native_input_directory_single_process",
    }


def _run_rf3_foundry_legacy(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.mode != "off":
        raise BatchRequestError("The historical Foundry RF3 image only supports mode='off'")
    allowed = {"checkpoint", "n_recycles", "num_steps", "raise_if_missing_msa_for_protein_of_length_n"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported historical RF3 options: {sorted(unknown)}")
    checkpoint = str(
        request.options.get("checkpoint")
        or "/reference/foundry/rf3_foundry_01_24_latest_remapped.ckpt"
    )
    rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    return_codes: list[int] = []
    started_batch = time.monotonic()
    for item in request.items:
        if item.input_path.suffix.lower() != ".json":
            raise BatchRequestError(f"Foundry RF3 item {item.item_id!r} must use JSON input")
        item_output = output_dir / item.item_id
        item_output.mkdir(parents=True, exist_ok=True)
        command = [
            "rf3", "fold",
            f"inputs={item.input_path}",
            f"ckpt_path={checkpoint}",
            f"out_dir={item_output}",
            f"n_recycles={int(request.options.get('n_recycles', 10))}",
            f"num_steps={int(request.options.get('num_steps', 50))}",
            f"diffusion_batch_size={request.samples_per_item}",
            f"seed={request.seed}",
            f"raise_if_missing_msa_for_protein_of_length_n={int(request.options.get('raise_if_missing_msa_for_protein_of_length_n', 10000))}",
        ]
        commands.append(command)
        started = time.monotonic()
        process = subprocess.run(command, check=False)
        elapsed = time.monotonic() - started
        return_codes.append(process.returncode)
        item_root = output_dir / item.item_id
        sample_dirs = sorted(
            path for path in item_root.rglob("*")
            if path.is_dir() and path.name.startswith("seed-") and "_sample-" in path.name
        ) if item_root.is_dir() else []
        outputs = [
            structure
            for sample_dir in sample_dirs
            for structure in sorted(sample_dir.glob("*.cif"))
        ]
        error = f"Foundry RF3 exited with code {process.returncode}" if process.returncode else ""
        rows.append(_result(item.item_id, outputs, request.samples_per_item, elapsed, error))
    return rows, {
        "return_code": 0 if all(code == 0 for code in return_codes) else 1,
        "item_return_codes": return_codes,
        "commands": commands,
        "elapsed_seconds": round(time.monotonic() - started_batch, 3),
        "batch_execution": "single_container_serial_native_rf3_fold_calls",
    }


def _run_pxdesign_v05(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.mode != "off":
        raise BatchRequestError("The historical PXDesign Protenix v0.5 image only supports mode='off'")
    allowed = {"model_name", "cycle", "step", "use_msa"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported PXDesign options: {sorted(unknown)}")
    staged = output_dir / ".mn-predict-pxdesign-inputs"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir()
    for item in request.items:
        payload = json.loads(item.input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            payload[0]["name"] = item.item_id
        elif isinstance(payload, dict):
            payload["name"] = item.item_id
        _rebase_protein_chain_paths(
            payload, item.input_path, item.item_id,
            marker="/input/protenix_v1/", engine="PXDesign Protenix v0.5",
        )
        (staged / f"{item.item_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    code = "\n".join([
        "from pathlib import Path",
        "import runner.batch_inference as batch_inference",
        "batch_inference.inference_configs.update({'load_checkpoint_dir': '/reference/pxdesign/release_data/checkpoint'})",
        "from runner.batch_inference import get_default_runner",
        "from runner.inference import infer_predict",
        f"input_dir = Path({str(staged)!r})",
        f"output_root = Path({str(output_dir)!r})",
        "for input_path in sorted(input_dir.glob('*.json')):",
        "    item_id = input_path.stem",
        f"    runner = get_default_runner(seeds=({request.seed},), n_cycle={int(request.options.get('cycle', 10))}, n_step={int(request.options.get('step', 200))}, n_sample={request.samples_per_item}, model_name={str(request.options.get('model_name') or 'protenix_base_default_v0.5.0')!r}, use_msa={bool(request.options.get('use_msa', True))!r})",
        "    runner.configs.dump_dir = str(output_root / item_id)",
        "    runner.configs.input_json_path = str(input_path)",
        "    runner.configs.need_atom_confidence = True",
        "    runner.dumper.need_atom_confidence = True",
        "    runner.dumper.base_dir = str(output_root / item_id)",
        "    infer_predict(runner, runner.configs)",
    ])
    command = [sys.executable, "-c", code]
    environment = os.environ.copy()
    environment.update({
        # The batch adapter mounts the shared reference tree read-only at
        # /reference, so PXDesign's pinned files are under /reference/pxdesign.
        "PROTENIX_ROOT_DIR": "/reference/pxdesign",
        "PROTENIX_DATA_ROOT_DIR": "/reference/pxdesign/release_data/ccd_cache",
        "TOOL_WEIGHTS_ROOT": "/reference/pxdesign/tool_weights",
        "HOME": "/cache/home",
        "XDG_CACHE_HOME": "/cache/xdg",
    })
    log_path = output_dir / "pxdesign_v05.pred.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    rows = []
    for item in request.items:
        outputs = _structures(output_dir / item.item_id, item.item_id)
        error = f"PXDesign Protenix v0.5 exited with code {process.returncode}" if process.returncode else ""
        rows.append(_result(item.item_id, outputs, request.samples_per_item, elapsed / len(request.items), error))
    return rows, {"return_code": process.returncode, "command": command, "log_path": str(log_path), "batch_execution": "single_container_serial_default_runner_calls"}


def _run_boltzgen_fold(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = {"config", "data_dir", "external_msa_dir", "external_msa_max_seqs", "moldir", "checkpoint", "recycling_steps", "sampling_steps"}
    unknown = set(request.options) - allowed
    if unknown:
        raise BatchRequestError(f"unsupported BoltzGen Fold options: {sorted(unknown)}")
    data_dir = Path(str(request.options.get("data_dir") or ""))
    msa_dir = Path(str(request.options.get("external_msa_dir") or ""))
    if not data_dir.is_dir():
        raise BatchRequestError("BoltzGen Fold requires options.data_dir with staged input PDB and NPZ files")
    if request.options.get("external_msa_dir") and not msa_dir.is_dir():
        raise BatchRequestError("BoltzGen Fold external MSA directory is missing")
    item_ids = [item.item_id for item in request.items]
    expected_item_ids = set(item_ids)
    design_pdb_ids = {
        path.stem for path in data_dir.glob("*.pdb") if path.is_file()
    }
    if design_pdb_ids != expected_item_ids:
        raise BatchRequestError(
            "BoltzGen Fold data_dir must contain exactly one staged PDB per manifest item; "
            f"expected {sorted(expected_item_ids)}, found {sorted(design_pdb_ids)}"
        )
    missing_design_masks = [
        item_id for item_id in item_ids if not (data_dir / f"{item_id}.npz").is_file()
    ]
    if missing_design_masks:
        raise BatchRequestError(
            "BoltzGen Fold data_dir is missing design masks for " + ", ".join(missing_design_masks)
        )
    if request.options.get("external_msa_dir"):
        missing_msa_dirs = [
            item_id for item_id in item_ids if not (msa_dir / item_id).is_dir()
        ]
        if missing_msa_dirs:
            raise BatchRequestError(
                "BoltzGen Fold external MSA directory is missing item folders for "
                + ", ".join(missing_msa_dirs)
            )
    config = str(request.options.get("config") or _boltzgen_gpu_profile())
    moldir = str(request.options.get("moldir") or "/reference/boltzgen-cache/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip")
    checkpoint = str(request.options.get("checkpoint") or "/reference/boltzgen-cache/models--boltzgen--boltzgen-1/snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltz2_conf_final.ckpt")
    os.environ["BOLTZGEN_WRITE_ALL_DIFFUSION_SAMPLES"] = "1"
    for item in request.items:
        if item.input_path.suffix.lower() not in {".yaml", ".yml"}:
            raise BatchRequestError(f"BoltzGen Fold item {item.item_id!r} must be a YAML spec")
    source_config = Path("/app/src/boltzgen/resources/config")
    if not source_config.is_dir():
        raise BatchRequestError("BoltzGen source config directory is missing from the image")
    fold_template = source_config / "fold_benchmark_pae.yaml"
    if not fold_template.is_file():
        raise BatchRequestError("BoltzGen fold_benchmark_pae.yaml is missing from the image")

    # BoltzGen's folding data loader consumes every PDB in data.design_dir.
    # Launching once per manifest item while pointing each launch at this full
    # directory repeats the complete batch N times. Match the application's
    # native workflow: one CLI invocation over the full staged directory.
    kit_run = output_dir / "kit_runs" / "batch"
    config_overlay = output_dir / "kit_config" / "batch"
    for path in (kit_run, config_overlay, data_dir / "refold_cif"):
        if path.exists():
            shutil.rmtree(path)
    config_overlay.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_config, config_overlay, dirs_exist_ok=True)
    shutil.copy2(fold_template, config_overlay / "fold.yaml")
    batch_spec = request.items[0]
    command = [
        "bash", "/kit/boltzgen/run.sh", "design", "--config", config,
        "--mode", request.mode, str(batch_spec.input_path), "--output", str(kit_run),
        "--seed", str(request.seed), "--num_designs", str(request.samples_per_item),
        "--diffusion_batch_size", str(request.samples_per_item), "--",
        "--steps", "folding", "--config_dir", str(config_overlay), "--config", "folding",
        f"data.design_dir={data_dir}", "data.cfg.suffix=.pdb", "data.cfg.target_id_regex=^(.+)$",
        "data.cfg.num_workers=1", f"data.cfg.moldir={moldir}",
        f"output={output_dir}", f"checkpoint={checkpoint}",
        f"recycling_steps={int(request.options.get('recycling_steps', 3))}",
        f"sampling_steps={int(request.options.get('sampling_steps', 200))}",
        f"diffusion_samples={request.samples_per_item}", "trainer.devices=1",
    ]
    if request.options.get("external_msa_dir"):
        command.extend([
            f"data.cfg.external_msa_dir={msa_dir}",
            f"data.cfg.external_msa_max_seqs={int(request.options.get('external_msa_max_seqs', 1024))}",
        ])
    refold_output_dir = data_dir / "refold_cif"
    if refold_output_dir.exists():
        shutil.rmtree(refold_output_dir)
    started_batch = time.monotonic()
    log_path = output_dir / "boltzgen-native.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.monotonic() - started_batch
    rows: list[dict[str, Any]] = []
    for item in request.items:
        sample_prefix = f"{item.item_id}_sample"
        output_files = sorted(
            (
                path.resolve()
                for path in refold_output_dir.glob(f"{sample_prefix}*.cif")
                if path.stem[len(sample_prefix):].isdigit()
            ),
            key=lambda path: int(path.stem[len(sample_prefix):]),
        )
        item_output = output_dir / item.item_id
        if item_output.exists():
            shutil.rmtree(item_output)
        item_output.mkdir(parents=True, exist_ok=True)
        copied = []
        for index, source in enumerate(output_files, start=1):
            destination = item_output / f"{item.item_id}_sample{index}.cif"
            shutil.copy2(source, destination)
            copied.append(destination)
        if process.returncode:
            error = f"BoltzGen exited with code {process.returncode}"
        elif len(copied) != request.samples_per_item:
            error = f"BoltzGen returned {len(copied)} of {request.samples_per_item} structures"
        else:
            error = ""
        rows.append(_result(
            item.item_id, copied, request.samples_per_item,
            elapsed / max(1, len(request.items)), error,
        ))
    result_code = 0 if process.returncode == 0 and all(row["status"] == "completed" for row in rows) else 1
    return rows, {
        "return_code": result_code,
        "native_call_return_codes": [process.returncode],
        "commands": [command],
        "log_path": str(log_path),
        "elapsed_seconds": round(elapsed, 3),
        "batch_execution": "single_container_single_native_design_call_for_full_input_directory",
        "native_design_call_count": 1,
        "input_item_count": len(request.items),
        "data_design_item_count": len(design_pdb_ids),
        "data_design_item_ids": sorted(design_pdb_ids),
        "native_design_call_item_ids": sorted(design_pdb_ids),
    }


def _boltzgen_gpu_profile() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "rtx4090"
    name = result.stdout.strip().lower() if result.returncode == 0 else ""
    return "rtx5090" if "5090" in name else "rtx4090"


def run_batch(request: BatchRequest, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.engine == "alphafast":
        return _run_alphafast(request, output_dir)
    if request.engine == "esmfold2_biohub":
        return _run_esmfold2_biohub(request, output_dir)
    if request.engine == "openfold3_p2":
        return _run_openfold3_p2(request, output_dir)
    if request.engine == "pxdesign_v05":
        return _run_pxdesign_v05(request, output_dir)
    if request.engine in {"protenix_v2_opt", "protenix_v2_stock"}:
        return _run_protenix_v2(request, output_dir)
    if request.engine == "boltzgen_fold_opt_msa":
        return _run_boltzgen_fold(request, output_dir)
    if request.engine == "rf3_foundry_legacy":
        return _run_rf3_foundry_legacy(request, output_dir)
    raise BatchRequestError(f"no historical/current batch adapter is registered for engine {request.engine!r}")
