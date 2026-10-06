#!/usr/bin/env python3
"""Build the Protenix optimization playbook report from raw attempt records."""

from __future__ import annotations

import json
import hashlib
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS = Path(os.environ.get(
    "PROTENIX_BENCHMARK_RESULTS",
    "/mnt/data/RESULTS/protenix-optimization-playbook-20261001",
)).resolve()
HERE = Path(__file__).resolve().parent
REPORT = HERE / "optimization-playbook-sweep-2026-10-01.md"
SNAPSHOT = HERE / "optimization-playbook-sweep-2026-10-01.json"
TARGETS = [
    ("CHRNA7", 502), ("SLC26A8", 970), ("NUP155", 1391), ("LCT", 1927),
    ("EP300", 2414), ("CEP350", 3117), ("DMD", 3685), ("PRKDC", 4128),
]
MODES = ("off", "fast", "big")
FAILURES = {
    "oom", "timeout", "kernel_refusal", "invalid_input", "error",
    "missing_structure", "invalid_output", "partial_output",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records() -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted((RESULTS / "records").glob("*.json"))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_pass(record: dict[str, Any]) -> bool:
    outputs = record.get("output_validation") or []
    return record.get("status") == "pass" and bool(outputs) and all(
        output.get("parse_exit_code") == 0 and output.get("confidence_json_valid") is True
        for output in outputs
    )


def mode_records(records: list[dict[str, Any]], target: str, mode: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in records
            if row.get("target") == target and row.get("mode") == mode
            and (target == "CHRNA7" or "-capacity" in str(row.get("run_id", "")))
        ],
        key=lambda row: str(row.get("ended_at") or row.get("started_at") or ""),
    )


def selected_attempt(records: list[dict[str, Any]], target: str, mode: str) -> dict[str, Any] | None:
    rows = mode_records(records, target, mode)
    passed = [row for row in rows if is_valid_pass(row)]
    if passed:
        return passed[-1]
    failures = [row for row in rows if row.get("status") in FAILURES]
    if failures:
        return failures[-1]
    usable = [row for row in rows if row.get("status") not in {"launcher_error", "interrupted"}]
    return usable[-1] if usable else (rows[-1] if rows else None)


def model_seconds(record: dict[str, Any]) -> float | None:
    values = record.get("model_forward_seconds") or []
    return max(values) if values else None


def fmt_s(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.1f}"


def peaks(record: dict[str, Any]) -> str:
    memory = record.get("nvml_samples") or {}
    values = [f"{gpu}:{data.get('peak_memory_mib', '—')}" for gpu, data in sorted(memory.items(), key=lambda row: int(row[0]))]
    return "/".join(values) + " MiB" if values else "—"


def used_rows(record: dict[str, Any] | None) -> str:
    if not record:
        return "—"
    values = record.get("msa_rows_used_per_prediction") or []
    return str(max(values)) if values else "—"


def lct_msa_followup_table(records: list[dict[str, Any]]) -> list[str]:
    full = next((row for row in records if row.get("target") == "LCT"
                 and row.get("mode") == "big" and "-capacity" in str(row.get("run_id", ""))), None)
    followups = sorted(
        [row for row in records if row.get("target") == "LCT" and row.get("mode") == "big"
         and row.get("msa_rows_test_cap")],
        key=lambda row: str(row.get("ended_at") or row.get("started_at") or ""),
    )
    if not followups:
        return []
    table = [
        "| MSA rows offered | Effective `N_msa` | GPUs | Result | Model s | Wall s | NVML peak MiB, GPUs 0/1 |",
        "|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in ([full] if full else []) + followups:
        offered = row.get("msa_rows_test_cap") or row.get("msa_rows_model_cap") or "—"
        values = row.get("msa_rows_used_per_prediction") or []
        effective = max(values) if values else "—"
        result = str(row.get("status", "unknown"))
        if row.get("status") == "oom":
            result += "; failed in MSA normalization, 5.68 GiB allocation"
        table.append(
            f"| {offered:,} | {effective} | {row.get('gpu_count', '—')} | {result} | "
            f"{fmt_s(model_seconds(row)) if is_valid_pass(row) else '—'} | "
            f"{fmt_s(row.get('wall_seconds'))} | {peaks(row)} |"
        )
    return table


def capped_off_fast_followups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_order = {target: index for index, (target, _) in enumerate(TARGETS)}
    mode_order = {mode: index for index, mode in enumerate(("off", "fast"))}
    rows = [
        row for row in records
        if row.get("mode") in mode_order
        and row.get("msa_rows_test_cap") == 8192
        and "msa8192-retry" in str(row.get("run_id", ""))
    ]
    return sorted(
        rows,
        key=lambda row: (
            target_order.get(str(row.get("target")), len(target_order)),
            mode_order[str(row["mode"])],
        ),
    )


def capped_failure_detail(record: dict[str, Any]) -> str:
    log_path = Path((record.get("paths") or {}).get("log", ""))
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    if "does not support n_token > 2560" in log:
        token_count = "unknown"
        for line in log.splitlines():
            if "N_token " in line:
                token_count = line.split("N_token ", 1)[1].split(",", 1)[0]
                break
        return f"Rejected before model; `N_token={token_count}` exceeds v2 limit 2,560"
    oom_line = next((line for line in log.splitlines() if "failed: CUDA out of memory" in line), "")
    if oom_line:
        request = "unknown"
        free = "unknown"
        if "Tried to allocate " in oom_line:
            request = oom_line.split("Tried to allocate ", 1)[1].split(" GiB", 1)[0] + " GiB"
        if "of which " in oom_line and " GiB is free" in oom_line:
            free = oom_line.split("of which ", 1)[1].split(" GiB is free", 1)[0] + " GiB"
        return f"CUDA OOM; requested {request}, {free} free"
    if is_valid_pass(record):
        return "Pass"
    return str(record.get("status", "unknown")).replace("_", " ")


def capped_off_fast_table(records: list[dict[str, Any]]) -> list[str]:
    rows = capped_off_fast_followups(records)
    if not rows:
        return []
    table = [
        "| Target | Mode | A3M rows offered | Effective `N_msa` | Result | Wall s | NVML peak MiB (GPU 0/1) |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        offered = row.get("msa_rows_variant_input", "—")
        used = row.get("msa_rows_used_per_prediction") or []
        effective = max(used) if used else "—"
        effective_text = f"{effective:,}" if isinstance(effective, int) else str(effective)
        target = row.get("target", "unknown")
        length = next((length for name, length in TARGETS if name == target), None)
        label = f"{target} ({length:,} aa)" if length is not None else str(target)
        table.append(
            f"| {label} | `{row.get('mode')}` | {offered:,} | {effective_text} | "
            f"{capped_failure_detail(row)} | {fmt_s(row.get('wall_seconds'))} | {peaks(row)} |"
        )
    return table


def cell_text(record: dict[str, Any] | None, *, skipped: bool = False) -> str:
    if record is None:
        return "Not run — stopped after earlier failure" if skipped else "Not run"
    status = "Pass" if is_valid_pass(record) else str(record.get("status", "unknown")).replace("_", " ").upper()
    rows = used_rows(record)
    ms = model_seconds(record)
    wall = record.get("wall_seconds")
    memory = peaks(record)
    if is_valid_pass(record):
        return f"{status}; {rows} rows; model {fmt_s(ms)} s; wall {fmt_s(wall)} s; peak {memory}"
    failure_time = record.get("time_to_failure_seconds")
    timing = f"time to failure {fmt_s(failure_time)} s" if failure_time is not None else f"wall {fmt_s(wall)} s"
    return f"{status}; {rows} rows; {timing}; peak {memory}; no valid structure"


def status_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = {}
    for mode in MODES:
        first_failure = None
        for index, (target, _) in enumerate(TARGETS):
            record = selected_attempt(records, target, mode)
            if record and record.get("status") in FAILURES:
                first_failure = {"target": target, "index": index, "status": record["status"], "run_id": record["run_id"]}
                break
        outcomes[mode] = {
            "gpu_assignment": ["0", "1"] if mode == "big" else ["0"],
            "first_prediction_failure": first_failure,
            "stop_rule": "Stop this mode at its first failed prediction; larger panel targets are not run.",
        }
    return outcomes


def warm_stats(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [
        row for row in records
        if row.get("target") == "CHRNA7" and row.get("mode") == mode
        and "-warm" in str(row.get("run_id", "")) and is_valid_pass(row)
    ]
    models = [model_seconds(row) for row in rows if model_seconds(row) is not None]
    walls = [float(row["wall_seconds"]) for row in rows if isinstance(row.get("wall_seconds"), (int, float))]
    return {
        "n": len(rows),
        "median_model_seconds": statistics.median(models) if models else None,
        "median_wall_seconds": statistics.median(walls) if walls else None,
        "model_seconds": models,
        "wall_seconds": walls,
    }


def cold_record(records: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    rows = [
        row for row in records
        if row.get("target") == "CHRNA7" and row.get("mode") == mode
        and "-cold" in str(row.get("run_id", "")) and is_valid_pass(row)
    ]
    return rows[0] if rows else None


def quality_text() -> str:
    root = RESULTS / "quality"
    sections = []
    for target in ("CHRNA7", "SLC26A8"):
        path = root / f"{target}-quality-comparison.json"
        if not path.is_file():
            continue
        data = load_json(path)
        comparison = data.get("candidate_vs_baseline_rmsd", {})
        within = data.get("within_mode_repeat_rmsd", {})
        lines = [f"### {target}", "", f"Baseline mode: `{data.get('baseline_mode')}`; {data.get('residue_count_ca')} Cα atoms.", "", "| Comparison | Pairs | Median RMSD (Å) | Range (Å) |", "|---|---:|---:|---:|"]
        for mode, item in within.items():
            if item.get("n_pairs"):
                lines.append(f"| {mode} repeats | {item['n_pairs']} | {item['median_rmsd_a']:.3f} | {item['min_rmsd_a']:.3f}–{item['max_rmsd_a']:.3f} |")
        for label, item in comparison.items():
            lines.append(f"| {label.replace('_', ' ')} | {item['n_pairs']} | {item['median_rmsd_a']:.3f} | {item['min_rmsd_a']:.3f}–{item['max_rmsd_a']:.3f} |")
        lines += ["", "Median confidence metrics:", "", "```json", json.dumps(data.get("median_confidence_metrics", {}), indent=2), "```", ""]
        sections.append("\n".join(lines))

    path = root / "confidence-stratified-ca-rmsd.json"
    if path.is_file():
        data = load_json(path)
        lines = ["### Confidence-stratified fits", "", "B-factors in the predicted mmCIF encode per-atom pLDDT. The fit uses residues with B-factor/pLDDT ≥70 in both structures.", "", "| Target | Comparison | All-residue global fit (Å) | High-confidence fit, high-confidence residues (Å) | High-confidence-fit residual, other residues (Å) | High-confidence Cα count |", "|---|---|---:|---:|---:|---:|"]
        for item in data:
            lines.append(
                f"| {item['target']} | {item['mobile'].split('-')[0]} vs {item['reference'].split('-')[0]} | "
                f"{item['global_fit_rmsd_all_ca_a']:.3f} | {item['high_confidence_fit_rmsd_high_confidence_ca_a']:.3f} | "
                f"{item['high_confidence_fit_rmsd_other_ca_a']:.3f} | {item['high_confidence_ca_count']} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build() -> None:
    source_manifest = load_json(RESULTS / "manifest.json")
    records = load_records()
    summary = status_summary(records)
    lct_msa_table = lct_msa_followup_table(records)
    capped_retry_table = capped_off_fast_table(records)
    target_specs = source_manifest["targets"]
    table = [
        "| Target | Length | MSA available / cap | `off` · GPU 0 | `fast` · GPU 0 | `big` · GPUs 0,1 |",
        "|---|---:|---:|---|---|---|",
    ]
    for index, (target, length) in enumerate(TARGETS):
        spec = target_specs[target]
        msa = f"{spec['msa_rows_available']:,} / {spec['msa_rows_after_model_cap']:,}"
        cells = []
        for mode in MODES:
            attempt = selected_attempt(records, target, mode)
            skipped = attempt is None and summary[mode]["first_prediction_failure"] and index > summary[mode]["first_prediction_failure"]["index"]
            cells.append(cell_text(attempt, skipped=bool(skipped)))
        table.append(f"| {target} | {length:,} aa | {msa} | " + " | ".join(cells) + " |")

    warm = {mode: warm_stats(records, mode) for mode in ("stock", "off", "fast", "big")}
    cold = {mode: cold_record(records, mode) for mode in ("stock", "off", "fast", "big")}
    warm_table = [
        "| Image/mode | Cold model s | Cold wall s | Warm runs | Median warm model s | Median warm wall s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("stock", "off", "fast", "big"):
        cold_row = cold[mode]
        warm_row = warm[mode]
        warm_table.append(
            f"| {mode} | {fmt_s(model_seconds(cold_row) if cold_row else None)} | "
            f"{fmt_s(cold_row.get('wall_seconds') if cold_row else None)} | {warm_row['n']} | "
            f"{fmt_s(warm_row['median_model_seconds'])} | {fmt_s(warm_row['median_wall_seconds'])} |"
        )

    hardware = source_manifest["hardware"]
    software = source_manifest["software"]
    settings = source_manifest["settings"]
    weight_file = Path("/mnt/db/reference_files/protenix/checkpoint/protenix-v2.pt")
    weights = {
        "identifier": "Protenix v2 checkpoint protenix-v2.pt (stock/PINS.json checkpoint)",
        "path": str(weight_file),
        "size_bytes": weight_file.stat().st_size if weight_file.is_file() else None,
        "sha256": sha256_file(weight_file) if weight_file.is_file() else None,
    }
    report_lines = [
        "# Protenix optimization playbook capacity sweep",
        "",
        "Run date: 2026-10-01  ",
        f"Candidate: `{software['candidate_image']}` (`{hardware['images']['candidate']}`)  ",
        f"Stock: `{software['stock_image']}` (`{hardware['images']['stock']}`)  ",
        "Hardware: two NVIDIA RTX 4090 GPUs, 24 GB each; driver 595.91.07; compute capability 8.9  ",
        f"Raw results: `{RESULTS}`",
        "",
        "## Setup",
        "",
        f"Protenix {software['protenix_version']} commit `{software['protenix_commit']}`; PyTorch {software['torch']}; Triton {software['triton']}; cuEquivariance {software['cuequivariance']}. "
        "The candidate preserves the stock Protenix source and core framework stack; it adds the matching Protenix v2 optimization kit at revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.",
        "",
        f"Weights: `{weights['identifier']}`; SHA-256 `{weights['sha256']}`; {weights['size_bytes']:,} bytes, mounted read-only from `{weights['path']}`.",
        "",
        f"Prediction settings: Protenix v2, {settings['cycle']} cycles, {settings['step']} diffusion steps, {settings['sample']} sample, seed {settings['seeds']}, {settings['dtype']}, MSA enabled, templates disabled. Protenix's model MSA cap is {settings['msa_model_max_size']:,} rows. Each cached A3M query was checked against its target sequence. The effective `N_msa` after Protenix deduplication is shown in each attempt record.",
        "",
        "GPU assignment: stock and candidate `off`/`fast` used GPU 0; candidate `big` used explicit row-pair tensor parallelism across GPUs 0 and 1. GPU 1 retained the background monitor workload. NVML sampled both cards each second, including that background usage. The host topology reports both cards under the same NUMA node and no NVLink connection.",
        "",
        "A pass requires exit success, a CIF that parses with the expected residue count, and a parseable confidence JSON. The original capacity sweep stopped each mode at its first prediction failure. Larger cells shown as skipped refer to that full-depth sweep; separate capped-depth follow-ups are reported below.",
        "",
        "## Capacity sweep",
        "",
        *table,
        "",
        "The first OOMs in the original full-depth sweep were: `off` at NUP155 (1,391 aa); `fast` at LCT (1,927 aa); and two-GPU `big` at LCT (1,927 aa). `off` passed SLC26A8 (970 aa); both optimized modes passed through NUP155 (1,391 aa). `big` also passed CHRNA7 and SLC26A8. CEP350, DMD, and PRKDC were skipped in that original sweep after LCT failed in `big`; capped `off`/`fast` retries are listed separately below.",
        "",
        "For the caught `off` NUP155 OOM, Protenix exited zero without a structure; the harness was corrected to classify OOMs from the log even when exit code is zero. The record preserves the CUDA allocation error and 99.4 s wall time. `fast` LCT failed in 32.7 s wall time. Two-GPU `big` LCT failed in 595.2 s wall time: a rank requested 5.68 GiB with 5.67 GiB free. NVML peaks were 24,005/20,326 MiB for GPUs 0/1. The initial `big` CHRNA7 launcher quoting error and the interrupted first NUP155 attempt are retained as separate non-prediction records; the corrected attempts passed.",
        "",
        *( ["## LCT MSA-depth follow-up", "",
            "To test whether MSA depth caused the LCT limit, the same two-GPU `big` prediction was rerun with only the first 8,192 rows of the cached A3M. The query row still matched the target. All other prediction settings, including 10 cycles and 200 diffusion steps, were held fixed.",
            "",
            *lct_msa_table,
            "",
            "At 8,192 rows Protenix reported `N_msa=8192`, completed, and wrote a parseable 1,927-residue CIF plus confidence JSON. This controlled result shows that lowering this LCT input from 16,384 to 8,192 rows was sufficient to avoid the observed OOM on this two-GPU setup. It is a capacity result; it does not establish that the reduced-depth prediction has equivalent structural quality.",
            ""] if lct_msa_table else []),
        *( ["## Capped MSA `off`/`fast` follow-up", "",
            "The remaining one-GPU `off` and `fast` checks used the first `min(cached A3M rows, 8,192)` rows for each target. Shallower MSAs were left whole; rows were not randomly sampled. The query row was checked against the target, and all predictions retained 10 cycles and 200 diffusion steps. The table reports both rows offered and Protenix's effective `N_msa` after deduplication.",
            "",
            *capped_retry_table,
            "",
            "Capping the MSA did not make the one-GPU `off` or `fast` LCT/EP300 attempts fit; those runs still hit CUDA OOM. NUP155 `off` also OOMed at its original 2,312-row depth. CEP350, DMD, and PRKDC were rejected before model forward because their token counts exceeded Protenix v2's built-in 2,560-token limit. Those are input-length rejections, not GPU-memory OOM measurements; the depth cap does not change token count. The 2-GPU `big` results use a separate execution path and are reported in the capacity and LCT sections above. This follow-up measures capacity only and does not compare structural quality at reduced MSA depth.",
            ""] if capped_retry_table else []),
        "## CHRNA7 cold and warm timing",
        "",
        "CHRNA7 was run once as the first attempt and three more times after the first attempt per image/mode. Each attempt used a fresh container and process. The `fast` and `big` attempts shared persistent Docker volumes for compiled JIT artifacts, so their later runs reused compiled kernels; model weights were reloaded for every attempt. Model seconds use the slowest rank for tensor parallel runs; wall seconds include model startup, preprocessing, inference, output, and CIF validation. The candidate cold and warm runs used 10 cycles, but the stock cold record used 3 cycles while its warm records used 10, so the stock cold/warm row is not a controlled comparison.",
        "",
        *warm_table,
        "",
        "On CHRNA7, candidate `off` matches the stock warm runtime. `fast` reduced median model time from about 31.6 s (`off`) to 18.0 s. Cold `fast` wall time includes the first LayerNorm extension build/JIT cost. Two-GPU `big` is a capacity mode and is slower on this short target.",
        "",
        "## Structural and confidence comparison",
        "",
        "Same-seed mode outputs were superposed by matching Cα residue keys. These checks compare the implementation outputs and repeatability; they do not measure agreement with an experimental structure. Confidence scores are model-native scores, not independent accuracy validation.",
        "",
        quality_text(),
        "",
        "The full-chain RMSDs for SLC26A8 and CHRNA7 are driven mainly by low-confidence regions and their relative placement. After fitting on residues with per-atom pLDDT ≥70, the high-confidence regions align closely across modes (0.10 Å for SLC26A8 `fast` vs `off`; 0.50 Å for `big` vs `off`; 1.12 Å for CHRNA7 `big` vs stock). Report both fits; do not infer global structural equivalence from confidence scores alone.",
        "",
        "## Interpretation and limits",
        "",
        "- `fast` was faster than `off` on CHRNA7 and SLC26A8, and completed NUP155 on one GPU where `off` OOMed. Its 1,391-aa NUP155 run reached 23,787 MiB sampled on GPU 0, leaving little memory headroom.",
        "- `big` explicitly row-pair-sharded a single prediction over both RTX 4090 cards. Full-cap LCT at 1,927 aa OOMed in MSA normalization with 16,384 rows, while the same prediction passed with 8,192 rows. This shows MSA depth was a limiting factor for this LCT run; it does not give a universal safe MSA depth or validate reduced-depth quality.",
        "- On RTX 4090, the kit's unsupported sm_89 kernel routes were explicitly ablated. Several selected transition/TriMul provider choices inherit measurements from sm_80; successful execution on these runs does not establish their speed on every 4090 shape.",
        "- The candidate remains experimental. The benchmark does not pass the release gate for general use: long-target quality is only sampled on short/medium proteins, most combinations skipped by the original stop rule remain untested, and the global structures show low-confidence-region differences.",
        "",
        "## Reproducibility and outputs",
        "",
        f"The report generator snapshots all {len(records)} structured attempt records in [`{SNAPSHOT.name}`]({SNAPSHOT.name}). Each record contains the command, image digest, upstream and kit pins, input/MSA hashes, settings, phase timings, effective MSA rows, exit/status, GPU assignment and sampled peaks, output parser results, confidence metrics, and raw artifact paths. Raw run logs, NVML CSVs, inputs, outputs, and per-attempt records remain under `{RESULTS}`.",
        "",
        "The app summary is generated from the same records and links each `off`/`fast`/`big` cell to its full attempt page in `mn-cofolding`.",
        "",
    ]
    report = "\n".join(report_lines)
    report = report.replace(f"[`{SNAPSHOT.name}`]({SNAPSHOT.name})", f"[{SNAPSHOT.name}]({SNAPSHOT.name})")
    REPORT.write_text(report, encoding="utf-8")

    snapshot = {
        "schema_version": 1,
        "benchmark": "Protenix optimization playbook capacity sweep",
        "run_date": source_manifest.get("run_date"),
        "results_root": str(RESULTS),
        "hardware": hardware,
        "software": software,
        "weights": weights,
        "settings": settings,
        "targets": source_manifest["targets"],
        "mode_outcomes": summary,
        "quality": {
            path.stem: load_json(path)
            for path in sorted((RESULTS / "quality").glob("*.json"))
        },
        "attempts": records,
    }
    source_manifest["weights"] = weights
    source_manifest["capacity_sweeps"] = summary
    source_manifest["report_snapshot"] = str(SNAPSHOT)
    source_manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RESULTS / "manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT} and {SNAPSHOT} with {len(records)} attempt records")


if __name__ == "__main__":
    build()
