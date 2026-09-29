"""mn-ligand batch adapter for PandaMap interaction analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from threading import Lock

import gemmi
import pandas as pd
from rdkit import Chem

from pandamap.core import HybridProtLigMapper

WATER = {"HOH", "WAT", "DOD"}
PLOT_LOCK = Lock()


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def _is_polymer_like_residue(residue: gemmi.Residue) -> bool:
    """Recognize author-numbered modified amino acids stored as HETATM."""
    atom_names = {atom.name.strip().upper() for atom in residue}
    return {"N", "CA", "C", "O"}.issubset(atom_names)


def _polymer_chains(
    structure: gemmi.Structure,
    *,
    include_polymer_like: bool = False,
) -> dict[str, list[gemmi.Residue]]:
    chains: dict[str, list[gemmi.Residue]] = {}
    for chain in structure[0]:
        residues = [
            residue for residue in chain
            if (
                gemmi.find_tabulated_residue(
                    residue.name.strip().upper()
                ).is_amino_acid()
                or (
                    include_polymer_like
                    and residue.entity_type == gemmi.EntityType.Polymer
                    and _is_polymer_like_residue(residue)
                )
            )
        ]
        if residues:
            chains[str(chain.name)] = residues
    return chains


def _sequence(residues: list[gemmi.Residue]) -> str:
    sequence = []
    for residue in residues:
        code = (
            gemmi.find_tabulated_residue(
                residue.name.strip().upper()
            ).one_letter_code or "X"
        ).strip().upper()
        sequence.append(code if len(code) == 1 else "X")
    return "".join(sequence)


def _align_indices(left: str, right: str) -> tuple[int, list[tuple[int, int]]]:
    rows, columns = len(left) + 1, len(right) + 1
    score = [[0] * columns for _ in range(rows)]
    trace = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        score[i][0], trace[i][0] = -2 * i, "U"
    for j in range(1, columns):
        score[0][j], trace[0][j] = -2 * j, "L"
    for i in range(1, rows):
        for j in range(1, columns):
            choices = (
                (score[i - 1][j - 1] + (2 if left[i - 1] == right[j - 1] else -1), "D"),
                (score[i - 1][j] - 2, "U"),
                (score[i][j - 1] - 2, "L"),
            )
            score[i][j], trace[i][j] = max(
                choices, key=lambda item: (item[0], item[1] == "D")
            )
    pairs: list[tuple[int, int]] = []
    i, j = len(left), len(right)
    while i or j:
        direction = trace[i][j]
        if direction == "D":
            i -= 1
            j -= 1
            pairs.append((i, j))
        elif direction == "U":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return score[-1][-1], pairs


def _author_residue_mapping(
    predicted: gemmi.Structure,
    reference: gemmi.Structure,
) -> tuple[dict[tuple[str, int, str], tuple[str, int, str]], dict[str, object]]:
    predicted_chains = _polymer_chains(predicted)
    reference_chains = _polymer_chains(
        reference, include_polymer_like=True
    )
    mapping: dict[tuple[str, int, str], tuple[str, int, str]] = {}
    assignments: list[dict[str, object]] = []
    available = set(reference_chains)
    for predicted_chain, predicted_residues in predicted_chains.items():
        candidates = []
        predicted_sequence = _sequence(predicted_residues)
        for reference_chain in available or set(reference_chains):
            reference_residues = reference_chains[reference_chain]
            reference_sequence = _sequence(reference_residues)
            alignment_score, pairs = _align_indices(
                predicted_sequence, reference_sequence
            )
            matches = sum(
                predicted_sequence[i] == reference_sequence[j]
                for i, j in pairs
            )
            candidates.append(
                (matches, alignment_score, len(pairs), reference_chain, pairs)
            )
        if not candidates:
            continue
        matches, alignment_score, _, reference_chain, pairs = max(candidates)
        available.discard(reference_chain)
        reference_residues = reference_chains[reference_chain]
        mapped = 0
        for predicted_index, reference_index in pairs:
            predicted_residue = predicted_residues[predicted_index]
            reference_residue = reference_residues[reference_index]
            predicted_code = _sequence([predicted_residue])
            reference_code = _sequence([reference_residue])
            if predicted_code != reference_code and reference_code != "X":
                continue
            mapping[(
                predicted_chain,
                predicted_residue.seqid.num,
                str(predicted_residue.seqid.icode).strip(),
            )] = (
                reference_chain,
                reference_residue.seqid.num,
                str(reference_residue.seqid.icode).strip(),
            )
            mapped += 1
        assignments.append({
            "predicted_chain": predicted_chain,
            "reference_chain": reference_chain,
            "predicted_residues": len(predicted_residues),
            "reference_residues": len(reference_residues),
            "mapped_residues": mapped,
            "identical_aligned_residues": matches,
            "alignment_score": alignment_score,
        })
    return mapping, {
        "policy": "chain-wise global protein-sequence alignment to source target author numbering",
        "mapped_residue_count": len(mapping),
        "chain_assignments": assignments,
    }


def _renumber_atom_line(
    line: str,
    mapping: dict[tuple[str, int, str], tuple[str, int, str]],
) -> str:
    try:
        key = (line[21:22].strip(), int(line[22:26]), line[26:27].strip())
    except ValueError:
        return line
    mapped = mapping.get(key)
    if mapped is None:
        return line
    chain, number, insertion = mapped
    return (
        line[:21] + (chain[:1] or " ") + f"{number:4d}"
        + (insertion[:1] or " ") + line[27:]
    )


def _prepare_complex(
    row: dict[str, object], workspace: Path, target: Path
) -> dict[str, object]:
    complex_file = str(row.get("complex_file") or "").strip()
    if complex_file:
        structure = gemmi.read_structure(str(workspace / complex_file))
        reference = gemmi.read_structure(
            str(workspace / str(row.get("reference_target") or ""))
        )
        if not reference or not reference[0]:
            raise ValueError("source target contains no model")
        residue_mapping, mapping_report = _author_residue_mapping(
            structure, reference
        )
        if not residue_mapping:
            raise ValueError(
                "could not map predicted protein residues to source target numbering"
            )
        ligands = []
        for chain in structure[0]:
            for residue in chain:
                name = residue.name.strip().upper()
                info = gemmi.find_tabulated_residue(name)
                if name in WATER or info.is_amino_acid() or info.is_nucleic_acid():
                    continue
                atoms = [atom for atom in residue if atom.element.name.upper() != "H"]
                if atoms:
                    ligands.append((chain.name, residue, atoms))
        if not ligands:
            raise ValueError("complex contains no non-polymer ligand")
        selected_name = str(
            row.get("ligand_residue_name") or ""
        ).strip().upper()
        selected_chain = str(row.get("ligand_chain") or "").strip()
        selected_insertion = str(
            row.get("ligand_insertion_code") or ""
        ).strip()
        try:
            selected_number = int(
                float(str(row.get("ligand_residue_number") or ""))
            )
        except ValueError:
            selected_number = None
        if selected_name and selected_number is not None:
            selected = next(
                (
                    item
                    for item in ligands
                    if item[0].strip() == selected_chain
                    and item[1].name.strip().upper() == selected_name
                    and item[1].seqid.num == selected_number
                    and str(item[1].seqid.icode).strip()
                    == selected_insertion
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    "selected target-complex ligand residue is unavailable"
                )
            ligand_chain, ligand_residue, _ = selected
        else:
            ligand_chain, ligand_residue, _ = max(
                ligands, key=lambda item: len(item[2])
            )
        protein, ligand = [], []
        for line in structure.make_pdb_string().splitlines():
            if line.startswith("ATOM  "):
                protein.append(_renumber_atom_line(line, residue_mapping))
            elif line.startswith("HETATM"):
                try:
                    same = (
                        line[21:22].strip() == str(ligand_chain).strip()
                        and int(line[22:26]) == ligand_residue.seqid.num
                    )
                except ValueError:
                    same = False
                if same:
                    ligand.append(line[:17] + "LIG" + line[20:21] + "Z" + f"{1:4d}" + line[26:])
        if not protein or not ligand:
            raise ValueError("could not isolate protein and ligand coordinates")
        target.write_text("\n".join(protein + ligand) + "\nEND\n")
        mapping_report["reference_target"] = str(
            row.get("reference_target") or ""
        )
        mapping_report["predicted_to_reference"] = [
            {
                "predicted_chain": predicted[0],
                "predicted_residue_number": predicted[1],
                "predicted_insertion_code": predicted[2],
                "reference_chain": mapped[0],
                "reference_residue_number": mapped[1],
                "reference_insertion_code": mapped[2],
            }
            for predicted, mapped in sorted(residue_mapping.items())
        ]
        return mapping_report
    receptor = workspace / str(row.get("mol_cond") or "")
    ligand_path = workspace / str(row.get("mol_pred") or "")
    if not receptor.is_file() or not ligand_path.is_file():
        raise FileNotFoundError("receptor or ligand pose is missing")
    reference_path = workspace / str(row.get("reference_target") or "")
    if not reference_path.is_file():
        raise FileNotFoundError("source target for residue mapping is missing")
    receptor_structure = gemmi.read_structure(str(receptor))
    reference_structure = gemmi.read_structure(str(reference_path))
    residue_mapping, mapping_report = _author_residue_mapping(
        receptor_structure, reference_structure
    )
    if not residue_mapping:
        raise ValueError(
            "could not map prepared receptor residues to source target numbering"
        )
    protein = [
        _renumber_atom_line(line, residue_mapping)
        for line in receptor.read_text(errors="replace").splitlines()
        if line.startswith("ATOM  ")
    ]
    molecule = Chem.MolFromMolFile(str(ligand_path), sanitize=False, removeHs=False)
    if molecule is None:
        raise ValueError("RDKit could not read ligand pose")
    ligand, serial = [], len(protein) + 1
    for line in Chem.MolToPDBBlock(molecule).splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        line = f"HETATM{serial:5d}" + line[11:]
        ligand.append(line[:17] + "LIG" + line[20:21] + "Z" + f"{1:4d}" + line[26:])
        serial += 1
    target.write_text("\n".join(protein + ligand) + "\nEND\n")
    mapping_report["reference_target"] = str(reference_path)
    mapping_report["prepared_to_reference"] = [
        {
            "prepared_chain": prepared[0],
            "prepared_residue_number": prepared[1],
            "prepared_insertion_code": prepared[2],
            "reference_chain": mapped[0],
            "reference_residue_number": mapped[1],
            "reference_insertion_code": mapped[2],
        }
        for prepared, mapped in sorted(residue_mapping.items())
    ]
    return mapping_report


def _native_value(value: object) -> object:
    if hasattr(value, "get_id"):
        return str(value.get_id())
    if hasattr(value, "resname") and hasattr(value, "id"):
        return f"{value.resname}:{value.id[1]}"
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def run(
    input_path: Path,
    native_dir: Path,
    summary_path: Path,
    interactions_path: Path,
    report_path: Path,
    *,
    max_workers: int,
) -> int:
    workspace = input_path.resolve().parents[1]
    native_dir.mkdir(parents=True, exist_ok=True)
    prepared = workspace / "prepared"
    prepared.mkdir(exist_ok=True)
    inputs = pd.read_csv(input_path).fillna("").to_dict("records")
    def analyze(item: tuple[int, dict[str, object]]):
        index, row = item
        pose_id = _safe_id(row.get("pose_id"), f"pose_{index:07d}")
        pose_dir = native_dir / pose_id
        pose_dir.mkdir()
        complex_path = prepared / f"{pose_id}.complex.pdb"
        meta = {key: row.get(key, "") for key in (
            "pose_id", "compound_id", "source_engine", "source_kind",
            "replicate", "prediction", "selection_criterion",
            "ligand_chain", "ligand_residue_name",
            "ligand_residue_number", "ligand_insertion_code",
            "ligand_heavy_atom_count",
        )}
        meta["pose_id"] = pose_id
        try:
            numbering = _prepare_complex(row, workspace, complex_path)
            (pose_dir / "residue_numbering.json").write_text(
                json.dumps(numbering, indent=2) + "\n"
            )
            mapper = HybridProtLigMapper(str(complex_path), ligand_resname="LIG")
            mapper.detect_interactions()
            image_path = pose_dir / "pandamap.png"
            # Matplotlib's global renderer state is not thread-safe. Keep
            # chemistry/contact detection parallel, but serialize only image
            # creation so one pose cannot corrupt another pose's renderer.
            with PLOT_LOCK:
                mapper.visualize(
                    output_file=str(image_path),
                    dpi=180,
                    title=str(row.get("compound_id") or pose_id),
                )
            affinity = mapper.estimate_binding_affinity()
            pose_rows = []
            counts, residues = {}, set()
            for interaction_type, interactions in mapper.interactions.items():
                counts[interaction_type] = len(interactions)
                for interaction in interactions:
                    residue = interaction.get("protein_residue")
                    chain = ""
                    residue_name = ""
                    residue_number = ""
                    if residue is not None:
                        residue_name = str(getattr(residue, "resname", ""))
                        residue_number = str(getattr(residue, "id", ("", "", ""))[1])
                        parent = getattr(residue, "parent", None)
                        chain = str(getattr(parent, "id", "") or "")
                    residues.add(f"{chain}:{residue_name}{residue_number}")
                    native = {key: _native_value(value) for key, value in interaction.items()}
                    pose_rows.append({
                        **meta, "interaction_type": interaction_type.replace("_", " "),
                        "protein_chain": chain, "protein_residue_name": residue_name,
                        "protein_residue_number": residue_number,
                        "distance_angstrom": interaction.get("distance", ""),
                        "angle_degree": interaction.get("angle", ""),
                        "native_fields_json": json.dumps(native, sort_keys=True),
                    })
            return pose_rows, {
                **meta, "success": True, "interaction_count": len(pose_rows),
                "contacted_residue_count": len(residues),
                "contacted_residues": "; ".join(sorted(residues)),
                "interaction_counts_json": json.dumps(counts, sort_keys=True),
                "empirical_delta_g_kcal_mol": affinity.get("dG_estimated", ""),
                "empirical_delta_g_interpretation": affinity.get("interpretation", ""),
            }, None
        except Exception as exc:
            failure = {**meta, "error": str(exc)}
            return [], {
                **meta, "success": False, "interaction_count": 0,
                "contacted_residue_count": 0, "contacted_residues": "",
                "interaction_counts_json": "{}", "empirical_delta_g_kcal_mol": "",
                "empirical_delta_g_interpretation": "", "error": str(exc),
            }, failure
    details, summaries, failures = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        for pose_rows, summary, failure in executor.map(
            analyze, enumerate(inputs, start=1)
        ):
            details.extend(pose_rows)
            summaries.append(summary)
            if failure is not None:
                failures.append(failure)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    columns = [
        "pose_id", "compound_id", "source_engine", "source_kind", "replicate",
        "prediction", "selection_criterion", "ligand_chain",
        "ligand_residue_name", "ligand_residue_number",
        "ligand_insertion_code", "ligand_heavy_atom_count",
        "interaction_type", "protein_chain",
        "protein_residue_name", "protein_residue_number", "distance_angstrom",
        "angle_degree", "native_fields_json",
    ]
    pd.DataFrame(details, columns=columns).to_csv(interactions_path, index=False)
    report = {
        "engine": "PandaMap", "expected_count": len(inputs),
        "analyzed_count": len(inputs) - len(failures),
        "failed_count": len(failures), "interaction_count": len(details),
        "failures": failures,
        "protein_residue_numbering": "immutable imported-source author numbering",
        "residue_numbering_policy_version": 2,
        "delta_g_note": "PandaMap empirical estimate; not an experimental or rigorous free energy.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--native-output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    return run(
        args.input, args.native_output, args.summary, args.interactions,
        args.report, max_workers=args.max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
