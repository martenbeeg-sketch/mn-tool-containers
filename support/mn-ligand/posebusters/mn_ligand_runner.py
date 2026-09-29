"""mn-ligand adapter for parallel PoseBusters validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import gemmi
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from posebusters import PoseBusters
from mn_ligand_qualification import qualify_molecules


WATER = {"HOH", "WAT", "DOD"}


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def _complex_components(
    complex_path: Path,
    smiles: str,
    output_dir: Path,
    pose_id: str,
) -> tuple[Path, Path, str]:
    structure = gemmi.read_structure(str(complex_path))
    if not structure or not structure[0]:
        raise ValueError("predicted complex contains no model")
    pdb_text = structure.make_pdb_string()
    protein_lines = [
        line
        for line in pdb_text.splitlines()
        if line.startswith("ATOM  ")
    ]
    if not protein_lines:
        raise ValueError("predicted complex contains no protein atoms")
    protein_path = output_dir / f"{pose_id}.protein.pdb"
    protein_path.write_text("\n".join(protein_lines) + "\nEND\n")

    ligand_residues = []
    for chain in structure[0]:
        for residue in chain:
            name = residue.name.strip().upper()
            if name in WATER:
                continue
            residue_info = gemmi.find_tabulated_residue(name)
            if residue_info.is_amino_acid() or residue_info.is_nucleic_acid():
                continue
            heavy_atoms = [
                atom for atom in residue if atom.element.name.upper() != "H"
            ]
            if heavy_atoms:
                ligand_residues.append((chain.name, residue, heavy_atoms))
    if not ligand_residues:
        raise ValueError("predicted complex contains no non-polymer ligand")
    chain_name, residue, atoms = max(
        ligand_residues,
        key=lambda item: len(item[2]),
    )
    template = Chem.MolFromSmiles(str(smiles or "").strip())
    if template is None:
        raise ValueError("no valid source SMILES is available for ligand topology")
    template = Chem.RemoveHs(template, sanitize=False)
    if template.GetNumAtoms() != len(atoms):
        raise ValueError(
            "predicted/source ligand heavy-atom counts differ "
            f"({len(atoms)} vs {template.GetNumAtoms()})"
        )

    predicted_elements = [atom.element.name.title() for atom in atoms]
    template_elements = [atom.GetSymbol() for atom in template.GetAtoms()]
    method = "source topology with native atom order"
    if predicted_elements == template_elements:
        ligand = Chem.Mol(template)
        conformer = Chem.Conformer(ligand.GetNumAtoms())
        for index, atom in enumerate(atoms):
            conformer.SetAtomPosition(
                index,
                (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)),
            )
        ligand.RemoveAllConformers()
        ligand.AddConformer(conformer, assignId=True)
    else:
        pdb_lines = []
        for index, atom in enumerate(atoms, start=1):
            pdb_lines.append(
                "HETATM"
                f"{index:5d} "
                f"{atom.name[:4]:<4} "
                f"{residue.name[:3]:>3} "
                f"{(chain_name or 'L')[:1]}{1:4d}    "
                f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                f"{1.0:6.2f}{0.0:6.2f}          "
                f"{atom.element.name:>2}"
            )
        inferred = Chem.MolFromPDBBlock(
            "\n".join(pdb_lines) + "\nEND\n",
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )
        if inferred is None:
            raise ValueError("RDKit could not reconstruct predicted ligand coordinates")
        try:
            ligand = AllChem.AssignBondOrdersFromTemplate(template, inferred)
        except ValueError as exc:
            raise ValueError(
                "predicted ligand connectivity could not be mapped to source topology"
            ) from exc
        method = "source topology mapped onto proximity-inferred native coordinates"
    ligand.SetProp("_Name", pose_id)
    ligand.SetProp("source_smiles", str(smiles))
    ligand.SetProp("coordinate_reconstruction", method)
    ligand_path = output_dir / f"{pose_id}.ligand.sdf"
    writer = Chem.SDWriter(str(ligand_path))
    writer.write(ligand)
    writer.close()
    return ligand_path, protein_path, method


def _is_boolean_series(series: pd.Series) -> bool:
    values = {
        value
        for value in series.dropna().tolist()
        if not (isinstance(value, float) and math.isnan(value))
    }
    return bool(values) and values.issubset({True, False})


def _applicable_binary_columns(validator: PoseBusters) -> list[str]:
    columns: list[str] = []
    for module in validator.config.get("modules", []):
        module_name = str(module.get("name") or "")
        suffix = str(module.get("rename_suffix") or "")
        rename_outputs = module.get("rename_outputs") or {}
        for output in module.get("chosen_binary_test_output") or []:
            renamed = str(rename_outputs.get(output) or f"{output}{suffix}")
            columns.append(renamed.lower().replace(" ", "_"))
    return list(dict.fromkeys(columns))


def run(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    report_path: Path,
    *,
    max_workers: int,
) -> int:
    workspace = input_path.resolve().parents[1]
    input_rows = pd.read_csv(input_path).fillna("")
    prepared_dir = workspace / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    valid_rows: list[dict[str, object]] = []
    preparation_failures: list[dict[str, object]] = []
    for index, row in input_rows.iterrows():
        record = row.to_dict()
        pose_id = _safe_id(record.get("pose_id"), f"pose_{index + 1:07d}")
        try:
            if str(record.get("complex_file") or ""):
                ligand_path, protein_path, preparation_method = _complex_components(
                    workspace / str(record["complex_file"]),
                    str(record.get("smiles") or ""),
                    prepared_dir,
                    pose_id,
                )
            else:
                ligand_path = workspace / str(record.get("mol_pred") or "")
                protein_path = workspace / str(record.get("mol_cond") or "")
                preparation_method = "native ligand pose and source receptor"
                if not ligand_path.is_file() or not protein_path.is_file():
                    raise FileNotFoundError("pose or receptor input is missing")
            valid_rows.append(
                {
                    **record,
                    "pose_id": pose_id,
                    "_mol_pred": str(ligand_path),
                    "_mol_cond": str(protein_path),
                    "preparation_method": preparation_method,
                }
            )
        except Exception as exc:
            preparation_failures.append(
                {
                    **record,
                    "pose_id": pose_id,
                    "passed_all": False,
                    "passed_test_count": 0,
                    "failed_test_count": 1,
                    "failed_checks": "input preparation",
                    "input_preparation_error": str(exc),
                }
            )

    full = pd.DataFrame()
    summaries: list[dict[str, object]] = []
    if valid_rows:
        table = pd.DataFrame(
            {
                "mol_pred": [row["_mol_pred"] for row in valid_rows],
                "mol_cond": [row["_mol_cond"] for row in valid_rows],
            }
        )
        validator = PoseBusters(
            config="dock",
            max_workers=max(1, int(max_workers)),
            chunk_size=None,
        )
        full = validator.bust_table(table, full_report=True).reset_index()
        configured_binary_columns = _applicable_binary_columns(validator)
        row_by_path = {
            str(row["_mol_pred"]): row for row in valid_rows
        }
        metadata_rows = [
            row_by_path.get(str(path), {})
            for path in full["file"].astype(str)
        ]
        metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
        full = pd.concat(
            [metadata.drop(columns=["_mol_pred", "_mol_cond"], errors="ignore"), full],
            axis=1,
        )
        boolean_columns = [
            column
            for column in configured_binary_columns
            if column in full.columns and _is_boolean_series(full[column])
        ]
        for _, result in full.iterrows():
            failed = [
                column
                for column in boolean_columns
                if result.get(column) is False
            ]
            passed = [
                column
                for column in boolean_columns
                if result.get(column) is True
            ]
            summaries.append(
                {
                    "pose_id": result.get("pose_id", ""),
                    "compound_id": result.get("compound_id", ""),
                    "source_engine": result.get("source_engine", ""),
                    "source_kind": result.get("source_kind", ""),
                    "replicate": result.get("replicate", ""),
                    "prediction": result.get("prediction", ""),
                    "selection_criterion": result.get(
                        "selection_criterion", ""
                    ),
                    "passed_all": not failed and bool(passed),
                    "passed_test_count": len(passed),
                    "failed_test_count": len(failed),
                    "failed_checks": "; ".join(failed),
                    "input_preparation_error": "",
                    "preparation_method": result.get("preparation_method", ""),
                }
            )
    summaries.extend(preparation_failures)
    summary = pd.DataFrame(summaries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not full.empty:
        full.to_csv(output_path, index=False)
    else:
        output_path.write_text("")
    summary.to_csv(summary_path, index=False)
    passed_count = int(summary.get("passed_all", pd.Series(dtype=bool)).eq(True).sum())
    report = {
        "success": bool(len(summary)),
        "input_count": int(len(input_rows)),
        "validated_count": int(len(summary)),
        "passed_count": passed_count,
        "failed_count": int(len(summary) - passed_count),
        "preparation_failure_count": len(preparation_failures),
        "max_workers": max(1, int(max_workers)),
        "config": "dock",
        "applicable_checks": (
            configured_binary_columns if valid_rows else []
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if len(summary) == len(input_rows) else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dock", "qualify-molecules"),
        default="dock",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/input/validation_inputs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/native/posebusters_full.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/workspace/posebusters_summary.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/workspace/posebusters_report.json"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--molecule-input",
        type=Path,
        default=Path("/workspace/normalized/generated_compounds.sdf"),
    )
    parser.add_argument(
        "--molecule-table",
        type=Path,
        default=Path("/workspace/normalized/generated_compounds.csv"),
    )
    parser.add_argument(
        "--qualification-output",
        type=Path,
        default=Path("/workspace/qualified"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--conformer-count", type=int, default=20)
    parser.add_argument("--min-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-heavy-atoms", type=int, default=80)
    parser.add_argument("--max-absolute-charge", type=int, default=2)
    parser.add_argument("--max-sa-score", type=float, default=6.0)
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        import posebusters

        print(f"PoseBusters {posebusters.__version__}")
        return
    if args.mode == "qualify-molecules":
        report = qualify_molecules(
            args.molecule_input,
            args.molecule_table,
            args.qualification_output,
            seed=args.seed,
            conformer_count=args.conformer_count,
            max_workers=args.max_workers,
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
            max_absolute_charge=args.max_absolute_charge,
            max_sa_score=args.max_sa_score,
        )
        print(json.dumps(report, indent=2))
        return
    raise SystemExit(
        run(
            args.input,
            args.output,
            args.summary,
            args.report,
            max_workers=args.max_workers,
        )
    )


if __name__ == "__main__":
    main()
