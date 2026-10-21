#!/alphafold3_venv/bin/python
"""Prepare per-chain PDB/mmCIF uploads as AlphaFold 3 JSON templates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


MAX_TEMPLATE_FILE_BYTES = 20 * 1024 * 1024
MAX_TEMPLATE_TOTAL_BYTES = 50 * 1024 * 1024
MAX_TEMPLATE_ALIGNMENT_CELLS = 25_000_000
TEMPLATE_SUFFIXES = {".pdb", ".ent", ".cif", ".mcif", ".mmcif"}


def protein_entity_indices(payload: dict[str, Any]) -> dict[str, int]:
    chain_to_entity: dict[str, int] = {}
    for entity_index, entity in enumerate(payload.get("sequences", [])):
        protein = entity.get("protein") if isinstance(entity, dict) else None
        if not isinstance(protein, dict):
            continue
        raw_ids = protein.get("id")
        chain_ids = [raw_ids] if isinstance(raw_ids, str) else raw_ids
        if not isinstance(chain_ids, list) or not chain_ids:
            continue
        for raw_chain_id in chain_ids:
            chain_id = str(raw_chain_id).strip()
            if not chain_id:
                continue
            previous = chain_to_entity.get(chain_id)
            if previous is not None and previous != entity_index:
                raise ValueError(f"AF3 JSON reuses protein chain ID {chain_id!r} in multiple sequence entries")
            chain_to_entity[chain_id] = entity_index
    return chain_to_entity


def sequence_alignment_indices(query: str, template: str) -> tuple[list[int], list[int]]:
    if not query or not template:
        raise ValueError("Template and target protein sequences must both be non-empty")
    if len(query) * len(template) > MAX_TEMPLATE_ALIGNMENT_CELLS:
        raise ValueError("Template alignment is too large for automatic mapping")
    import gemmi

    scoring = gemmi.AlignmentScoring()
    scoring.match = 3
    scoring.mismatch = -2
    scoring.gapo = -6
    scoring.gape = -1
    alignment = gemmi.align_string_sequences(list(query), list(template), [0] * len(template), scoring)
    query_indices: list[int] = []
    template_indices: list[int] = []
    query_index = template_index = 0
    cigar = alignment.cigar_str()
    operations = list(re.finditer(r"(\d+)([MID])", cigar))
    if not operations or "".join(item.group(0) for item in operations) != cigar:
        raise ValueError("Could not interpret the template sequence alignment")
    for operation in operations:
        length, code = int(operation.group(1)), operation.group(2)
        if code == "M":
            query_indices.extend(range(query_index, query_index + length))
            template_indices.extend(range(template_index, template_index + length))
            query_index += length
            template_index += length
        elif code == "I":
            query_index += length
        else:
            template_index += length
    if query_index != len(query) or template_index != len(template):
        raise ValueError("Template alignment did not cover both protein sequences")
    if not query_indices:
        raise ValueError("No target residues map to this structure template")
    return query_indices, template_indices


def load_single_chain_template(filename: str, content: bytes) -> tuple[str, str]:
    import gemmi

    suffix = Path(filename).suffix.lower()
    if suffix not in TEMPLATE_SUFFIXES:
        raise ValueError(f"Unsupported template format for {filename!r}; use PDB or mmCIF")
    try:
        structure_text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Template {filename!r} must be a text PDB or mmCIF file") from exc
    is_mmcif = suffix in {".cif", ".mcif", ".mmcif"}
    try:
        if is_mmcif:
            document = gemmi.cif.read_string(structure_text)
            if len(document) == 0:
                raise ValueError("the mmCIF file has no data block")
            structure = gemmi.make_structure_from_block(document.sole_block())
        else:
            structure = gemmi.read_pdb_string(structure_text)
            structure.setup_entities()
            structure.add_entity_types()
    except Exception as exc:
        raise ValueError(f"Could not parse template {filename!r}: {exc}") from exc
    if len(structure) == 0:
        raise ValueError(f"Template {filename!r} contains no coordinate models")

    peptide_types = {gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD}
    polymer_chains = []
    for chain in structure[0]:
        polymer = chain.get_polymer()
        if not len(polymer):
            continue
        try:
            entity = structure.get_entity_of(polymer)
        except Exception:
            continue
        names = entity.full_sequence or [residue.name for residue in polymer]
        sequence = gemmi.one_letter_code(names).replace(" ", "").upper()
        if sequence:
            polymer_chains.append((chain, list(polymer), sequence, entity.polymer_type in peptide_types))
    protein_chains = [entry for entry in polymer_chains if entry[3]]
    if len(protein_chains) != 1 or len(polymer_chains) != 1:
        raise ValueError(
            f"Template {filename!r} must contain exactly one polymer chain and it must be protein; "
            f"found {len(protein_chains)} protein and {len(polymer_chains) - len(protein_chains)} other polymer chains"
        )

    chain, polymer_residues, sequence, _ = protein_chains[0]
    if is_mmcif:
        return structure_text, sequence

    observed_sequence = gemmi.one_letter_code([residue.name for residue in polymer_residues]).replace(" ", "").upper()
    full_indices, observed_indices = sequence_alignment_indices(sequence, observed_sequence)
    observed_to_full = dict(zip(observed_indices, full_indices, strict=True))
    residue_to_label_seq: dict[tuple[str, str, str, str], str] = {}
    for observed_index, residue in enumerate(polymer_residues):
        full_index = observed_to_full.get(observed_index)
        if full_index is None:
            continue
        insertion_code = str(residue.seqid.icode).strip().replace("\x00", "")
        if insertion_code in {"", ".", "?"}:
            insertion_code = "."
        residue_to_label_seq[(chain.name, str(residue.seqid.num), residue.name.upper(), insertion_code)] = str(full_index + 1)

    template_structure = structure.clone()
    template_structure.remove_ligands_and_waters()
    template_structure.remove_empty_chains()
    try:
        document = template_structure.make_mmcif_document()
        block = document.sole_block()
        info = dict(structure.info.items())
        deposition_date = str(info.get("_pdbx_database_status.recvd_initial_deposition_date", "")).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", deposition_date):
            raise ValueError("PDB template has no valid deposition date; use mmCIF with revision history")
        history = block.init_loop(
            "_pdbx_audit_revision_history.",
            ["ordinal", "major_revision", "minor_revision", "revision_date", "details"],
        )
        history.add_row(["1", "1", "0", deposition_date, "."])
        label_seq_ids = block.find_loop("_atom_site.label_seq_id")
        values = [
            block.find_values("_atom_site.auth_asym_id"),
            block.find_values("_atom_site.auth_seq_id"),
            block.find_values("_atom_site.label_comp_id"),
            block.find_values("_atom_site.pdbx_PDB_ins_code"),
        ]
        for atom_index, (auth_asym_id, auth_seq_id, comp_id, insertion_code) in enumerate(zip(*values, strict=True)):
            normalized_ins_code = str(insertion_code).strip()
            if normalized_ins_code in {"", ".", "?"}:
                normalized_ins_code = "."
            key = (str(auth_asym_id).strip(), str(auth_seq_id).strip(), str(comp_id).strip().upper(), normalized_ins_code)
            label_seq_id = residue_to_label_seq.get(key)
            if label_seq_id is not None:
                label_seq_ids[atom_index] = label_seq_id
        return document.as_string(), sequence
    except Exception as exc:
        raise ValueError(f"Could not convert template {filename!r} to mmCIF: {exc}") from exc


def prepare(input_json: Path, template_dir: Path, output_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads(input_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read AF3 JSON input: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), list):
        raise ValueError("Input must be AlphaFold 3 JSON with a sequences list")
    chain_to_entity = protein_entity_indices(payload)
    if not chain_to_entity:
        raise ValueError("Input contains no protein chains")
    files = sorted(path for path in template_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"Template directory contains no files: {template_dir}")
    total_bytes = 0
    used_entities: set[int] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    template_output_dir = output_dir / "templates"
    template_output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for upload_index, path in enumerate(files):
        suffix = path.suffix.lower()
        if suffix not in TEMPLATE_SUFFIXES:
            raise ValueError(f"Unsupported template format for {path.name!r}; use PDB or mmCIF")
        content = path.read_bytes()
        if not content or len(content) > MAX_TEMPLATE_FILE_BYTES:
            raise ValueError(f"Template {path.name!r} must be non-empty and at most 20 MiB")
        total_bytes += len(content)
        if total_bytes > MAX_TEMPLATE_TOTAL_BYTES:
            raise ValueError("Templates exceed the 50 MiB total upload limit")
        chain_id = path.stem
        entity_index = chain_to_entity.get(chain_id)
        if entity_index is None:
            raise ValueError(f"Template filename {path.name!r} must match an AF3 protein chain ID, such as A.pdb")
        if entity_index in used_entities:
            raise ValueError(f"More than one template was supplied for AF3 protein entity {chain_id!r}")
        protein = payload["sequences"][entity_index]["protein"]
        query = "".join(str(protein.get("sequence") or "").split()).upper()
        mmcif_text, template_sequence = load_single_chain_template(path.name, content)
        query_indices, template_indices = sequence_alignment_indices(query, template_sequence)
        relative = Path("templates") / f"af3-template-{entity_index}-{upload_index}.cif"
        (output_dir / relative).write_text(mmcif_text, encoding="utf-8")
        templates = protein.get("templates") or []
        if not isinstance(templates, list):
            raise ValueError("AF3 protein templates field must be a list")
        templates.append(
            {
                "mmcifPath": relative.as_posix(),
                "queryIndices": query_indices,
                "templateIndices": template_indices,
            }
        )
        protein["templates"] = templates
        used_entities.add(entity_index)
        count += 1
    payload["version"] = 2
    output_json = output_dir / "input.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"output_json": str(output_json), "template_count": count, "chains": sorted(chain_to_entity)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert per-chain PDB/mmCIF files into AF3 JSON templates with mmcifPath references."
    )
    parser.add_argument("--input-json", type=Path, required=True, help="AF3 JSON input with target protein sequences")
    parser.add_argument("--template-dir", type=Path, required=True, help="directory containing A.pdb, B.cif, etc.")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for input.json and converted template CIFs")
    args = parser.parse_args()
    try:
        result = prepare(args.input_json, args.template_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"af3-torch-prepare-templates: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
