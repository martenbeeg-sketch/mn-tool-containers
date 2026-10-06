#!/usr/bin/env python3
"""Prepare AlphaFold 3 chemical-component tables, then run the preview CLI."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path


RUNNER = Path("/opt/alphafold3/run_alphafold.py")


def _cache_fingerprint(*values: object) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized_sequence(sequence: str) -> str:
    return "".join(str(sequence or "").split()).upper()


def _a3m_query_sequence(content: str) -> str:
    in_query = False
    query: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_query:
                break
            in_query = True
        elif in_query:
            query.extend(char for char in line if char.isupper())
    return "".join(query).replace("-", "")


def _read_shared_msa(sequence: str) -> str | None:
    root = Path(os.environ.get("AF3_SHARED_MSA_CACHE_DIR", "/msa_cache"))
    cleaned = _normalized_sequence(sequence)
    path = root / f"{hashlib.sha256(cleaned.encode('utf-8')).hexdigest()}.a3m"
    try:
        content = path.read_text(encoding="utf-8").rstrip("\x00").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in content or _a3m_query_sequence(content) != cleaned:
        return None
    return content


def _write_shared_msa(sequence: str, content: str) -> None:
    root = Path(os.environ.get("AF3_SHARED_MSA_CACHE_DIR", "/msa_cache"))
    cleaned = _normalized_sequence(sequence)
    if not cleaned or _a3m_query_sequence(content) != cleaned:
        return
    path = root / f"{hashlib.sha256(cleaned.encode('utf-8')).hexdigest()}.a3m"
    if path.is_file():
        return
    _atomic_write(path, content.rstrip("\x00").replace("\r\n", "\n"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_msa_cache() -> None:
    """Patch the AF3 preview MSA hook with an on-disk, cross-run cache."""
    import dataclasses

    from alphafold3.common import folding_input
    from alphafold3.data import msa_server

    cache_root = Path(os.environ.get("AF3_MSA_CACHE_DIR", "/msa_cache/mn-cofolding/alphafold3"))
    cache_root.mkdir(parents=True, exist_ok=True)
    original_query = msa_server._query_server

    def fill_missing_msas(fold_input, *, host_url="https://api.colabfold.com", user_agent="alphafold3/1.0"):
        protein_chains = fold_input.protein_chains
        rna_chains = fold_input.rna_chains
        missing_unpaired = list(dict.fromkeys(
            chain.sequence for chain in protein_chains if chain.unpaired_msa is None
        ))
        unique_paired = list(dict.fromkeys(
            chain.sequence for chain in protein_chains if chain.paired_msa is None
        ))
        seq_to_a3m: dict[str, str] = {}
        paired_seq_to_a3m: dict[str, str] = {}

        # One process-shared lock makes check/query/write atomic across jobs. This
        # avoids paying for the same remote MSA when two workers start together.
        lock_path = cache_root / ".msa-server.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            unpaired_dir = cache_root / "unpaired"
            missing = []
            for sequence in missing_unpaired:
                fingerprint = _cache_fingerprint("unpaired-v1", host_url, user_agent, True, sequence)
                path = unpaired_dir / f"{fingerprint}.a3m"
                if path.is_file():
                    seq_to_a3m[sequence] = path.read_text(encoding="utf-8")
                else:
                    shared = _read_shared_msa(sequence)
                    if shared is None:
                        missing.append(sequence)
                    else:
                        seq_to_a3m[sequence] = shared
                        _atomic_write(path, shared)
            if missing:
                print(f"Querying ColabFold MSA server for {len(missing)} uncached unique protein sequence(s)...")
                results = original_query(missing, host_url=host_url, user_agent=user_agent)
                if len(results) != len(missing):
                    raise RuntimeError("ColabFold MSA server returned an unexpected number of unpaired MSAs")
                for sequence, a3m in zip(missing, results):
                    fingerprint = _cache_fingerprint("unpaired-v1", host_url, user_agent, True, sequence)
                    _atomic_write(unpaired_dir / f"{fingerprint}.a3m", a3m)
                    _write_shared_msa(sequence, a3m)
                    seq_to_a3m[sequence] = a3m
                print("MSA query complete and cached.")
            elif missing_unpaired:
                print(f"Using cached MSA for {len(missing_unpaired)} unique protein sequence(s).")

            if len(unique_paired) > 1:
                fingerprint = _cache_fingerprint("paired-v1", host_url, user_agent, True, unique_paired)
                paired_path = cache_root / "paired" / f"{fingerprint}.json"
                if paired_path.is_file():
                    paired_seq_to_a3m = json.loads(paired_path.read_text(encoding="utf-8"))
                    print(f"Using cached paired MSA for {len(unique_paired)} unique protein sequence(s).")
                else:
                    print(f"Querying ColabFold for paired MSA ({len(unique_paired)} unique protein chains)...")
                    paired_results = original_query(
                        unique_paired, use_pairing=True, host_url=host_url, user_agent=user_agent
                    )
                    if len(paired_results) != len(unique_paired):
                        raise RuntimeError("ColabFold MSA server returned an unexpected number of paired MSAs")
                    paired_seq_to_a3m = dict(zip(unique_paired, paired_results))
                    _atomic_write(paired_path, json.dumps(paired_seq_to_a3m, sort_keys=True))
                    print("Paired MSA query complete and cached.")

        rna_needing_stub = [chain for chain in rna_chains if chain.unpaired_msa is None]
        if rna_needing_stub:
            print(
                f"Using query-sequence stub for {len(rna_needing_stub)} RNA chain(s)"
                " (ColabFold does not support RNA)."
            )
        if not missing_unpaired and not rna_needing_stub and not paired_seq_to_a3m:
            return fold_input

        new_chains = []
        for chain in fold_input.chains:
            if isinstance(chain, folding_input.ProteinChain):
                unpaired = chain.unpaired_msa
                paired = chain.paired_msa
                if unpaired is None:
                    unpaired = seq_to_a3m.get(chain.sequence, f">query\n{chain.sequence}\n")
                if paired is None:
                    paired = paired_seq_to_a3m.get(chain.sequence, "")
                chain = folding_input.ProteinChain(
                    id=chain.id,
                    sequence=chain.sequence,
                    ptms=chain.ptms,
                    description=chain.description,
                    unpaired_msa=unpaired,
                    paired_msa=paired,
                    templates=list(chain.templates) if chain.templates is not None else None,
                )
            elif isinstance(chain, folding_input.RnaChain):
                unpaired = chain.unpaired_msa
                if unpaired is None:
                    unpaired = f">query\n{chain.sequence}\n"
                chain = folding_input.RnaChain(
                    id=chain.id,
                    sequence=chain.sequence,
                    modifications=list(chain.modifications),
                    description=chain.description,
                    unpaired_msa=unpaired,
                )
            new_chains.append(chain)
        return dataclasses.replace(fold_input, chains=new_chains)

    msa_server.fill_missing_msas = fill_missing_msas


def _json_path(arguments: list[str]) -> Path | None:
    for index, value in enumerate(arguments):
        if value.startswith("--json_path="):
            return Path(value.partition("=")[2])
        if value == "--json_path" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
    return None


def _component_codes(value: object) -> set[str]:
    codes: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"ccdCodes", "ccd_codes"}:
                values = item if isinstance(item, list) else [item]
                codes.update(str(code).upper() for code in values if isinstance(code, str))
            elif key in {"ptmType", "modificationType", "ptm_type", "modification_type"} and isinstance(item, str):
                codes.add(item.removeprefix("CCD_").upper())
            else:
                codes.update(_component_codes(item))
    elif isinstance(value, list):
        for item in value:
            codes.update(_component_codes(item))
    return codes


def _prepare_ccd(input_path: Path | None) -> None:
    from alphafold3.constants import ccd_fetch
    import alphafold3

    extra_codes = _component_codes(json.loads(input_path.read_text(encoding="utf-8"))) if input_path else set()
    codes = ccd_fetch.codes_for_input(extra=extra_codes)
    cache_root = Path(os.environ.get("AF3_CCD_CACHE_DIR", "/cache/alphafold3/ccd"))
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256("\0".join(codes).encode()).hexdigest()[:24]
    cache_dir = cache_root / key
    lock_path = cache_root / f"{key}.lock"

    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        expected = (
            cache_dir / "ccd.pickle",
            cache_dir / "ccd.pickle.blobs",
            cache_dir / "ccd.pickle.index",
            cache_dir / "chemical_component_sets.pickle",
            cache_dir / "libcifpp" / "components.cif",
        )
        if not all(path.is_file() for path in expected):
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            temp_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
            try:
                ccd_fetch.write_pickles(
                    codes,
                    str(temp_dir / "ccd.pickle"),
                    str(temp_dir / "chemical_component_sets.pickle"),
                    libcifpp_dir=str(temp_dir / "libcifpp"),
                )
                temp_dir.replace(cache_dir)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        package_root = Path(alphafold3.__file__).resolve().parent
        destinations = {
            cache_dir / "ccd.pickle": package_root / "constants" / "converters" / "ccd.pickle",
            cache_dir / "chemical_component_sets.pickle": package_root / "constants" / "converters" / "chemical_component_sets.pickle",
            cache_dir / "ccd.pickle.blobs": package_root / "constants" / "converters" / "ccd.pickle.blobs",
            cache_dir / "ccd.pickle.index": package_root / "constants" / "converters" / "ccd.pickle.index",
            cache_dir / "libcifpp" / "components.cif": package_root / "../share/libcifpp/components.cif",
        }
        for source, target in destinations.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            temporary_link = target.with_name(f".{target.name}.{os.getpid()}.link")
            if temporary_link.exists() or temporary_link.is_symlink():
                temporary_link.unlink()
            temporary_link.symlink_to(source.resolve())
            os.replace(temporary_link, target)


def main() -> None:
    arguments = sys.argv[1:]
    input_path = _json_path(arguments)
    if input_path is not None:
        _prepare_ccd(input_path)
    elif "--help" in arguments or "-h" in arguments:
        _prepare_ccd(None)
    _install_msa_cache()
    sys.argv = [str(RUNNER), *arguments]
    runpy.run_path(str(RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
