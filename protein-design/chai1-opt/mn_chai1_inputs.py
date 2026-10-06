#!/usr/bin/env python3
"""Convert mounted Chai A3Ms to the kit's aligned parquet input format."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    args = list(argv)
    depth = 1024
    cleaned: list[str] = []
    input_path: Path | None = None
    index = 0
    while index < len(args):
        value = args[index]
        if value == "--msa-depth":
            if index + 1 >= len(args):
                raise SystemExit("--msa-depth requires a positive row count")
            depth = int(args[index + 1])
            index += 2
            continue
        if value.startswith("--msa-depth="):
            depth = int(value.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(value)
        if value == "--input" and index + 1 < len(args):
            input_path = Path(args[index + 1])
        elif value.startswith("--input="):
            input_path = Path(value.split("=", 1)[1])
        index += 1
    if depth < 1:
        raise SystemExit("--msa-depth must be positive")

    if input_path is not None and input_path.is_file() and input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        msa_dir_value = payload.get("msa_dir") if isinstance(payload, dict) else None
        if msa_dir_value:
            msa_dir = Path(str(msa_dir_value))
            if not msa_dir.is_absolute():
                msa_dir = (input_path.parent / msa_dir).resolve()
            if msa_dir.is_dir():
                from chai_lab.data.parsing.msas.aligned_pqt import (
                    a3m_to_aligned_dataframe,
                    expected_basename,
                )
                from chai_lab.data.parsing.msas.data_source import MSADataSource

                for a3m in sorted(msa_dir.glob("*.a3m")):
                    frame = a3m_to_aligned_dataframe(
                        a3m,
                        MSADataSource.UNIREF90,
                        insert_pairing_key=False,
                    )
                    if len(frame.index) > depth:
                        frame = frame.iloc[:depth].reset_index(drop=True)
                    query = str(frame.iloc[0]["sequence"]).upper()
                    target = msa_dir / expected_basename(query)
                    frame.to_parquet(target, index=False)
                    print(
                        f"[mn-chai1] MSA converted file={a3m.name} rows={len(frame.index)} depth_limit={depth} output={target.name}",
                        flush=True,
                    )

    os.execvp("bash", ["bash", "/kit/chai1/run.sh", *cleaned])
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
