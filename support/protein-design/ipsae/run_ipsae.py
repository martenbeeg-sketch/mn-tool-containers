#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


HEADER = [
    "ID",
    "ipsae",
    "ipsae_d0chn",
    "ipsae_d0dom",
    "iptm_af",
    "iptm_d0chn",
    "binder_chain",
    "target_chain",
    "type",
    "engine",
    "pae_format",
    "error",
]


def _empty_row(args, error_msg: str) -> dict:
    return {
        "ID": args.id,
        "ipsae": "",
        "ipsae_d0chn": "",
        "ipsae_d0dom": "",
        "iptm_af": "",
        "iptm_d0chn": "",
        "binder_chain": args.binder_chain,
        "target_chain": args.target_chain,
        "type": "",
        "engine": args.engine,
        "pae_format": args.pae_format,
        "error": error_msg,
    }


def _find_summary_file(structure: Path, pae_cutoff: float, dist_cutoff: float) -> Path | None:
    stem = structure.with_suffix("")
    pae_str = f"{int(pae_cutoff):02d}"
    dist_str = f"{int(dist_cutoff):02d}"

    candidates = [stem.parent / f"{stem.name}_{pae_str}_{dist_str}.txt"]
    if structure.suffix.lower() in {".cif", ".pdb"}:
        candidates.append(structure.parent / f"{structure.stem}_{pae_str}_{dist_str}.txt")

    for c in candidates:
        if c.exists():
            return c

    globbed = sorted(structure.parent.glob(f"{structure.stem}_*_*.txt"), key=os.path.getmtime, reverse=True)
    return globbed[0] if globbed else None


def _parse_summary(summary_path: Path, binder_chain: str, target_chain: str) -> dict | None:
    with summary_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    header_idx = None
    headers = []
    for i, line in enumerate(lines):
        if line.startswith("Chn1") and "ipSAE" in line and "Type" in line:
            header_idx = i
            headers = re.split(r"\s+", line)
            break

    if header_idx is None:
        return None

    rows = []
    for line in lines[header_idx + 1 :]:
        if line.startswith("#"):
            continue
        cols = re.split(r"\s+", line)
        if len(cols) < len(headers):
            continue
        row = dict(zip(headers, cols))
        if "Chn1" not in row or "Chn2" not in row or "ipSAE" not in row:
            continue
        rows.append(row)

    if not rows:
        return None

    def score(r):
        # prefer explicit max type, then asym, then other
        t = (r.get("Type") or "").lower()
        pri = 0 if t == "max" else (1 if t == "asym" else 2)
        try:
            v = float(r.get("ipSAE", "nan"))
        except ValueError:
            v = float("-inf")
        return (pri, -v)

    pair_rows = [
        r
        for r in rows
        if (r.get("Chn1") == binder_chain and r.get("Chn2") == target_chain)
        or (r.get("Chn1") == target_chain and r.get("Chn2") == binder_chain)
    ]
    use_rows = pair_rows if pair_rows else rows
    best = sorted(use_rows, key=score)[0]

    return {
        "ipsae": best.get("ipSAE", ""),
        "ipsae_d0chn": best.get("ipSAE_d0chn", ""),
        "ipsae_d0dom": best.get("ipSAE_d0dom", ""),
        "iptm_af": best.get("ipTM_af", ""),
        "iptm_d0chn": best.get("ipTM_d0chn", ""),
        "type": best.get("Type", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--pae", required=True)
    parser.add_argument("--pae-format", default="")
    parser.add_argument("--engine", default="")
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--target-chain", default="B")
    parser.add_argument("--pae-cutoff", type=float, default=10)
    parser.add_argument("--dist-cutoff", type=float, default=15)
    parser.add_argument("--ipsae-script", default="/opt/ipsae/ipsae.py")
    parser.add_argument("--output", default="ipsae_row.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    structure = Path(args.structure)
    pae = Path(args.pae)

    row = None
    try:
        cmd = [
            "python3",
            args.ipsae_script,
            str(pae),
            str(structure),
            str(args.pae_cutoff),
            str(args.dist_cutoff),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "ipsae.py failed").strip().replace("\n", " | ")
            row = _empty_row(args, err)
        else:
            summary = _find_summary_file(structure, args.pae_cutoff, args.dist_cutoff)
            if not summary:
                row = _empty_row(args, "ipSAE summary file not found")
            else:
                parsed = _parse_summary(summary, args.binder_chain, args.target_chain)
                if not parsed:
                    row = _empty_row(args, f"could not parse summary: {summary}")
                else:
                    row = {
                        "ID": args.id,
                        "ipsae": parsed["ipsae"],
                        "ipsae_d0chn": parsed["ipsae_d0chn"],
                        "ipsae_d0dom": parsed["ipsae_d0dom"],
                        "iptm_af": parsed["iptm_af"],
                        "iptm_d0chn": parsed["iptm_d0chn"],
                        "binder_chain": args.binder_chain,
                        "target_chain": args.target_chain,
                        "type": parsed["type"],
                        "engine": args.engine,
                        "pae_format": args.pae_format,
                        "error": "",
                    }
    except Exception as exc:
        row = _empty_row(args, f"exception: {exc}")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        writer.writerow(row)


if __name__ == "__main__":
    main()
