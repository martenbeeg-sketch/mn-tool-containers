# Boltz2 MSA depth and long-sequence capacity

Run date: 2026-10-01  
Image: `mn-boltz2-opt:2.2.1-cu130`  
Hardware: one or two NVIDIA RTX 4090 GPUs, 24 GB each  
Results directory: `/mnt/data/RESULTS/boltz2-length-capacity-20261001/`

## MSA depth setting

Boltz2's input preprocessing defaults to `--max_msa_seqs 8192` per protein
chain. This is a software cap, not a guarantee that every sequence can run at
that depth on a given GPU. The model configuration has a separate optional
`num_subsampled_msa=1024` setting, but `subsample_msa` was false in these runs,
so the optional subsampling was inactive. The tested full-depth jobs used up to
8,192 rows; when an A3M contained fewer than 8,192 rows, all of its rows were
used. See the [Boltz CLI and model arguments](https://github.com/jwohlwend/boltz/blob/main/src/boltz/main.py).

## Test setup

- Canonical UniProt FASTA sequences and cached A3Ms from
  `/mnt/db/reference_files/msa_cache/`; each A3M query row was checked against
  its target sequence.
- Prediction settings: 3 recycling steps, 200 diffusion steps, one diffusion
  sample, one sample processed at a time, `--override`.
- One-GPU comparisons ran `off`, `fast`, and `big` on one RTX 4090. The large
  two-GPU tests used `big` with row-pair sharding across both RTX 4090s.
- A pass means the run exited successfully and wrote one CIF. It does not
  establish structural accuracy.
- The reduced-depth tests capped the input to the first N A3M rows; they did
  not randomly subsample or compare prediction quality.

## Full cached MSA results

The “MSA rows available” column is the raw cached A3M depth. The number actually
fed to Boltz2 is the smaller of that depth and the cap (8,192 for the full-depth
runs).

| Target | Length | MSA rows available | `off`, 1 GPU | `fast`, 1 GPU | `big`, 1 GPU | `big`, 2 GPUs |
|---|---:|---:|---|---|---|---|
| CHRNA7 | 502 aa | 13,331 | Pass, 17.5 s | Pass, 6.1 s | Pass, 7.1 s | — |
| SLC26A8 | 970 aa | 6,499 | Pass, 51.9 s | Pass, 16.1 s | Pass, 17.5 s | — |
| NUP155 | 1,391 aa | 2,312 | Pass, 94.7 s | Pass, 24.0 s | Pass, 25.7 s | — |
| LCT | 1,927 aa | 24,269 | OOM | OOM | OOM | Not run |
| EP300 | 2,414 aa | 18,102 | OOM | OOM | OOM | Pass, 475.5 s model phase; 541.2 s wall |
| CEP350 | 3,117 aa | 12,628 | OOM | OOM | OOM | Pass, 801.0 s model phase; 894.3 s wall |
| DMD | 3,685 aa | 19,394 | OOM | OOM | OOM | Pass, 1,061.1 s model phase; 1,176.5 s wall |
| PRKDC | 4,128 aa | 4,955 | OOM | OOM | OOM | Pass, 1,314.4 s model phase; 1,438.0 s wall |

The one-GPU full-depth run fed 8,192 rows for CHRNA7, LCT, EP300, CEP350, and
DMD; it fed the entire cached MSA for SLC26A8, NUP155, and PRKDC. Each two-GPU
large-target run used the entire cached MSA up to the 8,192-row cap. Peak
per-GPU NVML memory observed for the two-GPU runs was 14,365/16,108 MiB for
EP300, 18,469/20,386 MiB for CEP350, 22,013/23,856 MiB for DMD, and
21,771/23,210 MiB for PRKDC.

## Reduced-depth checks on one GPU

| Target | MSA cap | `off` | `fast` | `big` |
|---|---:|---|---|---|
| LCT, 1,927 aa | 1,024 | OOM | OOM | Pass |
| EP300, 2,414 aa | 1,024 | OOM | OOM | OOM |
| EP300, 2,414 aa | 512 | — | — | OOM |

Reducing MSA depth helped the 1,927-aa LCT only in `big` mode. It did not make
EP300 fit on one GPU, even at 512 rows. EP300 did complete at the full 8,192-row
cap with two GPUs in `big` mode, which shows that sequence length and the
execution mode/GPU layout matter alongside MSA depth.

## What this establishes

- **Software cap:** 8,192 MSA sequences per chain is the default input ceiling.
- **No universal GPU-safe MSA depth:** on one 24-GB RTX 4090, 6,499 rows worked
  for the 970-aa SLC26A8, while 8,192 rows did not fit for the 1,927-aa LCT.
  A 1,024-row cap let LCT complete in `big`, but the same reduction did not
  make EP300 fit.
- **Two-GPU `big` capacity:** with row-pair sharding, all four tested targets
  from 2,414 to 4,128 aa completed using their full cached MSA (capped at 8,192
  rows). PRKDC's MSA had 4,955 rows, so it was not truncated by the cap.
- **Runtime:** the 3,117–4,128-aa two-GPU predictions took 13–22 minutes of
  model phase time, excluding preprocessing and setup.

These are capacity results, not a recommendation to reduce MSA depth for
quality-sensitive predictions. Use the full cached MSA when it fits; lower
`--max_msa_seqs` only as a memory workaround, and record the cap with the run.
