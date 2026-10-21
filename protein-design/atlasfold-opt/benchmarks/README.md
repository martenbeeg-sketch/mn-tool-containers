# AtlasFold-M optimizer-kit benchmark

## Result

AtlasFold-M completed the full five-target heterodimer panel in all four modes on one RTX 4090. The predictions used the same input sequences, weights, deterministic seed set, recycle count, and sample count in every mode. The benchmark generated 300 scored structures (20 target/mode profiles × 15 structures), and all 20 profiles are registered in MN Cofolding.

`fast` had a median **2.12× forward speedup** over `off` across the five targets. `big` had a median **1.62× forward speedup** and used 12,876 MiB peak GPU memory on each target in this run; it was slower than `off` for the smallest target. `exact` was bitwise identical to `off` for every prediction output file across all five targets.

All five targets also completed in `off`, so this panel does **not** demonstrate a larger maximum sequence length or a capacity extension from `big`. It measures runtime, memory, and quality on complexes up to 959 total residues.

## Benchmark conditions

| Item | Value |
|---|---|
| Image | `mn-atlasfold-opt:1.0.0-cu128`, image ID `sha256:ce0d4152a7729df890e2e757b0d38f5fac7bac8c90c522747b5c62b463c54d3c` |
| Upstream | AtlasFold v1.0.0, commit `992067e67df29b665c501e0d2e9ead9dd4ba9b69` |
| Optimizer kit | AtlasFold kit 0.2.44, revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` |
| Hardware | RTX 4090, compute capability 8.9, driver 595.91.07; **GPU 0 only** |
| Stack | CUDA 12.8, PyTorch 2.7.1+cu128, cuEquivariance 0.10.0; shared core 0.5.228.0 |
| Weights used | AtlasFold-M SHA256 `210e3e9cfaac92d276ae44910d9b66d0673ecd5c225261e64dccf58c8dfe123a`; AtlasLM-3B SHA256 `8097cc4a234853f19ca97435844829d84fe7f13215c900e74659fa9682cc0ddb` |
| Prediction settings | `--det 1`, seeds 0–2, 5 samples/seed, 10 recycles, 200 diffusion steps; 15 outputs/profile |
| Input evidence | One FASTA record with chains joined by `:`. No external AA MSA or templates; AtlasFold-M uses AtlasLM sequence embeddings. |
| Cache and weights | JIT cache persisted under `/mnt/db/reference_files/mn-cofolding/cache/atlasfold/jit`; reference weights mounted read-only. |

The kit supports `--n_gpu 1` only; a value above one is explicitly refused by the kit. `big` is its low-memory mode, not a multi-GPU mode. RTX 5090 is still deployment-only and was not tested.

The first unadjusted optimized warmups were refused because `ln_bf16` received 192 calls and served zero (`all_fallback:fp32_params`). Following the kit's supported `MODEL_OPT_LEVERS_OFF` control, the benchmark disables only `ln_bf16` for `exact`, `fast`, and `big`; all other kit gates and runtime lever reports remain active. The final synthetic warmups completed: off 15.517 s, exact 14.599 s, fast 13.477 s, and big 13.576 s. Warmup time is excluded from profile timings.

The RTX 4090 kit/core reports include cells inherited from compute capability 8.0 or 9.0 where no 8.9 measurement exists. Each run records the actual lever route, activation, and fallback in its job log; no 5090 or universal kernel certification is implied.

## Timings and memory

Each cell gives **forward seconds / app wall seconds (forward speedup vs `off`)**. Wall time includes app resource acquisition, Docker startup, model load, inference, and output handling. Forward time is summed across the three seed batches. Peak memory is device-wide `nvidia-smi` usage sampled every 0.5 s on GPU 0.

| Target (total aa) | off | exact | fast | big |
|---|---:|---:|---:|---:|
| 8JZT (210) | 27.65 / 39.08 s (1.00×) | 28.22 / 40.08 s (0.98×) | 20.54 / 33.08 s (1.35×) | 32.15 / 44.08 s (0.86×) |
| 8WK1 (417) | 54.38 / 66.09 s (1.00×) | 52.03 / 65.09 s (1.05×) | 23.42 / 36.08 s (2.32×) | 39.49 / 52.08 s (1.38×) |
| 8H0N (611) | 81.24 / 94.10 s (1.00×) | 76.37 / 90.10 s (1.06×) | 35.40 / 48.09 s (2.30×) | 50.28 / 63.09 s (1.62×) |
| 8BJJ (788) | 149.08 / 162.12 s (1.00×) | 143.54 / 157.12 s (1.04×) | 70.38 / 84.10 s (2.12×) | 84.04 / 98.10 s (1.77×) |
| 7WR3 (959) | 193.40 / 207.13 s (1.00×) | 186.86 / 201.13 s (1.03×) | 93.06 / 107.11 s (2.08×) | 104.47 / 119.12 s (1.85×) |

| Target (total aa) | off peak MiB | exact peak MiB | fast peak MiB | big peak MiB |
|---|---:|---:|---:|---:|
| 8JZT (210) | 13,812 | 12,876 | 13,542 | 12,876 |
| 8WK1 (417) | 16,620 | 12,876 | 15,146 | 12,876 |
| 8H0N (611) | 18,916 | 12,876 | 14,302 | 12,876 |
| 8BJJ (788) | 23,428 | 12,876 | 18,958 | 12,876 |
| 7WR3 (959) | 21,920 | 12,876 | 19,926 | 12,876 |

## Quality

DockQ is OpenStructure DockQ-v2 against the native two-chain assembly. Values are the mean ± sample standard deviation over 15 structures. The final column counts structures with DockQ ≥ 0.23; a completed prediction can still have low DockQ.

| Target (total aa) | off | exact | fast | big |
|---|---:|---:|---:|---:|
| 8JZT (210) | 0.8698 ± 0.0129 (15/15) | 0.8698 ± 0.0129 (15/15) | 0.8685 ± 0.0134 (15/15) | 0.8693 ± 0.0135 (15/15) |
| 8WK1 (417) | 0.0404 ± 0.0007 (0/15) | 0.0404 ± 0.0007 (0/15) | 0.0404 ± 0.0007 (0/15) | 0.0404 ± 0.0007 (0/15) |
| 8H0N (611) | 0.7070 ± 0.0147 (15/15) | 0.7070 ± 0.0147 (15/15) | 0.7077 ± 0.0143 (15/15) | 0.7077 ± 0.0143 (15/15) |
| 8BJJ (788) | 0.1855 ± 0.0514 (0/15) | 0.1855 ± 0.0514 (0/15) | 0.1945 ± 0.0166 (1/15) | 0.1943 ± 0.0134 (0/15) |
| 7WR3 (959) | 0.0969 ± 0.0763 (0/15) | 0.0969 ± 0.0763 (0/15) | 0.0839 ± 0.0804 (0/15) | 0.0687 ± 0.0741 (0/15) |

For each target, the absolute fast/big shift in mean DockQ from `off` is smaller than the observed `off` sample standard deviation across its 15 outputs. This is a descriptive comparison on this panel, not a claim of statistical equivalence. 8WK1, 8BJJ, and 7WR3 had low DockQ in all modes; the no-external-MSA condition and target difficulty limit interpretation of their absolute accuracy.

Exact validation compared atom identities and unaligned coordinates for all 75 matching samples: maximum coordinate difference **0 Å**, with all atom keys matching. It also compared every file in each target's `off` and `exact` output directories by SHA256: **49/49 files identical per target**, including model CIFs, confidence JSON/NPZ files, ranked outputs, and summary CSVs.

## MN Cofolding and files

In MN Cofolding, open the FoldBench heterodimer dashboard. AtlasFold-M appears as an engine with separate `off`, `exact`, `fast`, and `big` rows, inference and wall timings, speedups relative to `off`, GPU memory, confidence values, DockQ scores, and links to the registered job results.

Benchmark artifacts are under `/mnt/data/RESULTS/mn-cofolding-workdir/workdir/benchmarks/foldbench-heterodimer-length-20261004/atlasfold/`:

- `benchmark_summary.json` — conditions, 20 profile records, job IDs, timing, memory, exact checks, and quality summaries.
- `profile_summary.csv` and `structure_scores.csv` — dashboard tables and per-structure DockQ/confidence values.
- `exact_coordinate_comparison.json` and `exact_output_artifact_comparison.json` — coordinate and output-file equality evidence.
- `warmups/{off,exact,fast,big}/` — kit commands, stdout/stderr lever reports, and warmup telemetry.

Registered jobs are recorded in the benchmark root's `registered_jobs.json`; the app loader was checked against these results and returned 20 completed AtlasFold profiles and 300 structures.
