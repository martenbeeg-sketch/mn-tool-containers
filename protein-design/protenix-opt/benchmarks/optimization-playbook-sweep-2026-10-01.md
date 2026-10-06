# Protenix optimization playbook capacity sweep

Run date: 2026-10-01  
Candidate: `mn-protenix-opt:2.0.0-cu128` (`sha256:947bc264d3ac0ce793ae706c56067b53f963bdad8b3d7962889fe96a1a2c3153`)  
Stock: `mn-protenix:cu128` (`sha256:67b9b1f4f6a8c07b8a5a1d1061ec1faa85b999bd22bd24d77dd005a0b9558fc6`)  
Hardware: two NVIDIA RTX 4090 GPUs, 24 GB each; driver 595.91.07; compute capability 8.9  
Raw results: `/mnt/data/RESULTS/protenix-optimization-playbook-20261001`

## Setup

Protenix 2.0.0 commit `c3bfc365b3e1341a11935eddfe7bfdc308092147`; PyTorch 2.7.1+cu128; Triton 3.3.1; cuEquivariance 0.10.0 / torch 0.8.0. The candidate preserves the stock Protenix source and core framework stack; it adds the matching Protenix v2 optimization kit at revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.

Weights: `Protenix v2 checkpoint protenix-v2.pt (stock/PINS.json checkpoint)`; SHA-256 `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`; 1,859,785,497 bytes, mounted read-only from `/mnt/db/reference_files/protenix/checkpoint/protenix-v2.pt`.

Prediction settings: Protenix v2, 10 cycles, 200 diffusion steps, 1 sample, seed 101, bf16, MSA enabled, templates disabled. Protenix's model MSA cap is 16,384 rows. Each cached A3M query was checked against its target sequence. The effective `N_msa` after Protenix deduplication is shown in each attempt record.

GPU assignment: stock and candidate `off`/`fast` used GPU 0; candidate `big` used explicit row-pair tensor parallelism across GPUs 0 and 1. GPU 1 retained the background monitor workload. NVML sampled both cards each second, including that background usage. The host topology reports both cards under the same NUMA node and no NVLink connection.

A pass requires exit success, a CIF that parses with the expected residue count, and a parseable confidence JSON. The original capacity sweep stopped each mode at its first prediction failure. Larger cells shown as skipped refer to that full-depth sweep; separate capped-depth follow-ups are reported below.

## Capacity sweep

| Target | Length | MSA available / cap | `off` · GPU 0 | `fast` · GPU 0 | `big` · GPUs 0,1 |
|---|---:|---:|---|---|---|
| CHRNA7 | 502 aa | 13,331 / 13,331 | Pass; 13248 rows; model 31.8 s; wall 109.9 s; peak 0:10936/1:1816 MiB | Pass; 13248 rows; model 18.0 s; wall 37.3 s; peak 0:9950/1:1757 MiB | Pass; 13248 rows; model 111.2 s; wall 153.0 s; peak 0:8090/1:8440 MiB |
| SLC26A8 | 970 aa | 6,499 / 6,499 | Pass; 6477 rows; model 105.5 s; wall 183.1 s; peak 0:19753/1:477 MiB | Pass; 6477 rows; model 72.4 s; wall 91.4 s; peak 0:18825/1:444 MiB | Pass; 6477 rows; model 323.6 s; wall 366.2 s; peak 0:8510/1:10330 MiB |
| NUP155 | 1,391 aa | 2,312 / 2,312 | OOM; 2303 rows; time to failure 99.4 s; peak 0:20821/1:454 MiB; no valid structure | Pass; 2303 rows; model 154.8 s; wall 175.1 s; peak 0:23787/1:450 MiB | Pass; 2303 rows; model 602.5 s; wall 643.0 s; peak 0:8749/1:7553 MiB |
| LCT | 1,927 aa | 24,269 / 16,384 | Not run — stopped after earlier failure | OOM; 16384 rows; time to failure 32.7 s; peak 0:23203/1:427 MiB; no valid structure | OOM; 16384 rows; time to failure 595.2 s; peak 0:24005/1:20326 MiB; no valid structure |
| EP300 | 2,414 aa | 18,102 / 16,384 | Not run — stopped after earlier failure | Not run — stopped after earlier failure | Not run — stopped after earlier failure |
| CEP350 | 3,117 aa | 12,628 / 12,628 | Not run — stopped after earlier failure | Not run — stopped after earlier failure | Not run — stopped after earlier failure |
| DMD | 3,685 aa | 19,394 / 16,384 | Not run — stopped after earlier failure | Not run — stopped after earlier failure | Not run — stopped after earlier failure |
| PRKDC | 4,128 aa | 4,955 / 4,955 | Not run — stopped after earlier failure | Not run — stopped after earlier failure | Not run — stopped after earlier failure |

The first OOMs in the original full-depth sweep were: `off` at NUP155 (1,391 aa); `fast` at LCT (1,927 aa); and two-GPU `big` at LCT (1,927 aa). `off` passed SLC26A8 (970 aa); both optimized modes passed through NUP155 (1,391 aa). `big` also passed CHRNA7 and SLC26A8. CEP350, DMD, and PRKDC were skipped in that original sweep after LCT failed in `big`; capped `off`/`fast` retries are listed separately below.

For the caught `off` NUP155 OOM, Protenix exited zero without a structure; the harness was corrected to classify OOMs from the log even when exit code is zero. The record preserves the CUDA allocation error and 99.4 s wall time. `fast` LCT failed in 32.7 s wall time. Two-GPU `big` LCT failed in 595.2 s wall time: a rank requested 5.68 GiB with 5.67 GiB free. NVML peaks were 24,005/20,326 MiB for GPUs 0/1. The initial `big` CHRNA7 launcher quoting error and the interrupted first NUP155 attempt are retained as separate non-prediction records; the corrected attempts passed.

## LCT MSA-depth follow-up

To test whether MSA depth caused the LCT limit, the same two-GPU `big` prediction was rerun with only the first 8,192 rows of the cached A3M. The query row still matched the target. All other prediction settings, including 10 cycles and 200 diffusion steps, were held fixed.

| MSA rows offered | Effective `N_msa` | GPUs | Result | Model s | Wall s | NVML peak MiB, GPUs 0/1 |
|---:|---:|---:|---|---:|---:|---|
| 16,384 | 16384 | 2 | oom; failed in MSA normalization, 5.68 GiB allocation | — | 595.2 | 0:24005/1:20326 MiB |
| 8,192 | 8192 | 2 | pass | 1115.4 | 1161.3 | 0:17665/1:15236 MiB |

At 8,192 rows Protenix reported `N_msa=8192`, completed, and wrote a parseable 1,927-residue CIF plus confidence JSON. This controlled result shows that lowering this LCT input from 16,384 to 8,192 rows was sufficient to avoid the observed OOM on this two-GPU setup. It is a capacity result; it does not establish that the reduced-depth prediction has equivalent structural quality.

## Capped MSA `off`/`fast` follow-up

The remaining one-GPU `off` and `fast` checks used the first `min(cached A3M rows, 8,192)` rows for each target. Shallower MSAs were left whole; rows were not randomly sampled. The query row was checked against the target, and all predictions retained 10 cycles and 200 diffusion steps. The table reports both rows offered and Protenix's effective `N_msa` after deduplication.

| Target | Mode | A3M rows offered | Effective `N_msa` | Result | Wall s | NVML peak MiB (GPU 0/1) |
|---|---|---:|---:|---|---:|---|
| NUP155 (1,391 aa) | `off` | 2,312 | 2,303 | CUDA OOM; requested 3.69 GiB, 3.19 GiB free | 100.1 | 0:20821/1:496 MiB |
| LCT (1,927 aa) | `off` | 8,192 | 8,192 | CUDA OOM; requested 3.54 GiB, 2.99 GiB free | 85.2 | 0:23457/1:491 MiB |
| LCT (1,927 aa) | `fast` | 8,192 | 8,192 | CUDA OOM; requested 1.77 GiB, 1.53 GiB free | 25.8 | 0:22517/1:491 MiB |
| EP300 (2,414 aa) | `off` | 8,192 | 8,191 | CUDA OOM; requested 3.02 GiB, 1.98 GiB free | 83.1 | 0:15877/1:475 MiB |
| EP300 (2,414 aa) | `fast` | 8,192 | 8,191 | CUDA OOM; requested 3.02 GiB, 2.01 GiB free | 25.3 | 0:15849/1:453 MiB |
| CEP350 (3,117 aa) | `off` | 8,192 | 8,121 | Rejected before model; `N_token=3117` exceeds v2 limit 2,560 | 87.6 | 0:4067/1:456 MiB |
| CEP350 (3,117 aa) | `fast` | 8,192 | 8,121 | Rejected before model; `N_token=3117` exceeds v2 limit 2,560 | 29.8 | 0:4061/1:453 MiB |
| DMD (3,685 aa) | `off` | 8,192 | 8,188 | Rejected before model; `N_token=3685` exceeds v2 limit 2,560 | 90.7 | 0:4067/1:491 MiB |
| DMD (3,685 aa) | `fast` | 8,192 | 8,188 | Rejected before model; `N_token=3685` exceeds v2 limit 2,560 | 33.8 | 0:4061/1:444 MiB |
| PRKDC (4,128 aa) | `off` | 4,955 | 4,891 | Rejected before model; `N_token=4128` exceeds v2 limit 2,560 | 92.7 | 0:4067/1:445 MiB |
| PRKDC (4,128 aa) | `fast` | 4,955 | 4,891 | Rejected before model; `N_token=4128` exceeds v2 limit 2,560 | 36.3 | 0:4061/1:444 MiB |

Capping the MSA did not make the one-GPU `off` or `fast` LCT/EP300 attempts fit; those runs still hit CUDA OOM. NUP155 `off` also OOMed at its original 2,312-row depth. CEP350, DMD, and PRKDC were rejected before model forward because their token counts exceeded Protenix v2's built-in 2,560-token limit. Those are input-length rejections, not GPU-memory OOM measurements; the depth cap does not change token count. The 2-GPU `big` results use a separate execution path and are reported in the capacity and LCT sections above. This follow-up measures capacity only and does not compare structural quality at reduced MSA depth.

## CHRNA7 cold and warm timing

CHRNA7 was run once as the first attempt and three more times after the first attempt per image/mode. Each attempt used a fresh container and process. The `fast` and `big` attempts shared persistent Docker volumes for compiled JIT artifacts, so their later runs reused compiled kernels; model weights were reloaded for every attempt. Model seconds use the slowest rank for tensor parallel runs; wall seconds include model startup, preprocessing, inference, output, and CIF validation. The candidate cold and warm runs used 10 cycles, but the stock cold record used 3 cycles while its warm records used 10, so the stock cold/warm row is not a controlled comparison.

| Image/mode | Cold model s | Cold wall s | Warm runs | Median warm model s | Median warm wall s |
|---|---:|---:|---:|---:|---:|
| stock | 31.4 | 109.0 | 3 | 31.3 | 109.1 |
| off | 31.4 | 108.9 | 3 | 31.6 | 109.7 |
| fast | 29.4 | 173.3 | 3 | 18.0 | 37.3 |
| big | 121.4 | 167.6 | 3 | 111.2 | 153.0 |

On CHRNA7, candidate `off` matches the stock warm runtime. `fast` reduced median model time from about 31.6 s (`off`) to 18.0 s. Cold `fast` wall time includes the first LayerNorm extension build/JIT cost. Two-GPU `big` is a capacity mode and is slower on this short target.

## Structural and confidence comparison

Same-seed mode outputs were superposed by matching Cα residue keys. These checks compare the implementation outputs and repeatability; they do not measure agreement with an experimental structure. Confidence scores are model-native scores, not independent accuracy validation.

### CHRNA7

Baseline mode: `stock`; 502 Cα atoms.

| Comparison | Pairs | Median RMSD (Å) | Range (Å) |
|---|---:|---:|---:|
| stock repeats | 6 | 0.062 | 0.036–0.088 |
| off repeats | 6 | 0.075 | 0.042–0.126 |
| fast repeats | 6 | 0.051 | 0.037–0.087 |
| big repeats | 6 | 0.102 | 0.089–0.129 |
| off vs stock | 16 | 0.056 | 0.032–0.157 |
| fast vs stock | 16 | 0.092 | 0.075–0.140 |
| big vs stock | 16 | 14.435 | 14.390–14.450 |

Median confidence metrics:

```json
{
  "stock": {
    "plddt": 80.63029479980469,
    "gpde": 0.5838535726070404,
    "ptm": 0.6673065423965454,
    "iptm": 0.0,
    "ranking_score": 0.13346131145954132
  },
  "off": {
    "plddt": 80.63994216918945,
    "gpde": 0.5835455060005188,
    "ptm": 0.667631596326828,
    "iptm": 0.0,
    "ranking_score": 0.13352631777524948
  },
  "fast": {
    "plddt": 80.16798400878906,
    "gpde": 0.5983690619468689,
    "ptm": 0.6633541882038116,
    "iptm": 0.0,
    "ranking_score": 0.1326708346605301
  },
  "big": {
    "plddt": 80.83304595947266,
    "gpde": 0.5806997120380402,
    "ptm": 0.6623494625091553,
    "iptm": 0.0,
    "ranking_score": 0.13246989995241165
  }
}
```


### SLC26A8

Baseline mode: `off`; 970 Cα atoms.

| Comparison | Pairs | Median RMSD (Å) | Range (Å) |
|---|---:|---:|---:|
| fast vs off | 1 | 12.858 | 12.858–12.858 |
| big vs off | 1 | 38.484 | 38.484–38.484 |

Median confidence metrics:

```json
{
  "stock": {},
  "off": {
    "plddt": 75.77989959716797,
    "gpde": 0.7409395575523376,
    "ptm": 0.6302664279937744,
    "iptm": 0.0,
    "ranking_score": 0.12605328857898712
  },
  "fast": {
    "plddt": 76.02830505371094,
    "gpde": 0.7409864664077759,
    "ptm": 0.6287109851837158,
    "iptm": 0.0,
    "ranking_score": 0.12574219703674316
  },
  "big": {
    "plddt": 76.14437866210938,
    "gpde": 0.7536138892173767,
    "ptm": 0.6257773637771606,
    "iptm": 0.0,
    "ranking_score": 0.1251554787158966
  }
}
```


### Confidence-stratified fits

B-factors in the predicted mmCIF encode per-atom pLDDT. The fit uses residues with B-factor/pLDDT ≥70 in both structures.

| Target | Comparison | All-residue global fit (Å) | High-confidence fit, high-confidence residues (Å) | High-confidence-fit residual, other residues (Å) | High-confidence Cα count |
|---|---|---:|---:|---:|---:|
| CHRNA7 | fast vs stock | 0.075 | 0.036 | 0.170 | 402 |
| CHRNA7 | big vs stock | 14.432 | 1.124 | 33.057 | 401 |
| SLC26A8 | fast vs off | 12.858 | 0.097 | 23.355 | 646 |
| SLC26A8 | big vs off | 38.484 | 0.500 | 70.100 | 648 |

The full-chain RMSDs for SLC26A8 and CHRNA7 are driven mainly by low-confidence regions and their relative placement. After fitting on residues with per-atom pLDDT ≥70, the high-confidence regions align closely across modes (0.10 Å for SLC26A8 `fast` vs `off`; 0.50 Å for `big` vs `off`; 1.12 Å for CHRNA7 `big` vs stock). Report both fits; do not infer global structural equivalence from confidence scores alone.

## Interpretation and limits

- `fast` was faster than `off` on CHRNA7 and SLC26A8, and completed NUP155 on one GPU where `off` OOMed. Its 1,391-aa NUP155 run reached 23,787 MiB sampled on GPU 0, leaving little memory headroom.
- `big` explicitly row-pair-sharded a single prediction over both RTX 4090 cards. Full-cap LCT at 1,927 aa OOMed in MSA normalization with 16,384 rows, while the same prediction passed with 8,192 rows. This shows MSA depth was a limiting factor for this LCT run; it does not give a universal safe MSA depth or validate reduced-depth quality.
- On RTX 4090, the kit's unsupported sm_89 kernel routes were explicitly ablated. Several selected transition/TriMul provider choices inherit measurements from sm_80; successful execution on these runs does not establish their speed on every 4090 shape.
- The candidate remains experimental. The benchmark does not pass the release gate for general use: long-target quality is only sampled on short/medium proteins, most combinations skipped by the original stop rule remain untested, and the global structures show low-confidence-region differences.

## Reproducibility and outputs

The report generator snapshots all 43 structured attempt records in [optimization-playbook-sweep-2026-10-01.json](optimization-playbook-sweep-2026-10-01.json). Each record contains the command, image digest, upstream and kit pins, input/MSA hashes, settings, phase timings, effective MSA rows, exit/status, GPU assignment and sampled peaks, output parser results, confidence metrics, and raw artifact paths. Raw run logs, NVML CSVs, inputs, outputs, and per-attempt records remain under `/mnt/data/RESULTS/protenix-optimization-playbook-20261001`.

The app summary is generated from the same records and links each `off`/`fast`/`big` cell to its full attempt page in `mn-cofolding`.
