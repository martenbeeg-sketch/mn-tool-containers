# Protenix `off`, `fast`, and `big` implementation checks

Run date: 2026-10-01  
Candidate image: `mn-protenix-opt:2.0.0-cu128` (`sha256:947bc264d3ac0ce793ae706c56067b53f963bdad8b3d7962889fe96a1a2c3153`)  
Base image: `mn-protenix:cu128` (`sha256:67b9b1f4f6a8c07b8a5a1d1061ec1faa85b999bd22bd24d77dd005a0b9558fc6`)  
Hardware: NVIDIA RTX 4090, 24 GB; host has two GPUs  
Results directory: `/mnt/data/RESULTS/protenix-optimization-20261001/`

This is the initial implementation-check snapshot. The later controlled,
full-MSA stock/`off`/`fast`/two-GPU `big` sweep and its updated capacity
outcomes are in
[`optimization-playbook-sweep-2026-10-01.md`](optimization-playbook-sweep-2026-10-01.md).

## Modes and runtime

The image and protein-design workflow expose three values: `off`, `fast`, and
`big`. `off` delegates to stock Protenix. `fast` selects the kit acceleration
profile. `big` selects its memory-first profile and permits the row-pair
tensor-parallel route. The wrapper rejects other names and rejects multiple
GPUs unless `big` is selected.

The image is based on Protenix 2.0.0, Python 3.11, PyTorch 2.7.1+cu128,
Triton 3.3.1, and cuEquivariance 0.10.0 / torch 0.8.0. The optimizer kit
checkout was at `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. The checkpoint is
mounted from the host and is not baked into the image.

On RTX 4090 (compute capability 8.9), the wrapper explicitly disables four
kit levers without matching sm_89 cells:
`blk2_block_path`, `blk2_chunked_k2b`, `k2b_flash_triattention`, and
`fastln_prebuilt`. The stream-correct LayerNorm extension is built from source
on its first use. It is stored under `/opt/protenix-jit`; mounting the named
`mn-protenix-opt-jit` volume let a later run reuse it (`ninja: no work to do`).
The other selected kit routes were reconciled at the end of successful runs.

## Checks and predictions

| Mode / check | Input and configuration | Result |
|---|---|---|
| `off` | Candidate entrypoint `protenix --help` | Exit 0; stock CLI help was returned. |
| `fast` packaged smoke | Kit warm input, 76 tokens, one recycle, two diffusion steps, one sample | One CIF; model-forward 3.62 s; final gate `partial=false`. |
| `fast` packaged prediction | 1BRS barnase:barstar, 199 tokens, one sample, query-only MSA | First run: model-forward 8.58 s and compiled stream LayerNorm. Repeat with the same JIT volume: model-forward 4.96 s, wall 22.89 s, and Ninja reused the extension. Both final gates reported `partial=false`; CIF and confidence JSON parsed. These timings include different cold/warm setup and are not a controlled speed comparison. |
| `big` single-GPU capacity (no MSA; separate from the later cached-MSA sweep) | LCT P09848, 1,927 residues, one sample, 10 cycles, 200 diffusion steps, bf16 | One CIF and confidence JSON; model-forward 817.09 s, wall 976 s, maximum sampled GPU memory 23,230 MiB. Gemmi parsed one 1,927-residue chain. Final gate `partial=false`, with no fallback. |
| `big`, two-GPU preflight | RTX 4090 pair; `check --json`, `n_gpu=2` | Exit 0; row-pair sharding selected, 24 unit files, NCCL 2.26.2. This was a preflight only; no two-GPU prediction timing is reported. |

The long `big` prediction exercised the kit's single-GPU CLI directly with the
same memory-first configuration; it was not launched through the candidate
container entrypoint. The packaged entrypoint separately passed its `big`
preflight. This confirms the long-target memory path but is not a packaged
long-target end-to-end check.

## Interpretation and limits

These runs verify that the mode selector routes work, selected levers activate,
outputs are produced, and the 1,927-residue target fits on one RTX 4090 in
`big`. They do not establish structural accuracy or numerical equivalence to
stock predictions. No controlled stock-versus-optimized speed benchmark,
quality comparison, or actual two-GPU prediction was completed here.

Some transition and TriMul provider rows on sm_89 are inherited from sm_80
measurements (`stack_measured=False`). Successful execution on the tested
shapes does not establish their speed or quality for other inputs. See
[`off-fast-big-2026-10-01.json`](off-fast-big-2026-10-01.json) for the compact
run manifest and `/mnt/data/RESULTS/protenix-optimization-20261001/` for raw
logs and outputs.
