# Historical Protenix v2 lazy-init experiment: initial RTX 4090 comparison

Run date: 2026-10-01  
Stock image: `mn-protenix:cu128` (`sha256:67b9b1f4f6a8c07b8a5a1d1061ec1faa85b999bd22bd24d77dd005a0b9558fc6`)  
Candidate image: `mn-protenix-opt:2.0.0-cu128` (`sha256:35407e895fe1c20ed019e98291e280db46f87cb25e7aae1b4b0efc60d4719650`)  
Stock source: Protenix 2.0.0, commit `c3bfc365b3e1341a11935eddfe7bfdc308092147`  
Runtime: Python 3.11, torch 2.7.1+cu128, Triton 3.3.1, cuEquivariance 0.10.0 / torch 0.8.0  
Hardware: one NVIDIA RTX 4090, 24 GiB, driver 595.91.07, compute capability 8.9  
Output and raw logs: `/mnt/data/RESULTS/protenix-optimization-20261001/stock-p199/`
Packaged-image outputs and logs: `/mnt/data/RESULTS/protenix-optimization-20261001/candidate-image/`

## Input and settings

- Kit-supplied 1BRS barnase:barstar, 199 tokens, chain lengths 110 and 89.
- Both cached A3Ms have one row, the query sequence. Query rows were checked
  against the input sequences.
- `protenix pred --seeds 101 --model_name protenix-v2 --cycle 10 --step 200
  --sample 5 --dtype bf16 --use_msa True --use_template False
  --use_default_params True --need_atom_confidence True`.
- The same checkpoint and input were used in both arms. The checkpoint is
  mounted from `/mnt/db/reference_files/protenix`; it is not in the image.

## Results

| Run | Mode / condition | Wall s | Model forward s | Peak GPU MiB | Output |
|---|---|---:|---:|---:|---|
| Stock baseline | stock defaults | 92.57 | 15.04 | 4,512 | 5 CIFs + confidence JSON |
| Lazy-init candidate | stock defaults + `PTX_LAZY_INIT=1` | 27.28 | 14.12 | 4,512 | 5 CIFs + confidence JSON |
| Stock reference | deterministic recipe | 95.33 | 20.14 | 4,258 | 5 CIFs + confidence JSON |
| Lazy-init candidate | same deterministic recipe + `PTX_LAZY_INIT=1` | 33.37 | 20.62 | 4,574 | 5 CIFs + confidence JSON |

The earlier packaged experiment was run end to end with lazy initialization
enabled and disabled. It wrote five CIFs and five summary-confidence JSON files
in each run; Gemmi parsed all CIFs, and all summary JSON files parsed. The
disabled log reported `mode=off implementation=stock`; the experimental
enabled log reported `lever=lazy_init state=active`. That prototype called its
enabled setting `exact`; that label is retired and is not a selectable mode in
the current image.

| Image path | Historical condition | Wall s, three runs | Model forward s, three runs | Median wall s |
|---|---|---|---|---:|
| Stock image plus candidate image `off` | Stock execution | 92.57, 89.93, 94.54 | 15.04, 14.09, 13.99 | 92.57 |
| Candidate image | Lazy initialization enabled | 28.08, 27.34, 27.06 | 14.44, 14.22, 14.33 | 27.34 |

For this short input, the candidate reduced median end-to-end wall time by
70.5% (3.39× faster), while median model-forward time remained similar. The
gain is from CPU-side construction; this lever does not speed up the GPU model
forward pass. The candidate image was built from the local stock tag and has
the image ID recorded above.

In the strict lazy-init recheck, stock model construction took 96.51 s and
lazy construction took 0.48 s. After strict checkpoint loading, all 4,174
parameters and buffers hashed identically. In the deterministic prediction
comparison, all five CIF files were byte-identical and all five confidence
JSON files were equal. The deterministic recipe used `CUBLAS_WORKSPACE_CONFIG`,
`torch.use_deterministic_algorithms(True, warn_only=True)`, and the optimizer
kit's deterministic scatter implementation in both arms.

The default stock path is itself not repeatable at this workload with the same
seed: two stock runs produced aligned per-sample C-alpha RMSDs of 6.27–12.87 Å.
The non-deterministic lazy run differed from the first stock run by 5.08–17.03
Å. These default-mode values are not an exactness test; exactness was assessed
under the deterministic recipe above.

## Scope

This historical report validates one startup lever on one short input and one
RTX 4090. The input has only query MSAs, so it does not characterize MSA-heavy
runtime or quality. It does not establish long-sequence capacity, memory
reduction, speed of custom GPU kernels, or 5090 support. The current image
exposes `off`, `fast`, and `big`; this earlier experiment is not a fourth mode.
The run manifest is [`initial-lazy-init-2026-10-01.json`](initial-lazy-init-2026-10-01.json).
