# Protenix optimization candidate

**Status: experimental.** Keep `mn-protenix:cu128` available as the stock
fallback. The candidate has not passed the full quality and capacity release
gate.

This candidate adds the Protenix v2 kit's `off`, `exact`, `fast`, and `big`
modes on top of the pinned `mn-protenix:cu128` image. The stock image stays
available, and the candidate keeps its Protenix source, PyTorch, CUDA, Triton,
cuEquivariance, and checkpoint unchanged. `off` runs stock Protenix;
`exact` enables the kit's exact-preserving acceleration profile, `fast` uses
its faster numeric-tolerance profile, and `big` uses its memory-first profile.
For the kit's bitwise comparison recipe, pass `--det 1`. The earlier
lazy-initialization check was an implementation experiment, not an additional
user mode.

## Build

`./build.sh mn-protein-design` builds the stock image and then the candidate.
To build only the candidate, from `mn-tool-containers/`:

```bash
docker buildx build --load \
  --build-context protenixkit=../tools-to-implement/uplifting-biomolecular-modeling \
  -f protein-design/protenix-opt/Dockerfile \
  -t mn-protenix-opt:2.0.0-cu128 .
```

The candidate requires the local Protenix optimization kit checkout and the
`mn-protenix:cu128` base image. The build installs the kit's editable packages
and its autoload hook so `big` can launch its tensor-parallel ranks.

## Select a mode

To use it in the protein-design app, set these environment variables before
starting the app:

```bash
PROTENIX_CLI_IMAGE=mn-protenix-opt:2.0.0-cu128
PROTENIX_OPT_MODE=exact
```

`PROTENIX_OPT_MODE` defaults to `off`. The app rejects optimized modes if the
stock image is selected, and records the mode in the job parameters. `big` can
use the kit's row-pair tensor parallel path with two GPUs:

```bash
PROTENIX_CLI_IMAGE=mn-protenix-opt:2.0.0-cu128
PROTENIX_OPT_MODE=big
PROTENIX_OPT_N_GPU=2
```

For direct container use, add `-e PROTENIX_OPT_MODE=exact` (or `fast` or
`big`), mount the checkpoint tree at `/ref/protenix`, and mount a named volume at
`/opt/protenix-jit` if compiled kernels should persist between runs. Set
`PROTENIX_OPT_N_GPU=2` with `--gpus '"device=0,1"'` to select the row-pair
path. Multi-GPU mode is supported only by `big`; `exact` is single-GPU.

## RTX 4090 profile

The local stack is Protenix 2.0.0, PyTorch 2.7.1+cu128, Triton 3.3.1, and
cuEquivariance 0.10.0 on compute capability 8.9. The kit has no sm_89 cells for
`blk2_block_path`, `blk2_chunked_k2b`, or `k2b_flash_triattention`, and no
matching prebuilt fast-LayerNorm binary. On sm_89 the wrapper explicitly
ablates `blk2_block_path` and `fastln_prebuilt` in `exact`; it additionally
ablates the K2B-specific `blk2_chunked_k2b` and `k2b_flash_triattention` in
`fast` and `big`. LayerNorm uses the kit's source-built stream-correct CUDA
extension. All remaining selected levers are still subject to the kit's
fail-closed activation and end-of-run checks. Other GPU architectures do not
receive these automatic ablations.

The kit currently marks some transition and TriMul provider rows on sm_89 as
inherited from sm_80/A100 measurements (`stack_measured=False`). The 4090 run
proves those routes execute for the tested shapes; it does not establish their
speed or numerical quality across all shapes.

`fast` selects the kit's acceleration profile. `big` selects its memory-first
profile, including lazy relative-position features, memory releases, and
row-chunked diffusion conditioning and pair bias for inputs above 1,023 tokens.
It also lifts Protenix's stock token-count guard. The two-GPU path uses explicit
row-pair sharding. The app runs each candidate separately in that path because
the kit shards one inference at a time.

## Evidence and limits

The controlled, full-MSA RTX 4090 sweep is recorded in
[`benchmarks/optimization-playbook-sweep-2026-10-01.md`](benchmarks/optimization-playbook-sweep-2026-10-01.md),
with the attempt snapshot beside it. The raw inputs, logs, one-second NVML
samples, structures, confidence JSON files, and per-attempt records are under
`/mnt/data/RESULTS/protenix-optimization-playbook-20261001/`. The same records
drive the Protenix matrix and detail pages in `mn-cofolding`.

With the 8-target panel and identical full cached MSAs, `off` passed through
970 aa and OOMed at 1,391 aa; `fast` passed through 1,391 aa and OOMed at
1,927 aa; two-GPU row-pair `big` passed through 1,391 aa and OOMed at 1,927 aa.
The fail-fast rule left larger targets untested after each mode's first OOM.
`fast` lowered the median CHRNA7 model time from 31.6 s in candidate `off` to
18.0 s after warm-up. On CHRNA7 and SLC26A8, high-confidence regions aligned
closely after confidence-filtered superposition, while low-confidence regions
showed larger changes; see the report for both global and filtered fits.

`exact` was enabled after that sweep and has not yet been benchmarked on the
RTX 4090 stack. The report contains no `exact` timing, capacity, or structural
comparison results.

The earlier
[`benchmarks/off-fast-big-2026-10-01.md`](benchmarks/off-fast-big-2026-10-01.md)
is an initial implementation check. Its single-GPU 1,927-residue LCT prediction
used no MSA and is separate from the later full-MSA capacity sweep. Neither
result should be generalized to other inputs or treated as experimental
structure validation.

The candidate remains experimental because the long-target quality panel is
incomplete and modes differ in low-confidence regions. RTX 4090 is not the
kit's documented full fast-kernel composition; unsupported sm_89 kernels are
deliberately omitted and several provider measurements are inherited from
sm_80.

The copied kit code is Apache-2.0 licensed. See
[LICENSE.optimizer-kit](LICENSE.optimizer-kit).
