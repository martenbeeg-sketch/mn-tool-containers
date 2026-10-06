# Playbook for optimizing model containers

This guide describes how to make a separate, testable optimized image for a
protein-modeling engine while keeping its current stock image available. It
draws on the local `mn-boltz2-opt` and AF3 JAX memory work, and on the reference
kits in [`tools-to-implement/uplifting-biomolecular-modeling`](../tools-to-implement/uplifting-biomolecular-modeling/README.md).

The reference kits are pinned, as-is examples, not a maintained compatibility
layer. Their shared runtime and kernels do not make every engine interchangeable:
each kit has its own pinned upstream version, integration points, supported
modes, weights, framework versions, and GPU guards. Port only after matching
the kit's `STOCK.md` to the source and dependencies in the target image.

## Engine recipe and profile policy

For every new folding engine, keep one known stock image and one active
optimized image. The optimized image is the implementation under test; runtime
choices such as `off`, `exact`, `fast`, and `big` remain modes inside that image.
Do not keep several experimental optimized images active in the app catalog.

Use an engine-local layout like this, under `protein-design/` or `shared/`
depending on ownership:

```text
engine/
  Dockerfile                 # stock image; never replace it in place
  optimized/
    Dockerfile               # one active optimized recipe
  profiles/
    rtx4090.env              # Ada sm_89 deployment/validation profile
    rtx5090.env              # Blackwell sm_120 deployment/validation profile
  archive/                   # superseded candidates, never deleted silently
  benchmarks/                # manifests and measured reports
```

The two GPU profiles do not imply two images. They record the GPU architecture,
memory, driver/framework constraints, build targets, runtime defaults, kernel
gates, and validation status for the same recipe. A profile may be marked
`deployment-only` until it has completed a real hardware run. Keep the stock
recipe available for every profile, and never combine two networked machines
into one multi-GPU prediction; multi-GPU mode is valid only when the engine
explicitly shards one prediction across GPUs in the same host.

When an experiment is superseded, move its recipe and benchmark evidence into
`archive/` and remove it from the active image catalog/build command. Preserve
it for provenance and possible reactivation, but make the normal app path
unambiguous: one stock image, one active optimized image, and two hardware
profiles.

## What the modes mean

Use a consistent user-facing contract, while allowing each engine to implement
only modes it can support and validate:

| Mode | Intended behavior | What success means |
|---|---|---|
| `off` | Run the pinned upstream implementation. | A baseline for speed, memory, output, and errors. |
| `fast` | Enable measured kernels or execution changes that improve speed. | Faster on the named hardware and workload; any numeric changes are documented and remain within an agreed validation range. |
| `big` | Reduce peak memory so larger inputs may fit, often with chunking or sharding. | A target that fails for memory in `off` completes. It can be slower; speed is not its purpose. |
| `exact` (optional) | Use an optimization intended to preserve stock arithmetic. | Equivalence is demonstrated under the kit's documented deterministic conditions. |

Do not call a run `big` merely because it uses more GPUs, or `fast` because a
kernel is fused. For multi-GPU `big`, the model must explicitly partition a
single prediction's tensors/work across GPUs. Launching separate jobs on each
GPU does not combine their memory.

Each mode must identify what actually ran. Log the selected mode, active
optimizations, inactive or refused optimizations, GPU model, software pins, and
reason for any fallback. Do not silently label a stock fallback as `fast` or
`big`. A safe per-kernel fallback can be acceptable if it is reported; a strict
benchmark should fail when a required optimization was not active.

## RTX 4090 host profile

The tested workstation has two RTX 4090 cards with 24 GB each. Treat them as two
separate 24 GB devices, not one 48 GB memory pool. A two-GPU prediction helps
capacity only when that engine implements and validates sharding or another
explicit distributed layout. Communication over the host interconnect can also
make two-GPU runs slower than one-GPU runs. The topology captured on this host
reports the two cards on the same PCIe/NUMA node; do not assume NVLink bandwidth.

Before a benchmark, record `nvidia-smi -L`, driver version, each card's free and
total memory, topology (`nvidia-smi topo -m`), and the framework's reported
compute capability. RTX 4090 is Ada, compute capability 8.9 (`sm_89`). Check
that every custom kernel, extension, and prebuilt binary has a working `sm_89`,
PTX, or JIT path. An image's CUDA tag alone does not certify RTX 4090 support.
For example, the current OpenFold3 recipe lists `TORCH_CUDA_ARCH_LIST` as
`9.0;10.0;12.0+PTX`; inspect its actual runtime behavior on `sm_89` before
assuming it is a good 4090 baseline.

Keep persistent, writable JAX/XLA, Triton, and TorchInductor caches on the host
when supported by the engine. The first call can pay compilation and autotuning
costs; separate cold-start timing from warm inference timing. Cache entries may
be GPU, driver, framework, or shape specific. Do not copy a cache from another
GPU or software stack and count it as a validated run.

## RTX 5090 host profile

The deployment targets are two separate networked hosts, each with one RTX
5090, 32 GB of VRAM, compute capability 12.0 (`sm_120`), and the current
595.91.07 driver stack. Treat them as independent single-GPU machines. They
cannot be combined into one inference or one 64 GB memory pool over SSH or the
network. The engine-local `profiles/rtx5090.env` files record this target and
remain marked unvalidated until each active optimized image completes its smoke,
numerical, and capacity checks on a real 5090.

The image may contain both `sm_89` and `sm_120` support, but that is still one
image recipe. Keep the GPU-specific runtime settings, kernel gates, and
validation status in the profile; promote a new optimized recipe only when the
engine's implementation itself changes.

## Staged workflow

### 1. Inventory the target and its calling path

For the image and app, record:

- image name, Dockerfile, base image, upstream repository and exact commit or
  release, Python/framework/CUDA versions;
- command actually called by the app, wrapper environment variables, input and
  output formats, and how weights/reference files/MSAs are mounted;
- where preprocessing ends and model inference begins, including whether MSA
  search, templates, relaxation, or postprocessing run in the same process;
- current memory settings, precision, attention provider, chunk sizes, and
  accelerator defaults;
- license and terms for source, weights, and outputs. Keep user-provided or
  licensed weights outside the image unless there is a clear, permitted reason
  to bake them in.

Read the matching optimizer kit's `README.md`, `STOCK.md`, `CHANGES.md`, and
configuration for the intended GPU. Compare its pinned upstream source and
dependency lock with the local image. A kit for another release is a source of
ideas, not a drop-in patch. Verify every proposed hook against the exact local
code and fail closed if the expected code or shape has changed.

### 2. Preserve the known image and make one active candidate

Leave the stock Dockerfile, tag, and behavior intact. Add one optimized recipe
under the engine directory, for example
`protein-design/protenix/optimized/Dockerfile`, and use a distinct tag, for
example `mn-protenix-opt:<upstream>-cu128`. Pin the same source revision,
weights, inputs, and core dependencies for the first stock-versus-candidate
comparison. Avoid changing CUDA, PyTorch, the model revision, and optimization
code all at once: if the outcome changes, you need to know why.

Create `profiles/rtx4090.env` and `profiles/rtx5090.env` beside the recipe.
Keep the profile values conservative until the engine has been measured on
that card. If several optimized implementations are tried, keep only the
currently selected one in the active build/catalog and archive the others.

Retain the old image locally and in the repository's build recipe. Make the app
choose the image through its existing setting or Compose override, so a failed
candidate never removes access to the stock run. Keep results and JIT caches on
host-mounted directories, not in the container writable layer.

### 3. Establish an honest stock baseline

Run at least one small, known-good input with the existing image and save the
full command, logs, manifest, output files, runtime phases, GPU telemetry, and
exit status. Confirm that the result is a real model output rather than an
empty/partial output directory. Confirm the expected checkpoint and MSA were
actually loaded.

Separate and record these phases when applicable:

1. queue / container startup;
2. input conversion and MSA/template preparation;
3. model initialization and compilation;
4. model inference;
5. ranking, relaxation, and output writing.

The app's total job time and the model-only time answer different questions.
Measure both. A faster model kernel does not accelerate remote MSA search or
relaxation.

### 4. Add one optimization at a time

Start with low-risk measurements and changes before porting fused kernels:

- ensure the app is not loading unused models or running duplicate workers;
- measure the current attention and triangle/pair operations, recycling,
  diffusion/sampling, and preprocessing;
- check upstream-supported precision, attention, compilation, and chunking
  controls before writing custom code;
- identify the peak-memory tensors and whether memory is allocated, reserved,
  fragmented, or held by a cache;
- change one lever, record it, and rerun the same input.

Typical fast-mode candidates are fused attention/triangle operations, graph
capture, compilation, and reducing redundant launches. Typical big-mode
candidates are chunked attention/transitions, checkpointing or recomputation,
smaller live intermediates, sequential sampling, and explicit sharding. These
are examples, not guaranteed fixes. Some implementations trade speed for memory;
test their effect on this specific model and card.

Keep upstream operations as the fallback for unsupported GPU architectures,
shapes, dtypes, or library versions. Test shape boundaries, including the
sequence lengths and channel dimensions actually produced by this model. A
kernel that works on a 256-channel block may not support 384 channels or a
different attention layout.

### 5. Validate correctness and usefulness

For every mode, hold constant the input sequence/complex, model weights, MSA,
templates, recycle count, sample count, random seed, and all quality-affecting
options. Save an input manifest and hash large inputs where practical. Do not
compare a cached-MSA run with an unaligned single-sequence run and attribute the
difference to a kernel.

Use a small smoke target first, then representative short, medium, and long
targets. Compare structures with residue/atom correspondence and appropriate
superposition; record backbone RMSD or another engine-appropriate comparison.
Also record native confidence outputs when provided, such as pLDDT, pTM, ipTM,
PAE, ranking score, and interface measures. Not every engine emits all of these,
and scores from different engines are not necessarily calibrated against one
another. A successful process exit or written CIF proves capacity, not structural
accuracy.

For `fast`, use multiple seeds or repeated predictions to estimate the stock
engine's own variation, then judge the optimized outputs against that baseline.
For `exact`, use the kit's stated deterministic protocol and compare all output
values it promises to preserve. For `big`, compare quality with stock on inputs
that both modes can run, and separately test whether the larger target now
completes. Do not infer quality from a mode name.

### 6. Measure speed and memory reproducibly

- Run each mode at least three warm times after separately recording a cold
  compile/autotune run. Use the same GPU assignment and avoid unrelated GPU
  workloads.
- Report median model time and total wall time; do not hide MSA, setup, or
  postprocessing time. Record failures and timeouts, not only successful runs.
- Sample NVML memory during inference on **each** GPU. Record per-card peak
  memory, not just a single `nvidia-smi` snapshot. Include allocation/reservation
  metrics from the framework when available.
- Keep model settings fixed. In Boltz2, for example, record `--max_msa_seqs`,
  raw A3M rows available, rows actually fed, recycles, diffusion steps/samples,
  and whether one or two GPUs participated. Reducing MSA depth can make a job
  fit, but it changes the input evidence and is not by itself a kernel
  optimization or a quality-preserving result.
- Publish exact image digest/tag, upstream commit, optimizer kit revision,
  hardware profile and driver, weights identifier, input identifier, command,
  mode, active levers, and output location with the benchmark. Record the
  profile's validation status separately from the image tag.

## Lessons from the local Boltz2 and AF3 work

### Boltz2

The separate `mn-boltz2-opt:2.2.1-cu130` image keeps the original
`mn-boltz2:cu128` available. Its long-sequence report is
[`shared/boltz2/optimized/benchmarks/msa-depth-and-length-2026-10-01.md`](shared/boltz2/optimized/benchmarks/msa-depth-and-length-2026-10-01.md).
On the two-4090 host, short targets (502–1,391 aa) completed in all three tested
modes on one GPU; `fast` was faster in those measurements. `big` was a memory
mode, not a promise of higher speed. Large targets from EP300 through PRKDC
completed in explicitly sharded two-GPU `big` runs, while the tested full-depth
one-GPU runs OOMed. LCT fit on one GPU in `big` only after the MSA cap was
reduced to 1,024; the reduction did not make EP300 fit on one GPU.

This establishes specific capacity results for that image, workload, and
software stack. It does not establish a universal sequence-length limit or
transfer to another engine. Keep both the MSA depth and number of GPUs visible
when describing a result.

### AF3 JAX memory work

The local AF3 memory recipes retain the stock AF3 3.1.14 image and add separately
tagged variants. They apply tested memory controls such as pair-transition
sharding, row chunking, and sequential diffusion samples; guarded fast kernels
are separate levers. The first memory image completed EP300 on one RTX 4090 but
did not make all longer proteins fit. The later 64-row variant could be slower
and did not universally extend capacity. This is a useful example of why a
smaller chunk is not automatically faster and why each memory change needs an
ablation and an end-to-end benchmark.

The AF3 JAX patches should not be copied wholesale to Protenix, OpenFold3,
ColabFold, or Foundry. Even related AF3-derived models can use different tensor
layouts, operations, MSA handling, and confidence heads.

## Candidate engines in this repository

These are starting priorities, not claims that optimization is already
available or tested for each local image:

| Current image | Recommended first work | Notes |
|---|---|---|
| `mn-protenix:cu128` | Compare the local pinned Protenix commit (`c3bfc365…`) to both `protenix_v1` and `protenix_v2` kits. Start with the matching version only. | The reference kits provide engine-specific Torch kernels, modes, and explicit `big`/multi-GPU paths, but their documented headline benchmarks are on other cards. Verify RTX 4090 support, checkpoint compatibility, MSA, and the exact local Protenix revision. Keep each Protenix major version distinct. |
| `mn-openfold3:cu13` | Compare local OpenFold3 commit (`9d96d70…`) with `openfold3` and `openfold3_ob0`; verify `sm_89` baseline first. | The local Dockerfile's architecture list names `sm_90`, `sm_100`, and `sm_120` PTX, not `sm_89`. Establish a reliable stock 4090 run before attributing any gain to an optimizer. |
| `mn-colabfold:1.6.1-cu12` | Compare its exact ColabFold and JAX pins to the `colabfold` kit. | Treat alignment/MSA/template preparation, JAX compilation, and model inference as separate phases. Persist the alignment and JAX caches. The local Dockerfile upgrades `jax[cuda12]` under `<0.8`; compare the resolved version to the kit lock rather than assuming they match. |
| `mn-foundry:cu128` | Keep the stock shared image. For RF3 folding, build a separate `mn-rf3-opt:0.2.1-dev13-cu130` candidate from the RoseTTAFold3 kit and use its `off`, `exact`, `fast`, or `big` mode. | The kit pins Foundry `0.2.1.dev13`, Torch `2.13.0+cu130`, and Triton `3.7.1`, which differ from the installed shared image. It supports compute capability 8.0 and up; the RTX 4090 uses the kit's 8.0 kernel rows. The kit's `big --n_gpu P` shards one prediction across GPUs. Verify the mode and quality on the local workload; do not apply this contract to RFDiffusion3 or other Foundry entrypoints. |

For image pins and current app wiring, see `images.yaml` and the matching
Dockerfiles/Compose services. Keep one active optimized recipe per engine and
put superseded attempts in its `archive/` directory. In particular, do not
change the Protenix or OpenFold3 upstream revision just to make the optimizer
port easier; if a dependency upgrade is necessary, isolate and benchmark that
as its own change.

## RTX 4090 test panel and acceptance table

For capacity testing, reuse the same eight-target panel already used in the
Boltz2 report: CHRNA7 (502 aa), SLC26A8 (970 aa), NUP155 (1,391 aa), LCT
(1,927 aa), EP300 (2,414 aa), CEP350 (3,117 aa), DMD (3,685 aa), and PRKDC
(4,128 aa). Use canonical sequences and the existing MSA cache where the
engine supports it. Check that each cached alignment query matches the target;
record the raw depth and the depth actually used.

For each engine/image and each advertised mode (`off`, `fast`, `big`), test
targets in ascending length order. Stop that engine/mode's sweep at the first
failed prediction. Mark the remaining larger targets in that mode as
`not run — stopped after earlier failure (presumed beyond tested limit)`; do
not spend more GPU time on them in that sweep or present them as individually
observed failures. This is a fail-fast capacity estimate, not proof that every
larger target would fail. If the mode has no failure, continue through the
whole panel. If an engine does not implement a mode, label every cell for that
mode `not supported`. Distinguish these statuses from OOM, timeout, invalid
input, kernel refusal, and completed prediction. Do not leave failures out of
the summary. Use the same sequence, MSA, weights, recycle/sample settings, and
GPU count for the three modes wherever possible; if a mode requires different
hardware or input settings, report that difference beside the result.

Not every model can take every sequence, complex, MSA format, or length. First
define each engine's valid input representation. A run is `pass` only if the
process exits successfully and expected structure and metadata outputs parse.
Log OOM, unsupported input, kernel refusal, timeout, or output validation as
separate outcomes; do not collapse them into a generic failure. After the first
failure in a mode, mark all larger targets in that mode as skipped under the
fail-fast rule above. A targeted follow-up can test those skipped targets if a
later code, memory, MSA, or hardware change gives a reason to revisit them.

Use a table like this for every image/mode. Fill it with measured values only:

| Image + upstream pin | Target / input | MSA rows used | Mode | GPU(s) | Status | Model s | Wall s | Peak MiB per GPU | Quality / output |
|---|---|---:|---|---|---|---:|---:|---|---|
|  |  |  | `off` |  |  |  |  |  |  |
|  |  |  | `fast` |  |  |  |  |  |  |
|  |  |  | `big` |  |  |  |  |  |  |

Do not compare unlike workloads across engines as if they were a controlled
speed race. The useful comparison is each optimized engine against its own
stock baseline on identical inputs and settings; a secondary cross-engine view
can summarize capacity and native quality while keeping those differences
explicit.

### Make benchmark outcomes visible in `mn-cofolding`

The benchmark is not complete until its results can be reviewed in the
`mn-cofolding` Streamlit app. Keep the raw outputs and run manifests under the
configured results/work directory, and expose both:

- a summary matrix for each engine/image with one row per target and separate
  `off`, `fast`, and `big` cells showing pass/fail and elapsed time; and
- a link from each cell to the corresponding job result page, where the input,
  mode, image/version, logs, timing phases, GPU/memory data, confidence metrics,
  and predicted structures can be inspected and downloaded.

Record at least **prediction status**, **model/inference seconds**, and **full
job wall seconds** for every attempt. Include queue/setup and local MSA
preparation timings when available so a reader can tell whether a difference
came from inference or the surrounding pipeline. Show OOMs and other failures
with their reason and time-to-failure. A result that writes no valid structure
must not appear as a pass. Store one structured record per attempt (including
engine, image tag/digest, upstream pin, target, mode, seed, settings, MSA rows
used, GPU count, status, timestamps/durations, peak memory, output paths, and
job ID when submitted through the app) alongside a human-readable report. The
app's summary should be generated from or checked against those records, so
the visible pass/time matrix cannot drift from the benchmark files.

## Tests and release gate

Before an optimized tag becomes a normal app option, require:

1. A stock smoke prediction matching the current image and a candidate `off`
   run showing that off really invokes stock.
2. One end-to-end smoke prediction for every advertised mode on RTX 4090, with
   expected structures and readable logs.
3. Tests for mode parsing, lever selection, unsupported hardware/shape refusal,
   cache handling, and strict versus allowed partial fallback.
4. Numerical comparison against stock on representative inputs, including a
   sequence for which both modes complete.
5. A medium/long capacity sweep with OOM and partial-output detection, followed
   by repeat warm timing and per-GPU memory sampling.
6. An explicit two-GPU run only if the engine's single-inference sharding path
   is supported; verify which GPU owns which part of the model state and report
   the communication layout.
7. A run manifest and a benchmark report saved beside that optimizer recipe.
8. The same smoke and correctness checks on the RTX 5090 profile before
   marking `rtx5090.env` validated; until then, leave the profile explicitly
   deployment-only.

Keep the original image/tag, weights, and result folders intact. Publish the
candidate as experimental until it has passed this gate on the intended
hardware profile. A successful RTX 4090 run does not validate RTX 5090 kernels,
and a successful RTX 5090 run does not validate RTX 4090 behavior.

## Useful local references

- [Uplifting Biomolecular Modeling kits](../tools-to-implement/uplifting-biomolecular-modeling/README.md)
- [Boltz2 stock recipe](shared/boltz2/Dockerfile)
- [Boltz2 active optimized recipe and benchmark](shared/boltz2/optimized/benchmarks/msa-depth-and-length-2026-10-01.md)
- [AF3 stock recipe](protein-design/cofolding-af3/Dockerfile)
- [AF3 active optimized recipe](protein-design/cofolding-af3/optimized/Dockerfile)
- [AF3 archived candidates](protein-design/cofolding-af3/archive/)
- [Long-protein capability report](../mn-cofolding/benchmarks/long-protein-capability-2026-09-30/README.md)
- [Installed image catalog](images.yaml)
