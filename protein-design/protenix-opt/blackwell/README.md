# Protenix V2 stock LayerNorm: RTX 4090 and RTX 5090

The Phase 7B image `mn-protenix-opt:2.0.0-cu128-batch2` advertises
PyTorch SM120 support, but its separate stock `fast_layer_norm_cuda_v2.so`
contains only SM70/80/86/89/90/100 cubins and no PTX. Its compiler explicitly
omits SM120. The stock path fails on RTX 5090 with `no kernel image is available`;
the same image's independent PyTorch LayerNorm diagnostic succeeds there.

`../Dockerfile.sm120` adds SM120 to the existing compiler's requested targets
and rebuilds that extension offline. It retains the base image, Protenix source,
model, PyTorch/CUDA stack, mn-predict batch adapter and inference settings.
Weights remain mounted externally. The old image/tag stays available.

One image contains both Ada SM89 and consumer Blackwell SM120 kernels:
`mn-protenix-opt:2.0.0-cu128-sm89-sm120-batch3`. A separate image is unnecessary
for this fix. Architecture-specific optimized images can still be introduced
if later benchmarks justify different dependencies or performance settings.
SM100 support alone does not establish SM120 compatibility; see NVIDIA's
[Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

From the repository root:

```bash
./build.sh protenix-v2-opt-sm120
```

The base image must already be installed. Optional variables
`PROTENIX_SM120_BASE_IMAGE` and `PROTENIX_SM120_IMAGE` select another explicitly
reviewed base and new output tag. The helper records the resolved base image ID
in image labels. The build fails if the expected stock compiler anchor or CUDA
SM120 compiler support is absent, or either SM89/SM120 cubin is missing.
`/opt/mn-protenix-fixes/build-manifest.json` records compiler and binary hashes.

`check_layernorm.py` performs 48 small checks across float32/float16/bfloat16,
four channel widths and four scale/bias combinations. Run it only on an idle,
scheduler-leased GPU. These checks are kernel diagnostics, not protein inference
or proof of molecular quality. The acceptance record must also include normal
Studio graph execution of a retained scientific input on both GPU generations,
exact target/sequence identity, native process/lease receipts and Results export.

This change validates the `off` stock path. It does not certify the optimization
kit's `exact`, `fast` or `big` modes on Blackwell, claim a speed improvement, or
change scientific thresholds or generation budgets. Existing optimization
benchmarks remain separate in the parent directory.

Raw validation evidence is kept under
`/mnt/data/RESULTS/mn-molecular-design-studio/verification/protenix-blackwell-image-20261029`.
The Phase 7B acceptance report in the Studio repository records final deployment
and scientific acceptance status.
