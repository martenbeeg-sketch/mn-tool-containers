# RTX 4090 and RTX 5090 image compatibility

Keep one image per engine when its runtime stack can serve both Ada (RTX 4090,
sm_89) and Blackwell (RTX 5090, sm_120). Split images only when an engine or
compiled extension cannot use a common stack. A CUDA version in an image tag is
the user-space runtime/toolkit version; the installed NVIDIA driver still has
to support it.

## Engine image plan

| Engine | Shared image | GPU-specific build detail | Compatibility status |
| --- | --- | --- | --- |
| RFantibody | `mn-rfantibody:3e5a5bb-cu128` | Torch 2.7.1 cu128; build DGL 2.4.0 with sm_89 and sm_120. Checkpoints are mounted from `/data/reference_files/rfantibody`. | Passed Torch CUDA matmul, DGL CUDA graph, and read-only checkpoint-mount checks on RTX 4090 and RTX 5090. One shared image is sufficient. |
| OpenDDE optimized | `mn-opendde-opt:1.1.1-cu128` | Torch 2.7.1 cu128; include `rtx4090` and `rtx5090` run profiles in the same image. Weights are mounted from `/data/reference_files/opendde`. | Built and passed 5090 Torch matmul and profile checks; model inference not run. Profiles are staged from the pinned kit's A100 defaults and are not claimed to be separately tuned. |
| Genie3 | `mn-genie3:cu128` | Explicit Torch 2.7.1 cu128 wheel plus JAX 0.5.3. Weights stay under `/data/reference_files/genie3`. | Built; Torch and JAX CUDA matmul checks passed on 5090. Model inference not run. |
| PXDesign | `mn-pxdesign:cu128` | Torch 2.7.1 cu128 plus JAX 0.5.3. Model data stay under `/data/reference_files/pxdesign`. | Built; Torch and JAX CUDA matmul checks passed on 5090. Model inference not run. |

## Why the existing tags were insufficient

The previous 5090 smoke checks found unsupported CUDA kernels in the installed
Torch stacks: RFantibody used Torch 2.3.1 cu118, OpenDDE used Torch 2.7.1
cu126, and Genie3's Torch path used Torch 2.7.1 cu126. PXDesign's Torch path
worked, but its JAX 0.4.29 compile failed on the 5090. These observations apply
to the old images; they do not certify the rebuilt tags above.

The container build targets are:

```bash
bash build.sh rfantibody
bash build.sh opendde-opt
bash build.sh genie3
bash build.sh pxdesign
```

The OpenDDE, Genie3, and PXDesign checks above ran on the 5090 at pc004032. They
confirm GPU tensor kernels, not end-to-end inference. Their 4090 checks remain
outstanding. RFantibody was checked on RTX 4090 at pc002975 and RTX 5090 at
pc004032 and pc004033; these are runtime compatibility checks, not a full
design inference.
