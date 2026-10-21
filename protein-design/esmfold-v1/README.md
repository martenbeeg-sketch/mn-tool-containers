# ESMFold V1 container

Build a fresh image from the pinned recipe in this repository:

```bash
./build.sh esmfold-v1
```

The build starts from NVIDIA CUDA 12.8 devel, installs the pinned fair-esm 2.0.1 source bundle and PyTorch 2.7.0, then builds OpenFold's attention extension for RTX 4090 (SM 8.9). It does not require the old `ovo-esm:latest` image or a separate OVO checkout. The fair-esm source archive is included here and verified by SHA-256; OpenFold and dllogger are checked out at fixed commits.

The image intentionally excludes model weights. On first use, fair-esm downloads `esmfold_3B_v1.pt` into the shared reference tree:

```text
<reference_files>/esm_models/esmfold_v1/torch/hub/checkpoints/esmfold_3B_v1.pt
```

The app mounts that cache at `/reference/esmfold_v1_torch` and sets `TORCH_HOME`, so the downloaded checkpoint is reused by the cofolding and refolding workflows.

This is classic ESMFold V1 structure prediction. ESM-IF1 is a separate inverse-folding sequence-design tool; its checkpoint and workflow are not used by this image.
