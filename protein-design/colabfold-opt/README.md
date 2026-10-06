# Optimized ColabFold image

This candidate uses the local ColabFold optimization kit's pinned JAX 0.5.3
stack and defaults `COLABFOLD_OPT=fast`. It keeps the existing
`colabfold_batch <input> <output> [options]` command and cache mount. The stock
`mn-colabfold:1.6.1-cu12` image remains available and remains the app default.

Build it from the container repository with:

```bash
./build.sh colabfold-opt
```

To use it in the MN protein design benchmark worker, set
`MN_COLABFOLD_IMAGE=mn-colabfold-opt:1.6.1-jax0.5.3-cu124` in the app
environment and restart the app. The image supports kit modes through
`COLABFOLD_OPT=off|exact|fast|big`. The app worker assigns one GPU; for a
multi-GPU `big` run, invoke the container directly with the required GPUs
visible and either set `COLABFOLD_OPT_N_GPU=2` (or another supported count)
with `colabfold_batch`, or use the kit launcher with
`bash run.sh pred --mode big --n_gpu 2 ...`. The app mounts its existing
parameter directory at `/cache/params`; this image stores JAX compilations in
`/cache/params/.jax`, so the app's existing mount preserves them across runs.

The kit's JAX pin is deliberate: its optimization hooks and kernels are gated
against the tested ColabFold/AlphaFold source and JAX stack. This recipe does
not replace or patch JAX itself.
