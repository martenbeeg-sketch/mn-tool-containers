# Optimized AF2-IG image

This candidate builds the AF2-IG optimization kit as a separate image with its
pinned dl_binder_design commit and JAX 0.5.3 stack. Build it with:

```bash
./build.sh af2ig-opt
```

The kit's supported interface is its `run.sh pred` / `predict_pdb.py` path. A
container invocation follows the kit command line, for example:

```bash
docker run --rm --gpus '"device=0"' \
  -e AF2_PARAMS=/weights/af2ig \
  -v /path/to/af2ig-weights:/weights/af2ig:ro \
  -v /path/to/binder-target-pdbs:/inputs:ro \
  -v /path/to/output:/output \
  mn-af2ig-opt:cafa3853-jax0.5.3-cu124 \
  bash run.sh pred --mode fast --pdbdir /inputs --out /output
```

The app's current AF2 initial-guess evaluator uses ColabDesign and its own
input/output and metric contract. It is not interchangeable with this kit's
`predict_pdb.py` runner, so the app continues using its existing image and
workflow for AF2-IG. This candidate does not replace that path.
