# AF3 hardware profiles

The stock and optimized AF3 images are shared across GPU generations. These
files are host-side launch profiles, not additional Dockerfiles:

```bash
set -a
source protein-design/cofolding-af3/profiles/rtx5090.env
set +a
```

The RTX 4090 profile records the tested Ada baseline. The RTX 5090 profile
targets Blackwell `sm_120` but remains unvalidated until a real 5090 run is
completed. The profile selects one GPU by default; set `MN_COFOLDING_AF3_N_GPU=2`
only for a supported two-GPU run on the same host. Two networked machines
cannot be combined into one AF3 inference.

Choose `MN_COFOLDING_AF3_OPT_MODE` explicitly for each benchmark. Keeping the
profile default at `off` makes the stock-versus-optimized comparison visible.
