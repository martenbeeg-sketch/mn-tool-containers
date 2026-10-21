# BioEmu-2

Build with `./build.sh bioemu2`. The image pins Microsoft BioEmu-2 source to
`f99e0d899a2548ee6d9d54b6854b2bc35b6652d5`, uses Python 3.12 and the upstream
CUDA 13 extra, and supports the project’s RTX 4090 / RTX 5090 deployment target.
The upstream compute-capability floor is 7.5; those cards satisfy it, but neither
card is listed as end-to-end validated upstream. A 24 GB RTX 4090 can be memory
limited for large systems; start with a small sample batch. The 32 GB RTX 5090
has more headroom, but is also unverified with this preview.

The image does not bake model weights into its layers. BioEmu-2 downloads its
`bioemu2-preview` checkpoint on first use. Mount a persistent writable cache at
`/cache/huggingface` (`HF_HOME`) so the checkpoint is reused between jobs.

Examples:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" -v "$HOME/.cache/huggingface:/cache/huggingface" \
  mn-bioemu2:0.1.0rc0-f99e0d8-cu130 \
  sample-equilibrium --pdb-id 1AKE --num-samples 8 --out /work/results/adk
```

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" -v "$HOME/.cache/huggingface:/cache/huggingface" \
  mn-bioemu2:0.1.0rc0-f99e0d8-cu130 \
  propagate --x0-cif /work/target-design.cif --num-samples 4 \
  --lag-time-ns 1 --out /work/results/target-design
```

`propagate` samples model-based, time-lagged conformational changes from a
starting complex; it is not an atomistic molecular-dynamics simulation.
Outputs include `samples.xtc`, `topology.cif`, and per-frame
`physicality.json`. The upstream quickstart and speed/capacity guidance are in
[BioEmu-2](https://github.com/microsoft/bioemu2) and its
[capacity guide](https://github.com/microsoft/bioemu2/blob/main/docs/speed-and-capacity.md).
