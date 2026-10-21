# BoltzGen optimized candidate

This candidate keeps the patched BoltzGen source from
`mn-boltzgen:a07fea1-test`, then installs it on the optimization kit's pinned
CUDA 13 / PyTorch 2.13 stack. The image build creates a wheel from the retained
`/app` tree and updates the kit's local source pin to that wheel; the upstream
stock image and its Dockerfile remain available.

The runtime matches the kit record: Python 3.11.5, CUDA 13.0, PyTorch
2.13.0+cu130, Triton 3.7.1, and cuEquivariance 0.11.1 with CUDA 13 kernels.
The pinned checkpoint and auxiliary data remain outside the image in the
existing mounted Hugging Face cache.

## Modes and GPU support

The kit defines `off`, `exact`, `fast`, and `big`; all kit modes require
`--seed`. `exact` and `fast` require a diffusion batch of at least two designs
or at least 100 requested designs. `big` is a single-GPU memory mode. The kit
refuses `--devices > 1` for `exact`, `fast`, and `big`. `off` can use the
upstream multi-device runner for separate batch items; that does not shard one
prediction across both 4090s.

On structure-conditioned inputs, the kit documents a numeric difference in
`big`: its 64-row token-distance chunks use different TF32 accumulation shapes.
`exact` is the mode for output equality; `fast` allows documented numerical
changes. FoldBench records same-seed `exact` versus `off` coordinate equality
and scores each mode's emitted CIF against the native complex.

The kit's GPU-step runner also covers BoltzGen's `folding` and
`design_folding` pipeline steps, which load Boltz2 checkpoints. The chunked
`TokenDistanceModule` path can reduce their peak memory, and the design
diffusion levers apply where those steps use the shared diffusion modules.
Their refolding speed and quality are measured separately from design
generation in the MN Cofolding FoldBench panel.

The optimized image includes an opt-in benchmark writer switch:
`BOLTZGEN_WRITE_ALL_DIFFUSION_SAMPLES=1` writes one CIF per diffusion sample
alongside BoltzGen's usual highest-confidence CIF. The MN FoldBench runner uses
this switch to score five samples for each of three measured seeds (15 CIFs per
target and mode). Normal app runs do not set the variable and keep BoltzGen's
usual single-CIF output.

## Inputs

The local BoltzGen source accepts PDB/CIF structures through a YAML `file.path`
entity and can select chains. The optimized refolding data module accepts
optional per-chain A3Ms under
`external_msa_dir/<input-stem>/<chain>.a3m`. It aligns the A3M query to the
resolved coordinate sequence, projects every row onto those residues, and
rejects a query with less than 95% identity at mapped positions. This handles
unresolved loops and termini without shifting MSA columns. BoltzGen's native
A3M parser then builds the features. `external_msa_max_seqs` defaults to 1,024
unique rows. If the option is omitted, existing no-MSA behavior is unchanged.
The normal `boltzgen run` YAML design path and separate binder-data `.npz` path
are unchanged.

The MN FoldBench runner stages and passes the saved target-chain A MSA; binder
chain B remains sequence-only to match the target-template setup. It records
the full saved depths and the loaded-row cap separately. The general loader
accepts per-chain files, but current MN A3Ms do not carry shared taxonomy IDs,
so their rows are not paired across chains.

## RTX 4090 launch

Mount the existing BoltzGen Hugging Face cache at `/cache`, then use:

```bash
bash /kit/boltzgen/run.sh check --config rtx4090 --mode exact
bash /kit/boltzgen/run.sh design --config rtx4090 --mode exact SPEC.yaml \
  --output /workspace/run-exact --seed 0 --diffusion_batch_size 2
```

The `rtx5090` profile remains unvalidated until exercised on 5090 hardware.
