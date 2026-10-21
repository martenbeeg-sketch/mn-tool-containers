# Protenix V1 optimization image

This image packages the Protenix V1 kit's pinned Protenix 1.1.0 stack. The app mounts the existing checkpoint and data files at
`/reference/protenix`; no weights are copied into the image or downloaded during prediction. The kit cache is stored under the shared
MN Cofolding cache at `/cache/protenix_v1/jit`.

`./build.sh protenix-v1-opt` builds the kit base and this app wrapper. The RTX 4090 profile is used for local runs; an RTX 5090 profile is
included for deployment. Kit gates inspect the running GPU capability and record unavailable or shape-gated levers by name.

The MN Cofolding app routes `off`, `exact`, `fast`, and `big` through the kit's `run.sh`. Off/exact/fast use GPU 0. Big uses the kit's
rowpair sharding on GPUs 0 and 1. Every local A3M is capped at the configured maximum depth and supplied as unpaired MSA data; templates
are disabled. The matched FoldBench panel uses 16,384 rows per chain, three seeds, five samples per seed, and ten cycles.
