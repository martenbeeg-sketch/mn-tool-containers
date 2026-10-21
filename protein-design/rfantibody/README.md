# RFantibody image

Build with `./build.sh rfantibody`. The builder uses the local pinned checkout at
`../tools-to-implement/RFantibody` when available, or fetches upstream commit
`3e5a5bb1cb98befde9fc049ae3ff58cabb491f67` into a temporary build context.

The shared 4090/5090 image is `mn-rfantibody:3e5a5bb-cu128`. It uses Torch
2.7.1 cu128 and builds DGL 2.4.0 for sm_89 and sm_120. Torch CUDA and DGL graph
checks passed on RTX 4090 and RTX 5090.

Model weights stay outside the image in
`/data/reference_files/rfantibody/`: `RFdiffusion_Ab.pt`,
`ProteinMPNN_v48_noise_0.2.pt`, and `RF2_ab.pt`. Studio mounts this directory
read-only at `/ref/rfantibody`, which is also the value of
`RFANTIBODY_WEIGHTS`. The optional AbMPNN checkpoint stays in the shared
LigandMPNN reference folder.

Source code is under the upstream MIT license. The model weights have separate
terms; review upstream attribution and weight terms before redistributing the
built image or checkpoints.
