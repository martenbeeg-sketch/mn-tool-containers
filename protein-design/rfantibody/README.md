# RFantibody image

Build with `./build.sh rfantibody`. The builder uses the local pinned checkout at
`../tools-to-implement/RFantibody` when available, or fetches upstream commit
`3e5a5bb1cb98befde9fc049ae3ff58cabb491f67` into a temporary build context.

The resulting image is `mn-rfantibody:3e5a5bb-cu118-v1`. It installs the upstream
CUDA 11.8 environment and downloads the RFdiffusion antibody, ProteinMPNN, and
RF2 checkpoints listed in the Dockerfile. Studio's optional AbMPNN checkpoint
remains in the external reference-files mount.

Source code is under the upstream MIT license. The model weights have separate
terms; review upstream attribution and weight terms before redistributing the
built image or checkpoints.
