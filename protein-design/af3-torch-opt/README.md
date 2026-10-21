# AF3-Torch optimization image

This image is built from the AF3-Torch kit in `uplifting-biomolecular-modeling/af3_torch`. It runs xfold's PyTorch implementation of AlphaFold 3 with the kit's four modes: `off`, `exact`, `fast`, and `big`.

It also includes `af3-torch-prepare-templates`, a CPU-only preprocessing command for apps that receive separate PDB/mmCIF files. It aligns each file to the named protein chain, converts PDB to AF3-compatible mmCIF, and writes an AF3 JSON v2 input using relative `mmcifPath` references. The predictor itself already accepts these standard AF3 templates; the helper makes the upload preparation reusable across apps.

## Model parameters

The kit expects OpenFold3 preview-2 (`of3-p2-155k.pt`) converted by the pinned AlphaFold 3 fork into `of3_ported_weights.bin.zst`. The pinned converted artifact is SHA-256 `0c9a00607352395486e33a98367810e1ad56c87459289f40974579525d3959cf` (1,369,240,911 bytes). The app reads it from `/mnt/db/reference_files/mn-cofolding/af3_torch` on the host and mounts that directory read-only. This is separate from official AF3 parameters and from native OpenFold3/OpenBind-0.

The host may reuse an existing P2 source checkpoint only if the kit converter produces the exact converted digest above. Do not replace the existing source file in place. The kit's `run.sh install --weights DIR` checks the converted artifact without fetching weights; `--fetch` is not used by the MN Cofolding build or runner.

## GPU profiles

`rtx4090.env` is the local deployment profile. It selects the kit's generic `h100` config file while naming the actual RTX 4090 and keeping the JIT cache under the SM 8.9 key. The kit detects compute capability from `nvidia-smi` at runtime. `rtx5090.env` is ready for deployment on a CUDA 13 capable RTX 5090 host, with an SM 12.0 cache key; it has not been validated on RTX 5090 hardware.

The app reserves GPU 0 for `off`, `exact`, and `fast`. `big` reserves GPUs 0 and 1 and passes `--n_gpu 2`, which enables the kit's pair-stack row sharding. It uses the same checkpoint and local per-chain unpaired A3Ms across modes.

## Build

From `mn-tool-containers`, run `./build.sh af3-torch-opt`. The build copies the kit and shared core into a temporary context, builds the kit image as `mn-af3-torch-opt-kit:22bdeed-cu130`, and adds the reusable template-preparation command as a small wrapper layer tagged `mn-af3-torch-opt:22bdeed-cu130`. Weights are not baked into either image.

## Prepare uploaded templates

Put one single-chain structure per protein entity in a directory, naming each file after its AF3 chain ID (`A.pdb`, `B.cif`, and so on). Then run:

```bash
docker run --rm -v "$PWD":/work -w /work mn-af3-torch-opt:22bdeed-cu130 \
  af3-torch-prepare-templates \
  --input-json input.json --template-dir templates --output-dir prepared
```

The helper writes `prepared/input.json` and `prepared/templates/*.cif`. Run the usual `run.sh pred --json_path prepared/input.json` against that output. It needs the image's Gemmi installation but no GPU or model weights. PDB inputs need a valid `HEADER` deposition date; otherwise use an mmCIF containing revision history.

For example, once the PDB/MMCIF conversion above has completed:

```bash
docker run --rm --gpus '"device=0"' \
  -v "$PWD":/work -v /path/to/af3_torch_weights:/reference/af3_torch:ro \
  -e AF3_TORCH_PARAMS_DIR=/reference/af3_torch -w /work \
  mn-af3-torch-opt:22bdeed-cu130 \
  bash /kit/af3_torch/run.sh pred --config h100 --mode fast \
  --json_path /work/prepared/input.json --output_dir /work/out \
  --model_dir /reference/af3_torch --norun_data_pipeline
```
