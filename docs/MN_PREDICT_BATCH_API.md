# MN Predict Batch API

The image-side command `mn-predict batch` gives Scheduler adapters one stable
invocation while keeping each model's native input format inside its image.
The Scheduler remains responsible for queueing, resource leases, cancellation,
retry state, provenance, and scoring.

## Manifest (`mn-predict/v1`)

```json
{
  "api_version": "mn-predict/v1",
  "batch_id": "target-7wr3-boltz2-fast",
  "engine": "boltz2",
  "mode": "fast",
  "gpu_count": 1,
  "samples_per_item": 3,
  "seed": 1,
  "items": [
    {"id": "design-001", "input": "/work/yaml_inputs/design-001.yaml"},
    {"id": "design-002", "input": "/work/yaml_inputs/design-002.yaml"}
  ],
  "options": {
    "cache": "/models",
    "recycling_steps": 10,
    "sampling_steps": 200,
    "write_full_pae": true,
    "output_format": "mmcif"
  }
}
```

Run with the same argument shape in each batch-enabled image:

```text
docker run ... IMAGE mn-predict batch --manifest /work/batch.json --output-dir /work/output
```

`mn-predict capabilities --json` reports the image's supported modes and its
actual batch execution path. A successful run writes
`OUTPUT_DIR/mn-batch-result.json`, with a separate status and structure list
for every item. The container exits nonzero if any item does not produce the
requested number of structures or the native prediction command fails.

## Overath benchmark image overlays

Every runnable image tag in the frozen Overath matrix has a versioned overlay
that exposes `mn-predict/v1`. These image tags passed the capability check for
API version, engine dispatch, and declared mode support. The Scheduler
preflight report is the authority for runtime support: it exercises each
image/mode and scores every requested sample through DockQ.

The shared command gives the Scheduler one invocation per target batch. The
engine adapters preserve each model's native input format; some pass a full
item list to one model process, while others iterate over the items inside the
same container. Therefore “batch” means one Scheduler job and container for a
target's designs; it does not promise that every engine predicts all designs
simultaneously.

| Image tag | Engine adapter | Batch execution path |
|---|---|---|
| `mn-af3-torch-opt:22bdeed-cu130-batch1` | `af3_torch` | One container invocation over a staged JSON input directory. |
| `mn-alphafast:cuda126-backup-batch1` | `alphafast` | One AlphaFold3 input-directory process. |
| `mn-atlasfold-opt:1.0.0-cu128-batch2` | `atlasfold` | One multi-record FASTA invocation. |
| `mn-biohub-esm:3.4.1-cu128-batch1` | `esmfold2_biohub` | One container with the model loaded once for all requests. |
| `mn-boltz2-opt:2.2.1-cu130-batch1` | `boltz2` | One YAML-directory invocation through the persistent Boltz2 worker. |
| `mn-boltz2:cu128-batch1` | `boltz2_legacy` | One stock Boltz2 YAML-directory worker. |
| `mn-boltzgen-opt:a07fea1-cu130-batch1` | `boltzgen_fold_opt_msa` | Legacy overlay; its output glob also matched BoltzGen's unnumbered best-structure CIF. |
| `mn-boltzgen-opt:a07fea1-cu130-batch2` | `boltzgen_fold_opt_msa` | Superseded: the wrapper invoked BoltzGen per item while each invocation scanned the full shared `data.design_dir`, repeating the entire target batch. |
| `mn-boltzgen-opt:a07fea1-cu130-batch3` | `boltzgen_fold_opt_msa` | One native BoltzGen invocation scans the target batch once; the adapter verifies a one-to-one match between manifest items and staged PDBs and returns only numbered diffusion samples. |
| `mn-chai1-opt:0.6.1-cu130-batch1` | `chai1` | One items-manifest invocation. |
| `mn-cofolding-af3-memory:3.1.14-cu13-af3-batch1` | `alphafold3` | One container; serial per-item AF3-JAX runner calls. |
| `mn-cofolding-af3-memory:3.1.14-cu13-openfold3-batch1` | `openfold3` | One container; serial per-item OpenFold3 runner calls. |
| `mn-colabfold:1.6.1-cu12-batch1` | `colabfold` | One input-directory invocation; off mode only. |
| `mn-esmfold2-opt:0.1.0-cu130-batch1` | `esmfold2` | One native multi-item JSON process. |
| `mn-esmfold2-opt:0.1.0-cu130-fast-batch1` | `esmfold2_fast` | One native multi-item JSON process using the fast profile. |
| `mn-foundry:cu128-batch2` | `rf3_foundry_legacy` | One container; native RF3 calls run per item, with only the three seed/sample output directories reported. |
| `mn-opendde-opt:1.1.1-cu126-batch3` | `opendde_native` | One native JSON item-list invocation; duplicate row-pair files are excluded from samples. |
| `mn-openfold3-native-opt:0.5.0-ob0-cu128-batch2` | `openfold3_ob0` | One container; native runner calls run per item. |
| `mn-openfold3:cu13-batch2` | `openfold3_p2` | One container; serial native OpenFold3 calls. |
| `mn-protenix-opt:2.0.0-cu128-batch2` | `protenix_v2_opt` | One native input-directory process. |
| `mn-protenix-v1-opt:1.1.0-cu130-batch2` | `protenix_v1` | One native input-directory invocation. |
| `mn-protenix:cu128-batch2` | `protenix_v2_stock` | One native input-directory process. |
| `mn-pxdesign:cu128-batch6` | `pxdesign_v05` | One container; default runner calls run per item, with PXDesign's pinned CCD cache, checkpoint, and weights resolved from the read-only `/reference/pxdesign` subtree. |
| `mn-rf3-opt:0.2.1-dev13-cu130-batch1` | `rf3_foundry` | One native JSON item-list process with item and seed/sample IDs. |

The benchmark uses 22 distinct runnable image tags across 63 runnable
image/mode combinations. The four AF2-IG N/A rows describe the selected campaign
integration, not an inherent AF2-IG input limitation. The optimized
`mn-af2ig-opt` image supports `off`, `exact`, `fast`, and `big`; its
current native PDB-directory runner expects binder-first complex PDBs. The
separate MN Protein Design ColabDesign initial-guess workflow also supports a
target-only template PDB with the binder sequence in a sidecar, plus full-complex
variants with optional binder and interface templates. The target-only template
variant fits the no-binder-coordinate rule; the richer structural variants do
not. This campaign did not select or preflight either AF2-IG adapter for its
frozen target-only inputs, and target-MSA handling would need validation. The
N/A status therefore means unselected/unverified for this campaign, not that
AF2-IG lacks a target-only initial-guess path. AF2-IG is separate from the
`colabfold_batch` adapter. Other unsupported modes remain N/A in the condition
manifest with their reasons.

Each overlay is built from an existing base tag without replacing that tag.
Image labels record the base tag and the base image's local SHA256 ID. July
candidate aliases are identified by those recorded image IDs; an `off` mode
overlay is not treated as the historical original image.

These are versioned overlay images built on the existing model images. The
pre-existing tags remain available and keep their original image identities.
The pilot wrappers do not alter target MSA or target-only template contents;
those remain prepared and provenance-checked by the host workflow.

Build all current and historical-candidate overlays after the base images are
present:

```text
bash build.sh batch-api
```
