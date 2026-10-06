#!/usr/bin/env python3
"""MN Cofolding runner with guarded FlashPairformer kernels for AF3 3.1.14."""

from __future__ import annotations

import atexit
import math
import os
import runpy
import sys
from pathlib import Path


STOCK_RUNNER = Path("/usr/local/bin/cofolding-af3-runner-stock.py")
OPT_ROOT = Path("/opt/mn-cofolding-af3-opt")
FLASHPAIRFORMER = OPT_ROOT / "flashpairformer"
OPT_CORE = OPT_ROOT
MODES = {"off", "fast", "big"}
KERNEL_SETS = {"trimul", "triatt", "both"}
BIG_SAMPLE_CHUNK = 1
BIG_PAIR_ROWS = 64
BIG_TRIMUL_ROWS = 64
BIG_MEMORY_REPORT = {
    "sampler_calls": 0,
    "sampler_chunks": 0,
    "sampler_shapes": [],
    "conditioning_calls": 0,
    "conditioning_shapes": [],
    "trimul_chunk_calls": 0,
    "trimul_chunk_shapes": [],
    "diffusion_logits_calls": 0,
    "diffusion_logits_shapes": [],
}


def kernel_tile_buckets(*, tile: int = 64, maximum: int = 5120) -> list[int]:
    """Return the shared AF3 compilation buckets at the selected tile size."""
    if tile < 1 or maximum < tile:
        raise ValueError("tile and maximum must be positive, with maximum >= tile")
    return list(range(tile, maximum + 1, tile))


def sample_slices(num_samples: int, chunk_size: int) -> list[tuple[int, int]]:
    """Partition diffusion samples into fixed-size passes."""
    if num_samples < 1 or chunk_size < 1:
        raise ValueError("num_samples and chunk_size must be positive")
    return [
        (start, min(start + chunk_size, num_samples))
        for start in range(0, num_samples, chunk_size)
    ]


def triangle_row_slices(num_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    """Partition square pair tensors into output-row chunks."""
    if num_rows < 1 or chunk_size < 1:
        raise ValueError("num_rows and chunk_size must be positive")
    return [
        (start, min(start + chunk_size, num_rows))
        for start in range(0, num_rows, chunk_size)
    ]


def main() -> None:
    arguments = sys.argv[1:]
    mode = os.environ.get("MN_COFOLDING_AF3_OPT_MODE", "off").strip().lower()
    if mode not in MODES:
        raise SystemExit(f"MN_COFOLDING_AF3_OPT_MODE must be one of {sorted(MODES)}, got {mode!r}")
    if mode == "big":
        # A larger contiguous BFC pool avoids the fragmentation observed when
        # dynamic allocation was used for the 1,927-aa result materialization.
        # Keep preallocation on; nvidia-smi then reports the reservation, not
        # the model's live-tensor peak. Leave a small driver margin on cards
        # without a desktop attached.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
        print(
            "[mn-cofolding-af3-opt] big memory_policy=preallocate "
            f"fraction={os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}",
            flush=True,
        )

    helpers = runpy.run_path(str(STOCK_RUNNER), run_name="mn_cofolding_af3_runner_helpers")
    input_path = helpers["_json_path"](arguments)
    if input_path is not None:
        helpers["_prepare_ccd"](input_path)
    elif "--help" in arguments or "-h" in arguments:
        helpers["_prepare_ccd"](None)
    helpers["_install_msa_cache"]()

    # Keep the same padded shapes across off/fast/big. The shared toolkit's
    # kernel_tile policy pads to multiples of 64, avoiding the stock AF3
    # buckets' large jumps (for example 1400 -> 1536). This is part of the
    # measured mode configuration, not a change to model weights.
    if not any(arg == "--buckets" or arg.startswith("--buckets=") for arg in arguments):
        tile_buckets = ",".join(map(str, kernel_tile_buckets()))
        arguments.append(f"--buckets={tile_buckets}")
        print("[mn-cofolding-af3-opt] bucket_policy=kernel_tile tile=64", flush=True)

    if mode != "off" and "--jax_backend=cpu" not in arguments and "--jax_backend" not in arguments:
        kernel_set = os.environ.get("MN_COFOLDING_AF3_OPT_KERNELS", "both").strip().lower()
        if kernel_set not in KERNEL_SETS:
            raise SystemExit(
                f"MN_COFOLDING_AF3_OPT_KERNELS must be one of {sorted(KERNEL_SETS)}, got {kernel_set!r}"
            )
        label = mode if kernel_set == "both" else f"{mode}:{kernel_set}"
        # Big mode owns the triangle-multiplication site with a row-chunked
        # implementation. Keep fused triangle attention, but do not install
        # FlashPairformer's fused triangle multiplication over that site.
        flash_kernels = "triatt" if mode == "big" else kernel_set
        _install_flashpairformer(flash_kernels, display_mode=label)
        if mode == "big":
            _install_big_memory_mode()
    else:
        print(f"[mn-cofolding-af3-opt] mode=off requested={mode}", flush=True)

    sys.argv = [str(helpers["RUNNER"]), *arguments]
    runpy.run_path(str(helpers["RUNNER"]), run_name="__main__")


def _install_flashpairformer(mode: str, *, display_mode: str | None = None) -> None:
    # The package defaults to activating its kernels at import. Set its switch
    # to off first so this wrapper is the only place that selects the mode.
    sys.path.insert(0, str(OPT_CORE))
    sys.path.insert(0, str(FLASHPAIRFORMER))
    os.environ["AF3_FLASHPAIRFORMER"] = "off"
    try:
        import jax

        backend = jax.default_backend()
        label = display_mode or mode
        if backend != "gpu":
            print(f"[mn-cofolding-af3-opt] mode=off requested={label} backend={backend}", flush=True)
            return
        import af3_flashpairformer

        from af3_flashpairformer import compat

        af3_flashpairformer.install(mode)
        status = af3_flashpairformer.status()
        held = status["held"]
        requested = {
            "trimul": mode in ("trimul", "both"),
            "triatt": mode in ("triatt", "both"),
        }
        if all(not wanted or kind in held for kind, wanted in requested.items()):
            details = ",".join(f"{kind}:{reason}" for kind, reason in sorted(held.items()))
            print(
                f"[mn-cofolding-af3-opt] mode=stock requested={label} backend={backend} "
                f"compute_capability={status['compute_capability']} held={details}",
                flush=True,
            )
            return
        atexit.register(_report, af3_flashpairformer, compat, label)
        print(
            f"[mn-cofolding-af3-opt] mode={label} backend={backend} "
            f"jax={jax.__version__} kernels={status['tile_table']} held={held}",
            flush=True,
        )
    except Exception as exc:  # Fail the run instead of silently claiming an optimized run.
        raise RuntimeError(f"could not activate guarded AF3 optimization mode {mode!r}") from exc


def _report(af3_flashpairformer, compat, mode: str) -> None:
    print(
        f"[mn-cofolding-af3-opt] report mode={mode} "
        f"served={af3_flashpairformer.served_report()} "
        f"compatibility_fallbacks={compat.fallback_report()}",
        flush=True,
    )
    if mode.startswith("big"):
        print(f"[mn-cofolding-af3-opt] big_memory_report={BIG_MEMORY_REPORT}", flush=True)


def configure_big_pair_transition(config, *, rows: int = 64) -> tuple[object, object]:
    """Set AF3's built-in pair-transition row shard size for the big mode.

    This deliberately uses a config field that exists in the pinned 3.1.14
    model instead of copying the upstream toolkit's larger custom model
    rewrites. Smaller row blocks lower transition intermediates at the cost of
    extra dispatch and potentially longer runtime.
    """
    if rows < 1:
        raise ValueError("rows must be positive")
    global_config = getattr(config, "global_config", None)
    if global_config is None or not hasattr(global_config, "pair_transition_shard_spec"):
        raise RuntimeError(
            "big mode requires the AF3 GlobalConfig.pair_transition_shard_spec field"
        )
    previous = global_config.pair_transition_shard_spec
    selected = ((None, rows),)
    global_config.pair_transition_shard_spec = selected
    return previous, selected


_RELATIVE_FIELDS = ("token_index", "residue_index", "asym_id", "entity_id", "sym_id")


def _relative_encoding_rows(rows: dict, cols: dict, *, max_relative_idx: int = 32,
                            max_relative_chain: int = 2):
    """AF3's relative-position features for selected rows against all columns."""
    import jax
    import jax.numpy as jnp

    left_asym = rows["asym_id"][:, None]
    right_asym = cols["asym_id"][None, :]
    left_residue = rows["residue_index"][:, None]
    right_residue = cols["residue_index"][None, :]
    left_token = rows["token_index"][:, None]
    right_token = cols["token_index"][None, :]
    left_entity = rows["entity_id"][:, None]
    right_entity = cols["entity_id"][None, :]
    left_sym = rows["sym_id"][:, None]
    right_sym = cols["sym_id"][None, :]

    offset = jnp.clip(left_residue - right_residue + max_relative_idx,
                      0, 2 * max_relative_idx)
    same_asym = left_asym == right_asym
    offset = jnp.where(same_asym, offset, 2 * max_relative_idx + 1)
    features = [jax.nn.one_hot(offset, 2 * max_relative_idx + 2)]

    token_offset = jnp.clip(left_token - right_token + max_relative_idx,
                            0, 2 * max_relative_idx)
    same_residue = same_asym & (left_residue == right_residue)
    token_offset = jnp.where(same_residue, token_offset, 2 * max_relative_idx + 1)
    features.append(jax.nn.one_hot(token_offset, 2 * max_relative_idx + 2))

    same_entity = left_entity == right_entity
    features.append(same_entity.astype(features[0].dtype)[..., None])
    rel_chain = jnp.clip(left_sym - right_sym + max_relative_chain,
                         0, 2 * max_relative_chain)
    rel_chain = jnp.where(same_entity, rel_chain, 2 * max_relative_chain + 1)
    features.append(jax.nn.one_hot(rel_chain, 2 * max_relative_chain + 2))
    return jnp.concatenate(features, axis=-1)


def _pair_conditioning_rows(self, batch, embeddings, use_conditioning, *, rows: int):
    """Chunk official AF3's pair conditioning by token rows without changing parameters."""
    import jax
    import jax.numpy as jnp

    from alphafold3.model import model_config
    from alphafold3.model.network import diffusion_head, diffusion_transformer

    pair_embedding = use_conditioning * embeddings["pair"]
    token_features = batch.token_features
    columns = {name: getattr(token_features, name) for name in _RELATIVE_FIELDS}
    n_tokens = int(pair_embedding.shape[0])
    if n_tokens <= rows:
        return diffusion_head.DiffusionHead._mn_stock_conditioning(
            self, batch, embeddings, jnp.zeros(()), use_conditioning, parts="pair"
        )[1]

    n_blocks = math.ceil(n_tokens / rows)
    pair_channel = self.config.conditioning.pair_channel

    def make_block(block_index):
        start = jnp.minimum(block_index * rows, n_tokens - rows)
        pair_rows = jax.lax.dynamic_slice_in_dim(pair_embedding, start, rows, axis=0)
        row_features = {
            name: jax.lax.dynamic_slice_in_dim(value, start, rows, axis=0)
            for name, value in columns.items()
        }
        relative = _relative_encoding_rows(row_features, columns).astype(pair_rows.dtype)
        features = jnp.concatenate([pair_rows, relative], axis=-1)
        pair_cond = diffusion_head.hm.Linear(
            pair_channel, precision="highest", name="pair_cond_initial_projection"
        )(
            diffusion_head.hm.LayerNorm(
                use_fast_variance=False,
                create_offset=model_config.affine_norm(
                    self.global_config.model, "pair_cond_initial_norm"
                ),
                name="pair_cond_initial_norm",
            )(features)
        )
        for index in range(2):
            pair_cond += diffusion_transformer.transition_block(
                pair_cond,
                2,
                self.global_config,
                name=f"pair_transition_{index}",
            )
        return pair_cond

    # lax.map traces the Haiku modules once and loops over row slices at run
    # time, preserving a single copy of each stock parameter scope.
    blocks = jax.lax.map(make_block, jnp.arange(n_blocks, dtype=jnp.int32))
    remainder = n_tokens % rows
    if remainder:
        full_blocks = blocks[:-1].reshape((-1, n_tokens, pair_channel))
        tail = blocks[-1, rows - remainder :, :, :]
        result = jnp.concatenate([full_blocks, tail], axis=0)
    else:
        result = blocks.reshape((n_tokens, n_tokens, pair_channel))
    BIG_MEMORY_REPORT["conditioning_calls"] += 1
    BIG_MEMORY_REPORT["conditioning_shapes"].append(
        [n_tokens, int(pair_embedding.shape[1]), int(pair_embedding.shape[2])]
    )
    return result


def _chunked_sample(denoising_step, batch, key, config, global_config, *, chunk_size: int):
    """AF3 3.1.14 sampler in sample-sized passes, preserving stock random draws."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np

    from alphafold3.model import model_config
    from alphafold3.model.network import diffusion_head

    mask = batch.predicted_structure_info.atom_mask
    chai = global_config.model == "chai1"

    def apply_denoising_step(carry, step_noise):
        noise_level, noise_level_prev = step_noise
        sample_key, positions = carry
        sample_key, key_noise, key_aug = jax.random.split(sample_key, 3)
        positions = diffusion_head.random_augmentation(
            rng_key=key_aug, positions=positions, mask=mask
        )
        if chai:
            gamma = jnp.where(
                (noise_level_prev >= diffusion_head.CHAI_S_TMIN)
                & (noise_level_prev <= diffusion_head.CHAI_S_TMAX),
                min(diffusion_head.CHAI_S_CHURN / config.steps, 2.0**0.5 - 1.0),
                0.0,
            )
        else:
            gamma = config.gamma_0 * (noise_level > config.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)
        floor = 1e-6 if chai else 0.0
        noise_scale = config.noise_scale * jnp.sqrt(
            jnp.maximum(floor, t_hat**2 - noise_level_prev**2)
        )
        noise = jax.random.normal(key_noise, positions.shape) * noise_scale
        positions_noisy = positions + noise
        positions_denoised = denoising_step(positions_noisy, t_hat)
        if global_config.model in model_config.REALIGN_SAMPLER:
            positions_noisy = diffusion_head._kabsch(
                positions_noisy, positions_denoised, mask
            )
        grad = (positions_noisy - positions_denoised) / t_hat
        d_t = noise_level - t_hat
        if chai:
            positions_out = positions_noisy + d_t * grad
            grad2 = (positions_out - denoising_step(positions_out, noise_level)) / noise_level
            positions_out = jnp.where(
                noise_level > 0,
                positions_out + d_t * (grad2 + grad) / 2.0,
                positions_out,
            )
        else:
            positions_out = positions_noisy + config.step_scale * d_t * grad
        return (sample_key, positions_out), positions_out

    num_samples = int(config.num_samples)
    if chai:
        times = np.linspace(0.0, 1.0, 2 * config.steps + 1, dtype=np.float32)[1::2]
    else:
        times = np.linspace(0, 1, config.steps + 1, dtype=np.float32)
    noise_levels = np.asarray(
        diffusion_head.noise_schedule(
            times, smin=config.sigma_min, smax=config.sigma_max, p=config.rho
        ),
        dtype=np.float32,
    )
    if getattr(config, "max_sigma", 0.0):
        noise_levels = np.concatenate(
            [[config.max_sigma], noise_levels[noise_levels <= config.max_sigma]]
        )
    noise_levels = jnp.asarray(noise_levels, jnp.float32)

    key, noise_key = jax.random.split(key)
    # Draw once at the stock padded shape. This retains the same sample-specific
    # initial noise and key splitting as the unchunked 3.1.14 sampler.
    positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
    positions *= noise_levels[0]
    sample_keys = jax.random.split(key, num_samples)

    trajectories = []
    output_chunks = []
    for start, stop in sample_slices(num_samples, chunk_size):
        initial = (sample_keys[start:stop], positions[start:stop])
        vmapped_step = hk.vmap(
            apply_denoising_step,
            in_axes=(0, None),
            split_rng=(not hk.running_init()),
        )
        result, trajectory = hk.scan(
            vmapped_step,
            initial,
            (noise_levels[1:], noise_levels[:-1]),
            unroll=1,
        )
        output_chunks.append(result[1])
        if config.return_trajectory:
            trajectories.append(trajectory)

    positions_out = jnp.concatenate(output_chunks, axis=0)
    dense_mask = jnp.tile(mask[None], (num_samples, 1, 1))
    output = {"atom_positions": positions_out, "mask": dense_mask}
    if config.return_trajectory:
        output["trajectory"] = jnp.concatenate(trajectories, axis=1)
    BIG_MEMORY_REPORT["sampler_calls"] += 1
    BIG_MEMORY_REPORT["sampler_chunks"] += math.ceil(num_samples / chunk_size)
    BIG_MEMORY_REPORT["sampler_shapes"].append({
        "model": global_config.model,
        "samples": num_samples,
        "tokens": int(mask.shape[0]),
        "chunk": chunk_size,
    })
    return output


def _make_big_triangle_multiplication(base_cls):
    """Build an AF3-compatible triangle multiplication with bounded row tiles.

    The stock call materializes both gated projections and the complete
    triangle-einsum result at once. This version keeps one contracted half for
    the full plane, computes the other half and output in row blocks, and uses
    the original Haiku parameter names and AF3 scaling rules.
    """
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import tokamax

    from alphafold3.model import model_config
    from alphafold3.model.components import haiku_modules as hm

    def __call__(self, act, mask):
        if len(act.shape) != 3 or act.shape[0] != act.shape[1]:
            raise ValueError(
                "big triangle multiplication expects a square [N, N, C] pair tensor"
            )

        num_channels = int(act.shape[-1])
        num_residues = int(act.shape[0])
        hidden_dim = (
            num_channels
            if self.config.hidden_dim is None
            else int(self.config.hidden_dim)
        )
        equation = {
            "ikc,jkc->ijc": "cik,cjk->cij",
            "kjc,kic->ijc": "ckj,cki->cij",
        }.get(self.config.equation)
        if equation is None:
            raise ValueError(f"unsupported triangle equation: {self.config.equation!r}")
        outgoing = self.config.equation == "ikc,jkc->ijc"

        act = hm.LayerNorm(name="left_norm_input")(act)
        input_act = act
        mask = mask[None, ...]

        if self.config.use_glu_kernel:
            weights_projection, _ = hm.haiku_linear_get_params(
                act, num_output=hidden_dim * 2, name="projection"
            )
            weights_gate, _ = hm.haiku_linear_get_params(
                act,
                num_output=hidden_dim * 2,
                initializer=self.global_config.final_init,
                name="gate",
            )
            weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)

            def gated(values, values_mask):
                projection = tokamax.gated_linear_unit(
                    values, weights_glu, activation=jax.nn.sigmoid
                )
                return jnp.transpose(projection, (2, 0, 1)) * values_mask[None, ...]
        else:
            projection = hm.Linear(hidden_dim * 2, name="projection")
            gate = hm.Linear(
                hidden_dim * 2,
                name="gate",
                bias_init=1.0,
                initializer=self.global_config.final_init,
            )

            def gated(values, values_mask):
                projected = jnp.transpose(projection(values), (2, 0, 1))
                gate_values = jnp.transpose(gate(values), (2, 0, 1))
                return projected * jax.nn.sigmoid(gate_values) * values_mask[None, ...]

        center_norm = hm.LayerNorm(name="center_norm", axis=0, param_axis=0)
        output_projection = hm.Linear(
            num_channels,
            initializer=self.global_config.final_init,
            name="output_projection",
        )
        gating_linear = hm.Linear(
            num_channels,
            name="gating_linear",
            bias_init=1.0,
            initializer=self.global_config.final_init,
        )

        def halves(projected):
            projected = projected.reshape(
                hidden_dim, 2, *projected.shape[1:]
            )
            left, right = jnp.split(projected, 2, axis=1)
            return jnp.squeeze(left, axis=1), jnp.squeeze(right, axis=1)

        def finish(block, input_rows):
            if self.global_config.model in model_config.TRIANGLE_MUL_DIVIDE_BY_LENGTH:
                block = block / jnp.asarray(num_residues, block.dtype)
            block = center_norm(block)
            block = jnp.transpose(block, (1, 2, 0))
            block = output_projection(block)
            return block * jax.nn.sigmoid(gating_linear(input_rows))

        BIG_MEMORY_REPORT["trimul_chunk_calls"] += 1
        BIG_MEMORY_REPORT["trimul_chunk_shapes"].append(
            {
                "tokens": num_residues,
                "channels": num_channels,
                "hidden_dim": hidden_dim,
                "equation": self.config.equation,
                "rows": BIG_TRIMUL_ROWS,
            }
        )

        if num_residues <= BIG_TRIMUL_ROWS:
            left, right = halves(gated(act, mask[0]))
            return finish(jnp.einsum(equation, left, right), input_act)

        full_rows = (num_residues // BIG_TRIMUL_ROWS) * BIG_TRIMUL_ROWS
        full_block_count = full_rows // BIG_TRIMUL_ROWS
        starts = jnp.arange(full_block_count, dtype=jnp.int32) * BIG_TRIMUL_ROWS

        def project_rows(start, half_index):
            values = jax.lax.dynamic_slice_in_dim(
                act, start, BIG_TRIMUL_ROWS, axis=0
            )
            values_mask = jax.lax.dynamic_slice_in_dim(
                mask[0], start, BIG_TRIMUL_ROWS, axis=0
            )
            return halves(gated(values, values_mask))[half_index]

        if outgoing:
            # For out[c, i, j] = sum_k a[c, i, k] b[c, j, k], retain b
            # for the full plane and recalculate a for each output-row tile.
            full_blocks = jax.lax.map(
                lambda start: project_rows(start, 1), starts
            )
        else:
            # For in[c, i, j] = sum_k a[c, k, j] b[c, k, i], retain a
            # for the full plane and calculate the selected b columns per tile.
            full_blocks = jax.lax.map(
                lambda start: project_rows(start, 0), starts
            )
        full = jnp.transpose(full_blocks, (1, 0, 2, 3)).reshape(
            hidden_dim, full_rows, num_residues
        )
        if full_rows < num_residues:
            tail = halves(
                gated(act[full_rows:], mask[0, full_rows:])
            )[1 if outgoing else 0]
            full = jnp.concatenate([full, tail], axis=1)

        def output_rows(start):
            if outgoing:
                values = jax.lax.dynamic_slice_in_dim(
                    act, start, BIG_TRIMUL_ROWS, axis=0
                )
                values_mask = jax.lax.dynamic_slice_in_dim(
                    mask[0], start, BIG_TRIMUL_ROWS, axis=0
                )
                left = halves(gated(values, values_mask))[0]
                block = jnp.einsum(equation, left, full)
            else:
                values = jax.lax.dynamic_slice_in_dim(
                    act, start, BIG_TRIMUL_ROWS, axis=1
                )
                values_mask = jax.lax.dynamic_slice_in_dim(
                    mask[0], start, BIG_TRIMUL_ROWS, axis=1
                )
                right = halves(gated(values, values_mask))[1]
                block = jnp.einsum(equation, full, right)
            input_rows = jax.lax.dynamic_slice_in_dim(
                input_act, start, BIG_TRIMUL_ROWS, axis=0
            )
            return finish(block, input_rows)

        output_blocks = jax.lax.map(output_rows, starts).reshape(
            full_rows, num_residues, num_channels
        )
        if full_rows == num_residues:
            return output_blocks
        tail_start = full_rows
        if outgoing:
            left = halves(gated(act[tail_start:], mask[0, tail_start:]))[0]
            tail_block = jnp.einsum(equation, left, full)
        else:
            right = halves(gated(act[:, tail_start:, :], mask[0, :, tail_start:]))[1]
            tail_block = jnp.einsum(equation, full, right)
        tail_output = finish(tail_block, input_act[tail_start:])
        return jnp.concatenate([output_blocks, tail_output], axis=0)

    patched_cls = type(
        "TriangleMultiplication",
        (base_cls,),
        {
            "__call__": __call__,
            "__doc__": base_cls.__doc__,
            "__module__": base_cls.__module__,
            "__mn_big_trimul_chunk__": True,
        },
    )
    return patched_cls


def _install_big_triangle_multiplication() -> None:
    from alphafold3.model.network import modules

    base_cls = modules.TriangleMultiplication
    if getattr(base_cls, "__mn_big_trimul_chunk__", False):
        return
    modules.TriangleMultiplication = _make_big_triangle_multiplication(base_cls)


def _pair_logits_in_rows(pair_cond, scale, projection, rows: int, *, jax, jnp):
    """Project AF3 pair conditioning to diffusion logits without full pair_act."""
    num_rows = int(pair_cond.shape[0])
    dtype = pair_cond.dtype
    work_dtype = (
        jnp.float32 if dtype in (jnp.bfloat16, jnp.float16) else dtype
    )

    def normalize(values):
        values = values.astype(work_dtype)
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.var(values, axis=-1, keepdims=True)
        inv = scale * jax.lax.rsqrt(variance + jnp.asarray(1e-5, variance.dtype))
        return (inv * (values - mean)).astype(dtype)

    if num_rows <= rows:
        return projection(normalize(pair_cond))

    # Keep the Haiku linear projection in the stack's parameter scope. A JAX
    # control-flow map inside Haiku's layer_stack loses that scope, so this
    # short loop is unrolled while only one normalized row block is live.
    outputs = [
        projection(normalize(pair_cond[start : start + rows]))
        for start in range(0, num_rows, rows)
    ]
    return jnp.concatenate(outputs, axis=0)


def _make_big_diffusion_transformer(base_cls):
    """Shard official AF3 diffusion pair logits while keeping stock parameters."""
    import jax
    import jax.numpy as jnp

    from alphafold3.model.network import diffusion_transformer as dt

    stock_call = base_cls.__call__

    def __call__(self, act, mask, single_cond, pair_cond, extra_pair_bias=None):
        if (
            self.global_config.model != "alphafold3"
            or pair_cond is None
            or extra_pair_bias is not None
        ):
            return stock_call(
                self, act, mask, single_cond, pair_cond, extra_pair_bias
            )

        assert self.config.num_blocks % self.config.super_block_size == 0
        num_super_blocks = self.config.num_blocks // self.config.super_block_size
        pair_dtype = pair_cond.dtype if pair_cond is not None else jnp.float32
        pair_param_dtype = (
            jnp.float32
            if pair_dtype in (jnp.bfloat16, jnp.float16)
            else pair_dtype
        )
        with dt.hk.experimental.name_scope("pair_input_layer_norm"):
            pair_scale = dt.hk.get_parameter(
                "scale",
                (int(pair_cond.shape[-1]),),
                pair_param_dtype,
                init=jnp.ones,
            )
        BIG_MEMORY_REPORT["diffusion_logits_calls"] += 1
        BIG_MEMORY_REPORT["diffusion_logits_shapes"].append(
            [int(size) for size in pair_cond.shape]
        )

        def block(values, pair_logits):
            values += dt.self_attention(
                values,
                mask,
                pair_logits,
                self.config.attention,
                self.global_config,
                single_cond,
                name=self.name,
            )
            values += dt.transition_block(
                values,
                self.config.num_intermediate_factor,
                self.global_config,
                single_cond,
                name=self.name,
                mask=dt._trans_mask(self.global_config, mask),
            )
            return values, None

        def super_block(values):
            projection = dt.hm.Linear(
                (self.config.super_block_size, self.config.attention.num_head),
                name="pair_logits_projection",
            )
            pair_logits = _pair_logits_in_rows(
                pair_cond,
                pair_scale,
                projection,
                BIG_PAIR_ROWS,
                jax=jax,
                jnp=jnp,
            )
            pair_logits = jnp.transpose(pair_logits, [2, 3, 0, 1])
            return dt._stack(
                self.config.super_block_size,
                block,
                self.config.block_remat,
                with_per_layer_inputs=True,
            )(values, pair_logits)

        return dt._stack(
            num_super_blocks,
            super_block,
            False,
            with_per_layer_inputs=True,
        )(act)[0]

    return type(
        "Transformer",
        (base_cls,),
        {
            "__call__": __call__,
            "__doc__": base_cls.__doc__,
            "__module__": base_cls.__module__,
            "__mn_big_pair_logits__": True,
        },
    )


def _install_big_diffusion_transformer() -> None:
    from alphafold3.model.network import diffusion_transformer

    base_cls = diffusion_transformer.Transformer
    if getattr(base_cls, "__mn_big_pair_logits__", False):
        return
    diffusion_transformer.Transformer = _make_big_diffusion_transformer(base_cls)


def _install_big_model_memory_hooks() -> None:
    """Install 3.1.14-specific pair-conditioning and sampler memory levers.

    The upstream toolkit hooks are pinned to a different ``sample`` and
    ``_conditioning`` API. These replacements are deliberately limited to the
    model-specific schedules. Pair conditioning is chunked only for official
    AF3 because its clone paths have different layouts.
    """
    from alphafold3.model.network import diffusion_head

    cls = diffusion_head.DiffusionHead
    if not hasattr(cls, "_mn_stock_conditioning"):
        stock_conditioning = cls._conditioning
        cls._mn_stock_conditioning = stock_conditioning

        def conditioning(self, batch, embeddings, noise_level, use_conditioning, parts="both"):
            if self.global_config.model == "alphafold3" and parts == "pair":
                return None, _pair_conditioning_rows(
                    self, batch, embeddings, use_conditioning, rows=BIG_PAIR_ROWS
                )
            return stock_conditioning(
                self, batch, embeddings, noise_level, use_conditioning, parts=parts
            )

        conditioning.__wrapped__ = stock_conditioning
        cls._conditioning = conditioning

    if not hasattr(diffusion_head.sample, "__mn_big_sample_chunk__"):
        stock_sample = diffusion_head.sample

        def sample(denoising_step, batch, key, config, global_config=None):
            if (
                global_config is None
                or int(config.num_samples) <= BIG_SAMPLE_CHUNK
            ):
                return stock_sample(
                    denoising_step=denoising_step,
                    batch=batch,
                    key=key,
                    config=config,
                    global_config=global_config,
                )
            return _chunked_sample(
                denoising_step,
                batch,
                key,
                config,
                global_config,
                chunk_size=BIG_SAMPLE_CHUNK,
            )

        sample.__mn_big_sample_chunk__ = True
        sample.__wrapped__ = stock_sample
        diffusion_head.sample = sample

    print(
        "[mn-cofolding-af3-opt] big memory hooks enabled "
        f"samples_per_pass={BIG_SAMPLE_CHUNK} "
        f"official_af3_pair_conditioning_rows={BIG_PAIR_ROWS} "
        f"official_af3_diffusion_pair_logits_rows={BIG_PAIR_ROWS} "
        f"triangle_multiplication_rows={BIG_TRIMUL_ROWS}",
        flush=True,
    )


def _install_big_memory_mode() -> None:
    try:
        from alphafold3.model import model

        config_type = model.Model.Config
        probe = config_type()
        previous, selected = configure_big_pair_transition(probe)
        if getattr(config_type.__init__, "__mn_cofolding_big_mode__", False):
            return
        stock_init = config_type.__init__

        def init_with_big_pair_shards(self, *args, **kwargs):
            stock_init(self, *args, **kwargs)
            configure_big_pair_transition(self)

        init_with_big_pair_shards.__mn_cofolding_big_mode__ = True
        init_with_big_pair_shards.__wrapped__ = stock_init
        config_type.__init__ = init_with_big_pair_shards
        _install_big_triangle_multiplication()
        _install_big_diffusion_transformer()
        _install_big_model_memory_hooks()
        print(
            "[mn-cofolding-af3-opt] big mode enabled "
            f"pair_transition_shard_spec={selected!r} (stock={previous!r})",
            flush=True,
        )
    except Exception as exc:
        raise RuntimeError("could not activate AF3 big-mode pair transition sharding") from exc


if __name__ == "__main__":
    main()
