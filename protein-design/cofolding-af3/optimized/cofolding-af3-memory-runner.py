#!/usr/bin/env python3
"""MN Cofolding runner with guarded FlashPairformer kernels for AF3 3.1.14."""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import runpy
import sys
from types import SimpleNamespace
from pathlib import Path


STOCK_RUNNER = Path("/usr/local/bin/cofolding-af3-runner-stock.py")
AF3_MODEL_RUNNER = Path("/opt/alphafold3/run_alphafold.py")
OPT_ROOT = Path("/opt/mn-cofolding-af3-opt")
FLASHPAIRFORMER = OPT_ROOT / "flashpairformer"
OPT_CORE = OPT_ROOT
MODES = {"off", "exact", "fast", "big"}
KERNEL_SETS = {"trimul", "triatt", "both"}
BIG_SAMPLE_CHUNK = 1
BIG_PAIR_ROWS = 256
BIG_MEMORY_REPORT = {
    "sampler_calls": 0,
    "sampler_chunks": 0,
    "sampler_shapes": [],
    "conditioning_calls": 0,
    "conditioning_shapes": [],
}
AF3_3_1_14_ROWPAIR_PIN = {
    "atom_cross_attention": "1163e2f2cf5bef3617e61bf3fb99aa619ff1701a2b7fd1b4da424266b324633a",
    "confidence_head": "599ee626175c1f22c7798747ed101508ea3a2089753067e9aa27ba01b6ef6e56",
    "diffusion_head": "38a8feae5b0aab88569c4fedbe35942280823631468c239c487816d0397af93f",
    "diffusion_transformer": "47695d88735ff3527f7e8825510e3653c5dca1bc5242bdbbbc1c21bfa2c4c6bd",
    "distogram_head": "c9d2f092418e80bb3882de8c8611f230c9e51230f03047479e57076ad1c976f6",
    "evoformer": "182d2a1711bc7b71dc43e51eb19a35b74d4423113aebc36288af9ae221f33868",
    "feat_batch": "5e03fffd603db875ad3870586b6fb7a60c3970b6b996ba6d229fecb196340141",
    "haiku_modules": "ebc80aac0b672fef007025973886bcd3bb3bc1bd00a12629406aff5bb6821f2c",
    "mapping": "216ab9188061de38b8a2a15a95c3472ca5b9f414a9779fb4b7fe43bf22b07d80",
    "model": "9caf7c92c1839cc532eecb22fff5a76fbfd1db64f76d9dba9d4ff468c9121ffa",
    "modules": "ef5fb6dfd4cb04ae307406886dbb7625f36a8e2ecd0e7dbb90f2bf477035f161",
}
AF3_3_1_14_RUNNER_SHA256 = "b984b1a68282181d3c66b3b07e9e145f8c5a15e31d892b3d02edefc3ace08c27"
AF3_3_1_14_MODEL_CONFIG_SHA256 = "8480f28a8f7cd23bbab907a0886f5905debe5239498ddf9fb13afd71ab9b78b4"


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


def main() -> None:
    arguments = sys.argv[1:]
    mode = os.environ.get("MN_COFOLDING_AF3_OPT_MODE", "off").strip().lower()
    if mode not in MODES:
        raise SystemExit(f"MN_COFOLDING_AF3_OPT_MODE must be one of {sorted(MODES)}, got {mode!r}")
    try:
        n_gpu = int(os.environ.get("MN_COFOLDING_AF3_N_GPU", "1"))
    except ValueError as exc:
        raise SystemExit("MN_COFOLDING_AF3_N_GPU must be 1 or 2") from exc
    model_name = _argument_value(arguments, "--model", "")
    if n_gpu not in (1, 2):
        raise SystemExit(f"MN_COFOLDING_AF3_N_GPU must be 1 or 2, got {n_gpu}")
    if n_gpu > 1 and (mode != "big" or model_name != "alphafold3"):
        raise SystemExit("AF3 two-GPU row sharding is restricted to big mode with official alphafold3 weights")
    if mode == "big":
        # A larger contiguous BFC pool avoids the fragmentation observed when
        # dynamic allocation was used for the 1,927-aa result materialization.
        # Keep preallocation on; nvidia-smi then reports the reservation, not
        # the model's live-tensor peak.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
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

    if mode == "exact":
        backend_flag = next((arg.split("=", 1)[1] for arg in arguments if arg.startswith("--jax_backend=")), None)
        if backend_flag == "cpu" or any(
            arg == "--jax_backend" and index + 1 < len(arguments) and arguments[index + 1] == "cpu"
            for index, arg in enumerate(arguments)
        ):
            raise SystemExit("AF3 exact mode requires a GPU backend")
        _install_exact()
    elif mode in {"fast", "big"} and "--jax_backend=cpu" not in arguments and "--jax_backend" not in arguments:
        kernel_set = os.environ.get("MN_COFOLDING_AF3_OPT_KERNELS", "both").strip().lower()
        if kernel_set not in KERNEL_SETS:
            raise SystemExit(
                f"MN_COFOLDING_AF3_OPT_KERNELS must be one of {sorted(KERNEL_SETS)}, got {kernel_set!r}"
            )
        label = mode if kernel_set == "both" else f"{mode}:{kernel_set}"
        _install_flashpairformer(kernel_set, display_mode=label)
        if mode == "big":
            _install_big_memory_mode()
    else:
        print(f"[mn-cofolding-af3-opt] mode=off requested={mode}", flush=True)

    if n_gpu > 1:
        _run_rowpair_two_gpu(helpers, arguments, n_gpu=n_gpu)
        return

    sys.argv = [str(helpers["RUNNER"]), *arguments]
    runpy.run_path(str(helpers["RUNNER"]), run_name="__main__")


def _argument_value(arguments: list[str], name: str, default: str) -> str:
    """Read one argparse/absl value in either ``--name=value`` or split form."""
    prefix = f"{name}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument.split("=", 1)[1]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def _run_rowpair_two_gpu(helpers: dict, arguments: list[str], *, n_gpu: int) -> None:
    """Install the kit's row-sharded trunk recipe on the official 3.1.14 source.

    The 3.1.14 image adds model-family branches and split diffusion conditioning
    that are outside the kit's 3.1.4 head transcription. This port pins the
    actual image sources and installs the reviewed trunk sites only; the stock
    3.1.14 Model call, diffusion, confidence, and output code remain in charge.
    """
    if n_gpu != 2:
        raise RuntimeError(f"rowpair port currently supports exactly 2 GPUs, got {n_gpu}")
    if _argument_value(arguments, "--model", "") != "alphafold3":
        raise RuntimeError("rowpair port is source-reviewed for official alphafold3 weights only")
    if any(
        arg == "--jax_backend=cpu"
        or (arg == "--jax_backend" and index + 1 < len(arguments) and arguments[index + 1] == "cpu")
        for index, arg in enumerate(arguments)
    ):
        raise RuntimeError("two-GPU row sharding requires the GPU backend")

    sys.path.insert(0, str(OPT_ROOT))
    import cofolding_af3_rowpair as rowpair_adapter
    from opt_core.mem.rowpair_jax import alphafold3 as recipe
    from opt_core.mem.rowpair_jax import transition

    source_install = recipe.install
    recipe.PIN = {
        **recipe.PIN,
        "tree": "official AlphaFold 3 3.1.14 image; reviewed trunk port",
        "image": "mn-cofolding-af3-memory:3.1.14-cu13",
        "files": AF3_3_1_14_ROWPAIR_PIN,
    }

    def install_official_3114(rmesh, patches, **kwargs):
        kwargs["pin"] = AF3_3_1_14_ROWPAIR_PIN
        kwargs["sites"] = ("trunk",)
        kwargs["heads"] = "replicated"
        kwargs["conf"] = "replicated"
        return source_install(rmesh, patches, **kwargs)

    recipe.install = install_official_3114
    try:
        rowpair_adapter.install(n_gpu, heads="replicated", conf="replicated", schedule="ring")
    finally:
        recipe.install = source_install

    # Keep the model library's own 3.1.14 __call__ and new diffusion path. The
    # only model-level change is constraining the recycled pair carry to the
    # same mesh sharding as the row-sharded Evoformer output.
    model_module = recipe.library()["model"]
    modules_module = recipe.library()["modules"]
    from alphafold3.model import model_config

    model_config_sha = hashlib.sha256(Path(model_config.__file__).read_bytes()).hexdigest()
    if model_config_sha != AF3_3_1_14_MODEL_CONFIG_SHA256:
        raise RuntimeError(
            "AF3 rowpair model-config source guard refused model_config.py: "
            f"sha256={model_config_sha}; expected={AF3_3_1_14_MODEL_CONFIG_SHA256}"
        )
    # The 3.1.14 model-family extension uses this flag; an official AF3 config
    # has no such field, and its correct family value is False.
    if not hasattr(model_config.GlobalConfig, "of3_weights"):
        model_config.GlobalConfig.of3_weights = False
    stock_hk = model_module.hk

    # AF3 3.1.14 added the inference-time `use_dropout` keyword to calls of
    # PairFormerIteration and EvoformerIteration. The official prediction
    # runner always supplies the literal False; the kit's pinned 3.1.4
    # transcription predates that keyword. Accept the 3.1.14 boundaries while
    # refusing stochastic dropout, which this inference-only rowpair port does
    # not transcribe.
    rowpair_pairformer_call = modules_module.PairFormerIteration.__call__
    rowpair_evoformer_call = modules_module.EvoformerIteration.__call__

    def _pairformer_call_3114(self, *args, use_dropout=False, **kwargs):
        if use_dropout:
            raise RuntimeError("AF3 rowpair big mode supports inference with use_dropout=False only")
        return rowpair_pairformer_call(self, *args, **kwargs)

    def _evoformer_call_3114(self, *args, use_dropout=False, **kwargs):
        if use_dropout:
            raise RuntimeError("AF3 rowpair big mode supports inference with use_dropout=False only")
        return rowpair_evoformer_call(self, *args, **kwargs)

    _pairformer_call_3114.__wrapped__ = rowpair_pairformer_call
    _evoformer_call_3114.__wrapped__ = rowpair_evoformer_call
    modules_module.PairFormerIteration.__call__ = _pairformer_call_3114
    modules_module.EvoformerIteration.__call__ = _evoformer_call_3114

    class _RowpairModelHaiku:
        def __getattr__(self, name):
            return getattr(stock_hk, name)

        def fori_loop(self, lower, upper, body_fun, initial_value):
            if not (
                isinstance(initial_value, (tuple, list))
                and len(initial_value) == 2
                and isinstance(initial_value[0], dict)
                and "pair" in initial_value[0]
            ):
                return stock_hk.fori_loop(lower, upper, body_fun, initial_value)
            rmesh = rowpair_adapter._STATE["rmesh"]
            embeddings, key = initial_value
            embeddings = transition.constrain_carry(dict(embeddings), rmesh)
            embeddings, key = stock_hk.fori_loop(lower, upper, body_fun, (embeddings, key))
            embeddings = transition.constrain_carry(dict(embeddings), rmesh)
            return embeddings, key

    model_module.hk = _RowpairModelHaiku()
    runner_sha = hashlib.sha256(AF3_MODEL_RUNNER.read_bytes()).hexdigest()
    if runner_sha != AF3_3_1_14_RUNNER_SHA256:
        raise RuntimeError(
            "AF3 rowpair runner source guard refused run_alphafold.py: "
            f"sha256={runner_sha}; expected={AF3_3_1_14_RUNNER_SHA256}"
        )
    official_runner = runpy.run_path(
        str(AF3_MODEL_RUNNER), run_name="mn_cofolding_af3_3114_model_runner"
    )
    runner_module = SimpleNamespace(**official_runner)
    _bind_3114_rowpair_runner(runner_module, rowpair_adapter)
    print(
        "[mn-cofolding-af3-opt] big devices=2 sharding=rowpair model=alphafold3 "
        "trunk=row_sharded heads=stock-3.1.14",
        flush=True,
    )
    sys.argv = [str(AF3_MODEL_RUNNER), *arguments]
    from absl import app

    app.run(official_runner["main"])


def _bind_3114_rowpair_runner(runner_module: SimpleNamespace, rowpair_adapter) -> None:
    """Bind rowpair to AF3 3.1.14's actual ModelRunner API.

    The kit adapter targets a runner that exposes ``_jitted_apply``. The
    3.1.14 runner instead caches its forward callable in ``ModelRunner._model``
    and uses a cached ``_jitted_apply`` only for its optional lower-cache path.
    Keep that runner version pinned above, and replace only those two
    boundaries: a mesh-aware Haiku apply plus replicated input placement.
    """
    import functools

    from opt_core.mem.rowpair_jax import haiku as rowpair_haiku, shard

    model_runner = runner_module.ModelRunner
    if not hasattr(model_runner, "run_inference") or "_model" not in model_runner.__dict__:
        raise RuntimeError("AF3 rowpair runner adapter does not match the pinned 3.1.14 ModelRunner")

    rmesh = rowpair_adapter._STATE["rmesh"]
    hk = runner_module.hk
    model = runner_module.model

    def _mesh_model(self):
        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch, use_dropout=self._use_dropout)

        apply_fn = forward_fn.apply
        if not runner_module._NOJIT.value:
            apply_fn = rowpair_haiku.jit_apply(apply_fn, rmesh)
        self._preinit_tokamax_context()
        if runner_module._LOWERCACHE_DIR.value and not runner_module._NOJIT.value:
            self._jitted_apply = apply_fn
            return self._lowercached_model
        return functools.partial(apply_fn, self.model_params)

    mesh_model = functools.cached_property(_mesh_model)
    mesh_model.__set_name__(model_runner, "_model")
    model_runner._model = mesh_model

    stock_run_inference = model_runner.run_inference
    jax = runner_module.jax

    def run_inference(self, featurised_example, rng_key):
        original_device_put = jax.device_put
        replicated = shard.named(rmesh, shard.replicated_spec())

        def put_replicated(value, device=None, *args, **kwargs):
            return original_device_put(value, replicated)

        jax.device_put = put_replicated
        try:
            return stock_run_inference(self, featurised_example, rng_key)
        finally:
            jax.device_put = original_device_put

    model_runner.run_inference = run_inference
    print(
        "[mn-cofolding-af3-opt] ROWPAIR runner bound: "
        "ModelRunner._model=jit_apply(mesh) run_inference=replicated(mesh)",
        flush=True,
    )


def _install_exact() -> None:
    """Install the kit's bitwise GLU fusion, guarded for this AF3 source and stack."""
    import hashlib
    import importlib.metadata

    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(OPT_ROOT))
    from opt_core.kernels import pallas_glut

    from alphafold3.model import model_config
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import modules

    expected_modules_sha = "ef5fb6dfd4cb04ae307406886dbb7625f36a8e2ecd0e7dbb90f2bf477035f161"
    with open(modules.__file__, "rb") as stream:
        modules_sha = hashlib.sha256(stream.read()).hexdigest()
    jax_version = jax.__version__
    tokamax_version = importlib.metadata.version("tokamax")
    if modules_sha != expected_modules_sha:
        raise RuntimeError(
            "AF3 exact mode source guard refused modules.py: "
            f"sha256={modules_sha}; expected={expected_modules_sha}"
        )
    if jax_version != "0.11.1" or tokamax_version != "0.0.12":
        raise RuntimeError(
            "AF3 exact mode stack guard refused: "
            f"jax={jax_version}, tokamax={tokamax_version}; expected jax=0.11.1, tokamax=0.0.12"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"AF3 exact mode needs a GPU backend, got {jax.default_backend()}")

    stock_class = modules.TriangleMultiplication
    counts = {"traced_glut_sites": 0, "traced_stock_sites": 0, "aside": {}}

    class ExactTriangleMultiplication(stock_class):
        def __call__(self, act, mask):
            channels = int(act.shape[-1])
            hidden_dim = int(channels if self.config.hidden_dim is None else self.config.hidden_dim)
            eligible = (
                self.config.use_glu_kernel
                and channels == hidden_dim == 128
                and act.dtype == jnp.bfloat16
            )
            if not eligible:
                if self.config.use_glu_kernel:
                    reason = f"shape_or_dtype:{channels}:{hidden_dim}:{act.dtype}"
                    counts["aside"][reason] = counts["aside"].get(reason, 0) + 1
                counts["traced_stock_sites"] += 1
                return super().__call__(act, mask)

            mask_b = mask[None, ...]
            act = hm.LayerNorm(name="left_norm_input")(act)
            input_act = act
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
            if mask.dtype == act.dtype:
                projection = pallas_glut.glu_transposed_masked(
                    act, weights_glu, mask, activation=jax.nn.sigmoid
                )
            else:
                projection = pallas_glut.glu_transposed_masked(
                    act, weights_glu, None, activation=jax.nn.sigmoid
                )
                projection *= mask_b
            counts["traced_glut_sites"] += 1

            projection = projection.reshape(hidden_dim, 2, *projection.shape[1:])
            a, b = jnp.split(projection, 2, axis=1)
            a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)
            equation = {
                "ikc,jkc->ijc": "cik,cjk->cij",
                "kjc,kic->ijc": "ckj,cki->cij",
            }[self.config.equation]
            act = jnp.einsum(equation, a, b)
            num_residues = act.shape[-1]
            if self.global_config.model in model_config.TRIANGLE_MUL_DIVIDE_BY_LENGTH:
                act = act / jnp.asarray(num_residues, act.dtype)
            act = hm.LayerNorm(name="center_norm", axis=0, param_axis=0)(act)
            act = jnp.transpose(act, (1, 2, 0))
            act = hm.Linear(
                channels,
                initializer=self.global_config.final_init,
                name="output_projection",
            )(act)
            gate_out = hm.Linear(
                channels,
                name="gating_linear",
                bias_init=1.0,
                initializer=self.global_config.final_init,
            )(input_act)
            act *= jax.nn.sigmoid(gate_out)
            return act

    ExactTriangleMultiplication.__name__ = "TriangleMultiplication"
    ExactTriangleMultiplication.__qualname__ = "TriangleMultiplication"
    modules.TriangleMultiplication = ExactTriangleMultiplication
    atexit.register(
        lambda: print(
            "[mn-cofolding-af3-opt] exact report "
            f"traced_glut_sites={counts['traced_glut_sites']} "
            f"traced_stock_sites={counts['traced_stock_sites']} aside={counts['aside']}",
            flush=True,
        )
    )
    print(
        "[mn-cofolding-af3-opt] mode=exact backend=gpu "
        f"jax={jax_version} tokamax={tokamax_version} modules_sha256={modules_sha} "
        "kernel=pallas_glut channels=128 dtype=bfloat16 attention=stock",
        flush=True,
    )


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


def configure_big_pair_transition(config, *, rows: int = 256) -> tuple[object, object]:
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
        f"official_af3_pair_conditioning_rows={BIG_PAIR_ROWS}",
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
