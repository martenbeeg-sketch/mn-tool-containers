#!/usr/bin/env python3
"""MN Cofolding runner with guarded FlashPairformer kernels for AF3 3.1.14."""

from __future__ import annotations

import atexit
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


def main() -> None:
    arguments = sys.argv[1:]
    helpers = runpy.run_path(str(STOCK_RUNNER), run_name="mn_cofolding_af3_runner_helpers")
    input_path = helpers["_json_path"](arguments)
    if input_path is not None:
        helpers["_prepare_ccd"](input_path)
    elif "--help" in arguments or "-h" in arguments:
        helpers["_prepare_ccd"](None)
    helpers["_install_msa_cache"]()

    mode = os.environ.get("MN_COFOLDING_AF3_OPT_MODE", "off").strip().lower()
    if mode not in MODES:
        raise SystemExit(f"MN_COFOLDING_AF3_OPT_MODE must be one of {sorted(MODES)}, got {mode!r}")

    if mode != "off" and "--jax_backend=cpu" not in arguments and "--jax_backend" not in arguments:
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
        print(
            "[mn-cofolding-af3-opt] big mode enabled "
            f"pair_transition_shard_spec={selected!r} (stock={previous!r})",
            flush=True,
        )
    except Exception as exc:
        raise RuntimeError("could not activate AF3 big-mode pair transition sharding") from exc


if __name__ == "__main__":
    main()
