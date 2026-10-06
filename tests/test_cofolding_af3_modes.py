from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "protein-design"
    / "cofolding-af3"
    / "optimized"
    / "cofolding-af3-memory-runner.py"
)
SPEC = importlib.util.spec_from_file_location("cofolding_af3_opt_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_public_modes_are_off_fast_and_big():
    assert RUNNER.MODES == {"off", "exact", "fast", "big"}
    assert RUNNER.KERNEL_SETS == {"trimul", "triatt", "both"}


def test_big_mode_uses_the_af3_config_shard_setting():
    config = SimpleNamespace(
        global_config=SimpleNamespace(pair_transition_shard_spec=((2048, None), (None, 1024)))
    )
    previous, selected = RUNNER.configure_big_pair_transition(config)
    assert previous == ((2048, None), (None, 1024))
    assert selected == ((None, 256),)
    assert config.global_config.pair_transition_shard_spec == selected


def test_big_mode_refuses_an_unknown_af3_config_and_bad_row_size():
    with pytest.raises(RuntimeError, match="pair_transition_shard_spec"):
        RUNNER.configure_big_pair_transition(SimpleNamespace(global_config=SimpleNamespace()))
    with pytest.raises(ValueError, match="positive"):
        RUNNER.configure_big_pair_transition(
            SimpleNamespace(global_config=SimpleNamespace(pair_transition_shard_spec=())),
            rows=0,
        )


def test_shared_kernel_tile_buckets_cover_af3_shapes():
    buckets = RUNNER.kernel_tile_buckets()
    assert buckets[0] == 64
    assert buckets[-1] == 5120
    assert all(size % 64 == 0 for size in buckets)
    assert RUNNER.kernel_tile_buckets(tile=128, maximum=512) == [128, 256, 384, 512]


def test_big_sample_passes_split_without_dropping_the_final_sample():
    assert RUNNER.sample_slices(5, 1) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert RUNNER.sample_slices(5, 2) == [(0, 2), (2, 4), (4, 5)]
    with pytest.raises(ValueError, match="positive"):
        RUNNER.sample_slices(0, 1)


def test_chunked_sampler_imports_haiku_for_its_multi_sample_path():
    module = ast.parse(RUNNER_PATH.read_text())
    sampler = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_chunked_sample"
    )
    assert any(
        isinstance(node, ast.Import)
        and any(alias.name == "haiku" and alias.asname == "hk" for alias in node.names)
        for node in sampler.body
    )


def test_active_memory_recipe_uses_the_reviewed_pair_shard_size():
    assert RUNNER.BIG_PAIR_ROWS == 256
