from __future__ import annotations

"""Validation and result writing for the versioned ``mn-predict batch`` API."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_VERSION = "mn-predict/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MODES = {"off", "exact", "fast", "big"}


class BatchRequestError(ValueError):
    """Raised when a batch manifest cannot be executed safely."""


@dataclass(frozen=True)
class BatchItem:
    item_id: str
    input_path: Path
    template_path: Path | None = None


@dataclass(frozen=True)
class BatchRequest:
    batch_id: str
    engine: str
    mode: str
    gpu_count: int
    samples_per_item: int
    seed: int
    items: tuple[BatchItem, ...]
    options: dict[str, Any]


def load_request(path: Path, *, expected_engine: str) -> BatchRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchRequestError(f"cannot read JSON batch manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BatchRequestError("batch manifest must contain a JSON object")
    if payload.get("api_version") != API_VERSION:
        raise BatchRequestError(
            f"api_version must be {API_VERSION!r}; got {payload.get('api_version')!r}"
        )
    engine = str(payload.get("engine") or "").strip()
    if engine != expected_engine:
        raise BatchRequestError(
            f"manifest engine {engine!r} does not match this image's engine {expected_engine!r}"
        )
    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in _MODES:
        raise BatchRequestError(f"unsupported mode {mode!r}; expected one of {sorted(_MODES)}")
    batch_id = str(payload.get("batch_id") or "").strip()
    if not batch_id or not _SAFE_ID.fullmatch(batch_id):
        raise BatchRequestError("batch_id must contain only letters, numbers, dot, underscore, or hyphen")
    try:
        gpu_count = int(payload.get("gpu_count", 1))
        samples_per_item = int(payload.get("samples_per_item", 1))
        seed = int(payload.get("seed", 1))
    except (TypeError, ValueError) as exc:
        raise BatchRequestError("gpu_count, samples_per_item, and seed must be integers") from exc
    if gpu_count < 1:
        raise BatchRequestError("gpu_count must be at least one")
    if mode != "big" and gpu_count != 1:
        raise BatchRequestError(f"mode {mode!r} supports exactly one GPU in this image")
    if samples_per_item < 1:
        raise BatchRequestError("samples_per_item must be at least one")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise BatchRequestError("items must be a non-empty list")
    items: list[BatchItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise BatchRequestError(f"items[{index}] must be an object")
        item_id = str(raw.get("id") or "").strip()
        if not item_id or not _SAFE_ID.fullmatch(item_id) or item_id in {".", ".."}:
            raise BatchRequestError(
                f"items[{index}].id must be a unique path-safe identifier"
            )
        if item_id in seen:
            raise BatchRequestError(f"duplicate item id {item_id!r}")
        seen.add(item_id)
        raw_input = str(raw.get("input") or "").strip()
        if not raw_input:
            raise BatchRequestError(f"items[{index}].input is required")
        input_path = Path(raw_input).expanduser().resolve()
        if not input_path.is_file():
            raise BatchRequestError(f"items[{index}].input is not a file: {input_path}")
        raw_template = str(raw.get("template_path") or "").strip()
        template_path = Path(raw_template).expanduser().resolve() if raw_template else None
        items.append(BatchItem(item_id=item_id, input_path=input_path, template_path=template_path))
    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise BatchRequestError("options must be an object")
    return BatchRequest(
        batch_id=batch_id,
        engine=engine,
        mode=mode,
        gpu_count=gpu_count,
        samples_per_item=samples_per_item,
        seed=seed,
        items=tuple(items),
        options=dict(options),
    )


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
