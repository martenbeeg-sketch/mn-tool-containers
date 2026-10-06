#!/usr/bin/env python3
"""Confirm the pinned JAX runtime actually executes on an NVIDIA GPU."""

from __future__ import annotations

import os
import subprocess

import jax
import jax.numpy as jnp


devices = [device for device in jax.devices() if device.platform == "gpu"]
if not devices:
    raise SystemExit(f"No JAX GPU backend detected. Devices: {jax.devices()}")

a = jnp.arange(512 * 512, dtype=jnp.float32).reshape((512, 512)) / 512
b = jnp.eye(512, dtype=jnp.float32)
checksum = jnp.sum(a @ b).block_until_ready()
print(f"JAX {jax.__version__}; GPU devices: {devices}; matrix checksum: {float(checksum):.3f}")

if os.getenv("MN_REQUIRE_BLACKWELL", "0") == "1":
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Could not inspect GPU capability: {exc}") from exc
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not any(row.split(",", 1)[0].strip() in {"12.0", "12.1"} for row in rows):
        raise SystemExit(f"Expected Blackwell compute capability 12.0+, found: {rows}")
    print("Blackwell detected:", rows)
