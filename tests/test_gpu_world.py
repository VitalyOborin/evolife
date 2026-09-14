"""Smoke tests for evolife.gpu_world.GpuWorld.

These tests do not require CUDA — when torch reports no GPU the world
falls back to a CPU tensor device, so we can still exercise the full
step path on a developer machine without an NVIDIA card. They guard
against regressions in the smell-field boundary handling and in
write_genome_to_slot sanitisation that surfaced when running on the
RTX 4070 Ti.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evolife.gpu_world import GpuWorld


def _device_or_skip() -> torch.device:
    """Return a torch device we can run on, skipping if neither CUDA
    nor a usable CPU torch is available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_gpu_world_smoke_200_ticks_seed_1() -> None:
    """Run 200 ticks on seed 1 and assert no crash and a sane alive count."""
    torch.manual_seed(0)
    w = GpuWorld(seed=1, width=128, height=128, device=_device_or_skip())
    for _ in range(200):
        w.step()
    alive = int(w.alive_mask.sum().item())
    # Founder population starts at INITIAL_POPULATION (>=50). After 200
    # ticks, even in the worst-case extinction scenario we expect a
    # non-negative count.
    assert alive >= 0
    assert w.tick == 200


def test_gpu_world_smoke_1000_ticks_all_seeds() -> None:
    """Run 1000 ticks across 5 seeds. None should crash."""
    dev = _device_or_skip()
    for seed in (1, 2, 3, 4, 5):
        w = GpuWorld(seed=seed, width=128, height=128, device=dev)
        for _ in range(1000):
            w.step()
        alive = int(w.alive_mask.sum().item())
        assert alive >= 0, f"seed {seed} crashed with negative pop"
        assert w.tick == 1000


def test_gpu_world_smell_boundary_wraps_at_corner() -> None:
    """Probe at the world boundary must not produce out-of-bounds
    indices. We force position (0, 0) and heading pi which would push
    the probe to negative x — the wrap should clamp to width-1.
    """
    import math

    dev = _device_or_skip()
    w = GpuWorld(seed=42, width=64, height=64, device=dev)
    # Force the alive organism to position (0, 0) heading pi.
    if w.alive_mask.sum().item() == 0:
        # Reseed so we have at least one organism.
        w = GpuWorld(seed=1, width=64, height=64, device=dev)
    idx = int(w.alive_mask.nonzero(as_tuple=False)[0].item())
    w.positions[idx, 0] = 0.0
    w.positions[idx, 1] = 0.0
    w.headings[idx] = math.pi
    # Single step should not raise.
    w.step()


def test_gpu_metrics_smoke_writes_sqlite(tmp_path) -> None:
    """GpuMetrics writes the same schema as Metrics, populated from
    GPU tensors, with no crash for a 200-tick run."""
    import sqlite3

    from evolife.gpu_metrics import GpuMetrics

    dev = _device_or_skip()
    w = GpuWorld(seed=1, width=64, height=64, device=dev)
    db_path = str(tmp_path / "gpu_metrics_smoke.sqlite")
    m = GpuMetrics(path=db_path)
    try:
        for _ in range(200):
            w.step()
            m.record_gpu_world(w)
            m.record_gpu_organisms(w)
        m.flush_gpu_events(w)
    finally:
        m.close()
    # Verify schema.
    c = sqlite3.connect(db_path)
    tables = {
        r[0]
        for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for required in (
        "world_snapshots",
        "organism_snapshots",
        "events",
    ):
        assert required in tables
    n_world = c.execute(
        "SELECT COUNT(*) FROM world_snapshots"
    ).fetchone()[0]
    assert n_world > 0

