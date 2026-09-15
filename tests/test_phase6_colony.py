"""Tests for Phase 6 social marker field (colony signalling).

Covers:
- emit() adds to the right cell, wrapped toroidally.
- step() applies decay and (optional) diffusion.
- sample() reads a windowed mean.
- COLONY_MARKER_OFF=True (default) makes emit/sample no-ops.
- MemoryEcologyWorld wires the field correctly (init, step, emit on eat).
"""
from __future__ import annotations

import numpy as np
import pytest

import evolife.colony as col_mod
import evolife.config as cfg
from evolife.colony import ColonyMarkerField
from evolife.phase3 import MemoryEcologyWorld


@pytest.fixture(autouse=True)
def _restore_marker_defaults():
    saved = (
        cfg.COLONY_MARKER_OFF,
        cfg.COLONY_MARKER_DECAY,
        cfg.COLONY_MARKER_DIFFUSION,
        cfg.COLONY_MARKER_GRID_SCALE,
        cfg.COLONY_MARKER_EMIT,
        col_mod.COLONY_MARKER_OFF,
        col_mod.COLONY_MARKER_DECAY,
        col_mod.COLONY_MARKER_DIFFUSION,
        col_mod.COLONY_MARKER_GRID_SCALE,
    )
    yield
    (
        cfg.COLONY_MARKER_OFF,
        cfg.COLONY_MARKER_DECAY,
        cfg.COLONY_MARKER_DIFFUSION,
        cfg.COLONY_MARKER_GRID_SCALE,
        cfg.COLONY_MARKER_EMIT,
        col_mod.COLONY_MARKER_OFF,
        col_mod.COLONY_MARKER_DECAY,
        col_mod.COLONY_MARKER_DIFFUSION,
        col_mod.COLONY_MARKER_GRID_SCALE,
    ) = saved


def _enable_markers():
    cfg.COLONY_MARKER_OFF = False
    col_mod.COLONY_MARKER_OFF = False


def test_phase6_marker_field_initial_zeros():
    """Fresh field has zero intensity everywhere."""
    _enable_markers()
    f = ColonyMarkerField(width=80.0, height=80.0)
    assert f.grid.shape == (640, 640)  # GRID_SCALE=8 by default
    assert f.grid.sum() == 0.0


def test_phase6_emit_adds_to_cell():
    """emit(x, y) should add to the right cell."""
    _enable_markers()
    f = ColonyMarkerField(width=80.0, height=80.0)
    f.emit(10.0, 20.0, amount=1.0)
    # Cell at (10*8, 20*8) = (80, 160) should have 1.0.
    assert f.grid[160, 80] == pytest.approx(1.0, abs=1e-6)
    # All other cells still zero.
    assert f.grid.sum() == pytest.approx(1.0, abs=1e-6)


def test_phase6_emit_wraps_toroidally():
    """Out-of-bounds positions wrap to opposite side."""
    _enable_markers()
    f = ColonyMarkerField(width=80.0, height=80.0)
    f.emit(85.0, 0.0, amount=1.0)  # 85 % 80 = 5 world units
    # Cell at (5*8, 0) = (40, 0).
    assert f.grid[0, 40] == pytest.approx(1.0, abs=1e-6)


def test_phase6_step_applies_decay():
    """step() should multiply the field by the decay factor."""
    _enable_markers()
    cfg.COLONY_MARKER_DECAY = 0.5
    col_mod.COLONY_MARKER_DECAY = 0.5
    cfg.COLONY_MARKER_DIFFUSION = 0.0
    col_mod.COLONY_MARKER_DIFFUSION = 0.0
    f = ColonyMarkerField(width=80.0, height=80.0)
    f.emit(10.0, 20.0, amount=2.0)
    f.step()
    assert f.grid[160, 80] == pytest.approx(1.0, abs=1e-6)
    assert f.grid.sum() == pytest.approx(1.0, abs=1e-6)


def test_phase6_step_diffuses_to_neighbours():
    """Diffusion should spread intensity to 8 neighbours with the kernel."""
    _enable_markers()
    cfg.COLONY_MARKER_DIFFUSION = 1.0  # fully apply blurred version
    col_mod.COLONY_MARKER_DIFFUSION = 1.0
    cfg.COLONY_MARKER_DECAY = 1.0  # no decay to isolate diffusion
    col_mod.COLONY_MARKER_DECAY = 1.0
    f = ColonyMarkerField(width=80.0, height=80.0)
    # Drop a point mass at (10, 10) world.
    f.emit(10.0, 10.0, amount=1.0)
    # Snapshot the centre value before diffusion.
    centre_before = f.grid[80, 80]
    # Take one diffusion step.
    f.step()
    # Centre should drop (some went to neighbours).
    assert f.grid[80, 80] < centre_before
    # Neighbours should be > 0.
    assert f.grid[79, 80] > 0
    assert f.grid[80, 79] > 0


def test_phase6_sample_reads_window():
    """sample() returns mean intensity in a small window."""
    _enable_markers()
    f = ColonyMarkerField(width=80.0, height=80.0)
    f.emit(10.0, 10.0, amount=4.0)
    # At the centre (10, 10), there should be intensity in the
    # sampled window. Radius 1 = 9 cells. Mean = 4/9.
    val = f.sample(10.0, 10.0, radius=1.0)
    assert val == pytest.approx(4.0 / 9.0, abs=1e-6)


def test_phase6_master_switch_off():
    """COLONY_MARKER_OFF=True makes emit/sample no-ops (Phase 3/4/5 compat)."""
    # COLONY_MARKER_OFF is True by default.
    f = ColonyMarkerField(width=80.0, height=80.0)
    f.emit(10.0, 20.0, amount=5.0)
    assert f.grid.sum() == 0.0  # not stored
    assert f.sample(10.0, 20.0) == 0.0
    f.step()  # also no-op
    assert f.grid.sum() == 0.0


def test_phase6_world_field_initialised():
    """MemoryEcologyWorld creates a marker_field at init."""
    _enable_markers()
    w = MemoryEcologyWorld(seed=1, mode="hidden_season")
    assert hasattr(w, "marker_field")
    assert w.marker_field.grid.shape[0] > 0
    assert w.marker_field.grid.shape[1] > 0


def test_phase6_world_emits_on_positive_eat():
    """A positive eat should deposit a marker at the eat site."""
    _enable_markers()
    w = MemoryEcologyWorld(seed=2, mode="static_dual")
    org = w.organisms[0]
    # Place org on top of food_a.
    org.x, org.y = w.food_a[0].x, w.food_a[0].y
    # Snapshot of field before eat.
    before = w.marker_field.grid.copy()
    w._resolve_eat_phase3()
    after = w.marker_field.grid
    # Some cell should have a new marker (since the eat happened).
    assert (after > before).any()
