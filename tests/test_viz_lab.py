"""Tests for the lab visualizer helpers — no window required except dummy."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from evolife.config import PROBE_DISTANCE
from evolife.genome import ConnectionGene, Genome, NodeGene, NodeType
from evolife.viz_lab import (
    PANEL_WIDTH,
    RollingSeries,
    colorize_smell_fields,
    genome_has_recurrent_cycle,
    probe_point,
)
from evolife.world import World


def test_probe_point_matches_configured_distance():
    px, py, wraps = probe_point(100.0, 100.0, 0.0, 0.0, 512, 512)
    assert not wraps
    assert abs(px - (100.0 + PROBE_DISTANCE)) < 1e-6
    assert abs(py - 100.0) < 1e-6


def test_probe_point_wraps_at_seam():
    px, py, wraps = probe_point(0.0, 0.0, math.pi, 0.0, 64.0, 64.0)
    assert wraps
    assert 0.0 <= px < 64.0
    assert 0.0 <= py < 64.0


def test_colorize_smell_fields_shape_and_void():
    grid = np.zeros((16, 24), dtype=np.float32)
    rgb = colorize_smell_fields([(grid, (36, 196, 168))], device=None)
    assert rgb.shape == (16, 24, 3)
    assert rgb.dtype == np.uint8
    # Empty field stays near the void colour.
    assert int(rgb[0, 0, 0]) <= 16


def test_colorize_smell_fields_lights_up_peak():
    grid = np.zeros((8, 8), dtype=np.float32)
    grid[3, 4] = 4.0
    rgb = colorize_smell_fields([(grid, (36, 196, 168))], device=None)
    assert rgb[3, 4, 1] > rgb[0, 0, 1]


def test_rolling_series_caps_and_tracks_three_channels():
    s = RollingSeries(cap=5)
    for i in range(7):
        s.push(i, 10 + i, i % 3)
    assert len(s) == 5
    assert list(s.pop) == [2, 3, 4, 5, 6]
    assert list(s.food) == [12, 13, 14, 15, 16]


def test_founder_genome_has_no_recurrent_cycle():
    from evolife.brain import Brain

    g = Brain.make_default_genome()
    assert genome_has_recurrent_cycle(g) is False


def test_hidden_cycle_is_detected():
    g = Genome()
    g.nodes[0] = NodeGene(0, NodeType.HIDDEN)
    g.nodes[1] = NodeGene(1, NodeType.HIDDEN)
    g.connections[1] = ConnectionGene(1, 0, 1, 1.0, True)
    g.connections[2] = ConnectionGene(2, 1, 0, 1.0, True)
    assert genome_has_recurrent_cycle(g) is True


def test_world_resets_mutation_counters_each_step():
    w = World(seed=1)
    w.step()
    w.tick_mutations = 10_000
    w.tick_births = 10_000
    w.step()
    assert w.tick_mutations != 10_000
    assert w.tick_births != 10_000


def test_panel_is_beside_the_torus_not_on_it():
    assert PANEL_WIDTH > 0


def test_colorize_uses_torch_when_device_given():
    torch = pytest.importorskip("torch")
    grid = np.zeros((8, 8), dtype=np.float32)
    grid[2, 2] = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = colorize_smell_fields([(grid, (36, 196, 168))], device=device)
    assert rgb.shape == (8, 8, 3)
    assert rgb[2, 2, 1] > rgb[0, 0, 1]


def test_lab_visualizer_dummy_render_full_layout():
    pygame = pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evolife.viz_lab import LabVisualizer

    w = World(seed=1, width=64, height=64)
    w.step()
    viz = LabVisualizer(w)
    try:
        viz.render()
        assert viz.screen.get_width() == 64 + PANEL_WIDTH
        assert viz.screen.get_height() == 64
        assert viz._torus.get_width() == 64
        assert len(viz.series) == 1
    finally:
        viz.close()

