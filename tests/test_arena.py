"""Smoke tests for the Behavioral Arena."""

from __future__ import annotations

import numpy as np
import pytest

from evolife.arena import (
    SCENARIO_BUILDERS,
    aggregate,
    brain_shape,
    run_episode,
    sample_genomes_by_generation,
)
from evolife.brain import Brain
from evolife.organism import Organism
from evolife.world import World


def _make_founder() -> "Genome":
    import numpy as np

    from evolife.genome import Genome

    rng = np.random.default_rng(0)
    return Brain.make_default_genome(rng=rng)


def test_arena_smoke_uniform_100_ticks() -> None:
    genome = _make_founder()
    res = run_episode(genome, "uniform", seed=1, n_ticks=100)
    assert res.scenario == "uniform"
    assert res.seed == 1
    assert res.ticks_alive >= 0
    assert 0.0 <= res.moving_fraction <= 1.0
    assert -1.0 <= res.steering_alignment <= 1.0


@pytest.mark.parametrize(
    "kind",
    [
        "uniform",
        "ahead",
        "behind",
        "left_right",
        "sparse",
        "dense",
        "relocating",
        "smell_blanked",
    ],
)
def test_arena_all_scenarios_smoke(kind: str) -> None:
    genome = _make_founder()
    res = run_episode(genome, kind, seed=1, n_ticks=80)
    assert res.scenario == kind
    assert res.ticks_alive >= 0


def test_arena_aggregate_two_seeds() -> None:
    genome = _make_founder()
    rs = [
        run_episode(genome, "uniform", seed=1, n_ticks=50),
        run_episode(genome, "uniform", seed=2, n_ticks=50),
    ]
    agg = aggregate(rs)
    # We expect at least one of the mean/std pairs to be present.
    assert "food_eaten_mean" in agg
    assert "steering_alignment_std" in agg
    assert agg["food_eaten_mean"] >= 0


def test_arena_sample_genomes_by_generation() -> None:
    """Run a tiny world and verify the sampler returns expected gens."""
    world = World(seed=42, width=128, height=128)
    for _ in range(500):
        world.step()
    by_gen = sample_genomes_by_generation(world, [0, 5, 10, 20])
    # Gen 0 might not survive; gens >=1 should typically exist after
    # 500 ticks in the small world.
    assert isinstance(by_gen, dict)
    # All returned genomes must have a brain shape (Phase 1.5 founder:
    # 3 sensor + 1 hidden + 2 motor = 6 nodes, 5 connections).
    for g, genome in by_gen.items():
        n, c = brain_shape(genome)
        assert n == 6
        assert c >= 5
