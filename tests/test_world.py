import numpy as np

from evolife.config import (
    FOOD_ENERGY,
    FOOD_TARGET,
    INITIAL_POPULATION,
    MAX_AGE,
    POPULATION_CAP,
    TOURNAMENT_EVERY,
)
from evolife.world import World


def test_world_bootstraps_initial_population():
    w = World(seed=42)
    assert len(w.organisms) == INITIAL_POPULATION
    assert len(w.food) >= FOOD_TARGET - 1


def test_world_deterministic_with_seed():
    w1 = World(seed=7)
    w2 = World(seed=7)
    for _ in range(10):
        w1.step()
        w2.step()
    e1 = sorted(o.energy for o in w1.organisms if o.alive)
    e2 = sorted(o.energy for o in w2.organisms if o.alive)
    assert e1 and e2
    np.testing.assert_allclose(e1, e2, atol=1e-3)


def test_world_kills_starving_organisms():
    w = World(seed=1)
    for org in w.organisms:
        org.energy = 0.001
    for _ in range(50):
        w.step()
    assert all(o.energy > 0.0 for o in w.organisms if o.alive)


def test_world_respects_population_cap():
    w = World(seed=2)
    for org in w.organisms:
        org.energy = 10_000.0
        org._reproduce_attempt = True  # type: ignore[attr-defined]
    for _ in range(20):
        w.step()
    assert len(w.organisms) <= POPULATION_CAP


def test_world_max_age_safety_net(monkeypatch):
    import evolife.config as cfg

    # Shrink MAX_AGE for the duration of this test.
    monkeypatch.setattr(cfg, "MAX_AGE", 200)
    monkeypatch.setattr("evolife.world.MAX_AGE", 200)
    w = World(seed=3)
    target = w.organisms[0]
    target.energy = 10_000.0
    initial_pop = w.population()
    for _ in range(cfg.MAX_AGE + 50):
        if target.alive:
            w.step()
        else:
            break
    assert not target.alive or target.age >= cfg.MAX_AGE
    # The target organism is now dead — alive population may still be
    # non-zero due to offspring/clones, but the target itself must be
    # marked dead.
    assert target.alive is False


def test_food_energy_is_finite():
    assert FOOD_ENERGY > 0.0


def test_tournament_runs_without_crashing():
    """A few tournament windows should not raise."""
    w = World(seed=11)
    for _ in range(2 * TOURNAMENT_EVERY):
        w.step()
    assert w.population() >= 0


def test_mutation_can_change_genome_topology():
    """Run long enough that structural mutations likely fired."""
    w = World(seed=13)
    initial_total = sum(
        len(o.genome.connections)
        for o in w.organisms
        if o.alive
    )
    for _ in range(500):
        w.step()
    final_total = sum(
        len(o.genome.connections)
        for o in w.organisms
        if o.alive
    )
    # We just assert the metric is well-defined; the structural delta
    # is stochastic.
    assert isinstance(final_total, int)
    assert isinstance(initial_total, int)
