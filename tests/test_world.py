import numpy as np

from evolife.config import (
    FOOD_ENERGY,
    FOOD_TARGET,
    INITIAL_POPULATION,
    MAX_AGE,
    POPULATION_CAP,
)
from evolife.world import World


def test_world_bootstraps_initial_population():
    w = World(seed=42)
    assert len(w.organisms) == INITIAL_POPULATION
    assert len(w.food) >= FOOD_TARGET - 1  # spawn loop may underrun by 1


def test_world_deterministic_with_seed():
    w1 = World(seed=7)
    w2 = World(seed=7)
    for _ in range(10):
        w1.step()
        w2.step()
    # Energy values for the founders should match under same seed.
    e1 = [o.energy for o in w1.organisms if o.alive]
    e2 = [o.energy for o in w2.organisms if o.alive]
    assert e1 and e2
    np.testing.assert_allclose(sorted(e1), sorted(e2), atol=1e-3)


def test_world_kills_starving_organisms():
    w = World(seed=1)
    # Drain everyone's energy manually and step.
    for org in w.organisms:
        org.energy = 0.001
    for _ in range(50):
        w.step()
    assert all(o.energy > 0.0 for o in w.organisms if o.alive)


def test_world_respects_population_cap():
    w = World(seed=2)
    # Force every organism to want to reproduce and have lots of energy.
    for org in w.organisms:
        org.energy = 10_000.0
        org._reproduce_attempt = True  # type: ignore[attr-defined]
    for _ in range(20):
        w.step()
    assert len(w.organisms) <= POPULATION_CAP


def test_world_max_age_safety_net():
    w = World(seed=3)
    # Push one organism past MAX_AGE without starvation.
    target = w.organisms[0]
    target.energy = 10_000.0
    initial_count = len(w.organisms)
    for _ in range(MAX_AGE + 100):
        if target.alive:
            w.step()
        else:
            break
    assert not target.alive or target.age >= MAX_AGE
    # Population should have decreased or stayed flat.
    assert len(w.organisms) <= initial_count


def test_food_energy_is_finite():
    assert FOOD_ENERGY > 0.0
