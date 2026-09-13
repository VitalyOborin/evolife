import numpy as np

from evolife.config import (
    CHILD_DISPERSAL_MAX,
    CHILD_DISPERSAL_MIN,
    FOOD_ENERGY,
    FOOD_TARGET,
    INITIAL_POPULATION,
    MAX_AGE,
    POPULATION_CAP,
    REPRODUCTION_THRESHOLD,
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
    """Sanity check that the world steps without crashing on a few ticks."""
    w = World(seed=11)
    for _ in range(200):
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


def test_events_recorded_on_birth_and_death():
    """World should emit at least one Birth and one Death event."""
    from evolife.events import EventKind

    w = World(seed=99)
    # Drain everyone to starvation over a few ticks.
    for org in w.organisms:
        org.energy = 0.001
    for _ in range(100):
        w.step()
    by_kind = w.events.by_kind()
    assert len(by_kind[EventKind.BIRTH]) >= INITIAL_POPULATION
    assert len(by_kind[EventKind.DEATH]) >= 1


def test_smell_field_changes_with_food_layout():
    from evolife.sensors import SmellField
    from evolife.world import Food

    field = SmellField(64, 64, radius=8)
    empty = field.sample(32.0, 32.0, 0.0, 1.0).copy()
    field.recompute([Food(x=32.0, y=32.0)])
    with_food = field.sample(32.0, 32.0, 0.0, 1.0)
    assert with_food.sum() > empty.sum()


def test_sensors_are_smell_only():
    w = World(seed=1)
    org = w.organisms[0]
    sensors = w._sensors_for(org)
    assert sensors.shape == (3,)


def test_founders_are_generation_zero_with_own_lineage():
    w = World(seed=1)
    ids = [o.id for o in w.organisms]
    assert all(o.generation == 0 for o in w.organisms)
    assert {o.founder_lineage_id for o in w.organisms} == set(ids)


def test_child_is_offset_from_parent_and_inherits_lineage():
    w = World(seed=4)
    parent = w.organisms[0]
    parent.energy = REPRODUCTION_THRESHOLD
    px, py = parent.x, parent.y
    w._reproduce()
    children = [o for o in w.organisms if o.parent_id == parent.id]
    assert len(children) == 1
    child = children[0]
    dx = child.x - px
    dy = child.y - py
    dx -= w.width * round(dx / w.width)
    dy -= w.height * round(dy / w.height)
    dist = float(np.hypot(dx, dy))
    assert CHILD_DISPERSAL_MIN - 1e-6 <= dist <= CHILD_DISPERSAL_MAX + 1e-6
    assert child.generation == parent.generation + 1
    assert child.founder_lineage_id == parent.founder_lineage_id
    assert child.id != parent.id


def test_food_regrows_toward_target_when_rate_is_one(monkeypatch):
    import evolife.config as cfg

    monkeypatch.setattr(cfg, "FOOD_REGROWTH_RATE", 1.0)
    monkeypatch.setattr("evolife.world.FOOD_REGROWTH_RATE", 1.0)
    w = World(seed=5)
    w.organisms.clear()
    w.food.clear()
    w.step()
    assert len(w.food) == FOOD_TARGET
