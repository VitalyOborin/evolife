import copy

import numpy as np
import pytest

from evolife.brain import Brain
from evolife.config import (
    COMPAT_C_EXCESS,
    ESTABLISHED_MIN_AGE,
    ESTABLISHED_MIN_PEAK,
    REPRODUCTION_THRESHOLD,
    SPECIES_THRESHOLD,
)
from evolife.genome import ConnectionGene
from evolife.speciation import SpeciesManager, compatibility_distance
from evolife.world import World


def test_identical_genomes_have_zero_distance():
    g1 = Brain.make_default_genome(rng=np.random.default_rng(0))
    g2 = Brain.make_default_genome(rng=np.random.default_rng(0))
    assert compatibility_distance(g1, g2) == pytest.approx(0.0)


def test_one_excess_gene_is_not_divided_by_genome_size():
    g1 = Brain.make_default_genome(rng=np.random.default_rng(0))
    g2 = copy.deepcopy(g1)
    g2.connections[100] = ConnectionGene(
        innovation=100, in_node=0, out_node=3, weight=0.0, enabled=True
    )
    d = compatibility_distance(g1, g2)
    assert d == pytest.approx(COMPAT_C_EXCESS)
    assert d != pytest.approx(COMPAT_C_EXCESS / len(g1.connections))


def test_manager_puts_similar_genomes_in_one_species():
    g1 = Brain.make_default_genome(rng=np.random.default_rng(0))
    g2 = Brain.make_default_genome(rng=np.random.default_rng(0))
    mgr = SpeciesManager()
    a = mgr.assign(g1, tick=0, organism_id=0)
    b = mgr.assign(g2, tick=0, organism_id=1)
    assert a == b
    assert len(mgr.living()) == 1
    assert mgr.events[0].kind == "ORIGIN"


def test_manager_originates_when_distance_exceeds_threshold():
    g1 = Brain.make_default_genome(rng=np.random.default_rng(0))
    g2 = copy.deepcopy(g1)
    for innov in range(100, 104):
        g2.connections[innov] = ConnectionGene(
            innovation=innov, in_node=0, out_node=3, weight=0.0, enabled=True
        )
    assert compatibility_distance(g1, g2) > SPECIES_THRESHOLD
    mgr = SpeciesManager()
    parent = mgr.assign(g1, tick=0, organism_id=0)
    child = mgr.assign(g2, tick=1, organism_id=1, parent_species_id=parent)
    assert child != parent
    assert mgr.species[child].parent_species_id == parent


def test_founders_share_one_species():
    w = World(seed=1)
    ids = {o.species_id for o in w.organisms}
    assert len(ids) == 1
    assert w.n_species() == 1
    living = w.species_manager.living()
    assert living[0].member_count == len(w.organisms)
    assert living[0].established is False
    assert w.n_established_species() == 0


def test_child_stays_in_parent_species_when_close():
    w = World(seed=4)
    parent = w.organisms[0]
    parent.energy = REPRODUCTION_THRESHOLD
    w._reproduce()
    children = [o for o in w.organisms if o.parent_id == parent.id]
    assert len(children) == 1
    assert children[0].species_id == parent.species_id


def test_species_goes_extinct_when_last_member_dies():
    w = World(seed=1)
    sid = w.organisms[0].species_id
    w.food.clear()
    for org in w.organisms:
        org.energy = 0.001
    for _ in range(50):
        w.step()
        if w.population() == 0:
            break
    assert w.species_manager.species[sid].extinct
    assert any(e.kind == "EXTINCTION" and e.species_id == sid for e in w.species_manager.events)


class _Alive:
    def __init__(self, species_id: int) -> None:
        self.species_id = species_id
        self.alive = True


def test_species_establishes_only_after_age_and_peak():
    g = Brain.make_default_genome(rng=np.random.default_rng(0))
    mgr = SpeciesManager()
    sid = mgr.assign(g, tick=0, organism_id=0)
    orgs = [_Alive(sid) for _ in range(ESTABLISHED_MIN_PEAK)]
    mgr.sync(orgs, tick=ESTABLISHED_MIN_AGE)
    assert mgr.species[sid].established is False
    mgr.sync(orgs, tick=ESTABLISHED_MIN_AGE + 1)
    assert mgr.species[sid].established is True
    assert sum(1 for e in mgr.events if e.kind == "ESTABLISHED") == 1
    mgr.sync(orgs, tick=ESTABLISHED_MIN_AGE + 50)
    assert sum(1 for e in mgr.events if e.kind == "ESTABLISHED") == 1


def test_species_with_small_peak_does_not_establish():
    g = Brain.make_default_genome(rng=np.random.default_rng(0))
    mgr = SpeciesManager()
    sid = mgr.assign(g, tick=0, organism_id=0)
    orgs = [_Alive(sid) for _ in range(ESTABLISHED_MIN_PEAK - 1)]
    mgr.sync(orgs, tick=ESTABLISHED_MIN_AGE + 5_000)
    assert mgr.species[sid].established is False
    assert all(e.kind != "ESTABLISHED" for e in mgr.events)
