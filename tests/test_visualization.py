from evolife.organism import Organism
from evolife.speciation import Species
from evolife.visualization import (
    NEWBORN_COLOR,
    inspector_lines,
    organism_at,
    species_draw_color,
)


def _species(**kwargs) -> Species:
    from evolife.brain import Brain
    import numpy as np

    defaults = dict(
        id=3,
        representative=Brain.make_default_genome(rng=np.random.default_rng(0)),
        born_tick=4812,
        last_seen_tick=11152,
        member_count=47,
        peak_population=81,
        parent_species_id=0,
        established=True,
    )
    defaults.update(kwargs)
    return Species(**defaults)


def test_newborns_share_one_color_and_established_do_not():
    assert species_draw_color(0, False) == NEWBORN_COLOR
    assert species_draw_color(7, False) == NEWBORN_COLOR
    assert species_draw_color(0, True) != NEWBORN_COLOR
    assert species_draw_color(0, True) != species_draw_color(1, True)


def test_inspector_compares_behavior_against_parent():
    own = {
        "genome_nodes": 6.2,
        "genome_conns": 8.4,
        "moving_fraction": 0.38,
        "food_rate": 5.7,
        "transition_rate": 8.9,
        "exploration_rate": 7.5,
        "steering_alignment": 0.68,
        "state_dependence": 0.19,
    }
    other = {
        "genome_nodes": 5.0,
        "genome_conns": 6.0,
        "moving_fraction": 0.72,
        "food_rate": 4.1,
        "transition_rate": 1.2,
        "exploration_rate": 13.2,
        "steering_alignment": 0.31,
        "state_dependence": 0.02,
    }
    text = "\n".join(
        inspector_lines(_species(), tick=11_152, own=own, other_id=0, other=other)
    )
    assert "Species 3  established" in text
    assert "origin: tick 4,812" in text
    assert "parent: species 0" in text
    assert "age: 6,340 ticks" in text
    assert "72%" in text
    assert "38%" in text
    assert "0.31" in text
    assert "0.68" in text


def test_inspector_dashes_when_parent_has_no_living_members():
    own = {"genome_nodes": 6.0, "genome_conns": 8.0, "moving_fraction": 0.5}
    text = "\n".join(
        inspector_lines(_species(), tick=11_152, own=own, other_id=0, other=None)
    )
    assert "behavior vs species 0" in text
    assert "—" in text


def test_organism_at_picks_nearest_within_radius():
    a = Organism(id=1, x=10.0, y=10.0, heading=0.0, energy=1.0, species_id=0)
    b = Organism(id=2, x=30.0, y=10.0, heading=0.0, energy=1.0, species_id=1)
    assert organism_at([a, b], 12.0, 10.0, 12.0) is a
    assert organism_at([a, b], 200.0, 200.0, 12.0) is None
