import numpy as np
import pytest

from evolife.behavior import (
    descriptors,
    mean_descriptors,
    note_act,
    state_dependence,
    steering_alignment,
)
from evolife.organism import Organism
from evolife.world import World


def _org() -> Organism:
    return Organism(id=0, x=8.0, y=8.0, heading=0.0, energy=10.0)


def test_steering_alignment_is_pearson_of_smell_asymmetry_and_turn():
    org = _org()
    for i in range(20):
        right = i * 0.1
        turn = 0.5 * right
        note_act(org, [0.0, 0.0, right], turn=turn, speed=1.0, width=64, height=64)
    assert steering_alignment(org) == pytest.approx(1.0, abs=1e-6)


def test_state_dependence_is_zero_for_a_fixed_action():
    org = _org()
    for left in (0.0, 0.5, 1.0):
        for front in (0.0, 0.5, 1.0):
            for right in (0.0, 0.5, 1.0):
                note_act(
                    org,
                    [left, front, right],
                    turn=0.0,
                    speed=0.0,
                    width=64,
                    height=64,
                )
    assert state_dependence(org) == pytest.approx(0.0)


def test_rest_move_transition_updates_descriptors():
    w = World(seed=1)
    org = w.organisms[0]
    org.brain = _FixedBrain(turn=0.0, locomotion=0.0)
    w._act(org)
    org.brain = _FixedBrain(turn=0.0, locomotion=1.0)
    w._act(org)
    d = descriptors(org)
    assert d["transition_rate"] > 0.0
    assert d["moving_fraction"] > 0.0
    assert d["mean_speed"] > 0.0
    assert -1.0 <= d["steering_alignment"] <= 1.0
    assert d["state_dependence"] >= 0.0


def test_mean_descriptors_averages_living_members():
    a = _org()
    a.age = 10
    a.ticks_moving = 8
    b = Organism(id=1, x=8.0, y=8.0, heading=0.0, energy=10.0)
    b.age = 10
    b.ticks_moving = 2
    b.alive = False
    means = mean_descriptors([a, b])
    assert means is not None
    assert means["n"] == 1.0
    assert means["moving_fraction"] == 0.8


class _FixedBrain:
    def __init__(self, turn: float, locomotion: float) -> None:
        self._out = np.array([turn, locomotion], dtype=np.float32)

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        return self._out
