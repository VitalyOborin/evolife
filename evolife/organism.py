"""Organism — a single living thing in the world.

Pure data plus cheap running stats. The world drives the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .brain import Brain
from .genome import Genome


@dataclass
class Organism:
    """A living digital organism."""

    id: int
    x: float
    y: float
    heading: float  # radians, 0 == +x, pi/2 == +y (math convention).
    energy: float
    age: int = 0
    alive: bool = True
    brain: Brain | None = None
    genome: Genome | None = None
    parent_id: int | None = None
    children: int = 0
    peak_energy: float = 0.0
    generation: int = 0
    founder_lineage_id: int = 0
    food_eaten: int = 0
    species_id: int = 0
    movement_transitions: int = 0
    ticks_moving: int = 0
    speed_sum: float = 0.0
    longest_rest: int = 0
    longest_move: int = 0
    distance_sum: float = 0.0
    abs_turn_sum: float = 0.0
    time_to_first_food: int | None = None
    explored_bins: set[int] = field(default_factory=set)
    steer_n: int = 0
    steer_sum_x: float = 0.0
    steer_sum_y: float = 0.0
    steer_sum_xy: float = 0.0
    steer_sum_x2: float = 0.0
    steer_sum_y2: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * 108)
    _moving: bool | None = None
    _bout_len: int = 0

    def note_locomotion(self, speed: float) -> None:
        """Record one tick of rest or movement for behavioral metrics."""
        moving = speed > 0.0
        self.speed_sum += speed
        if moving:
            self.ticks_moving += 1
        if self._moving is not None and moving != self._moving:
            self.movement_transitions += 1
            self._bout_len = 1
        else:
            self._bout_len += 1
        if moving:
            self.longest_move = max(self.longest_move, self._bout_len)
        else:
            self.longest_rest = max(self.longest_rest, self._bout_len)
        self._moving = moving
