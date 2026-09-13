"""Organism — a single living thing in the world.

Pure data. The world drives the simulation; the organism does not know
about food or other organisms except through sensor readings produced by
World._sensors_for.
"""

from __future__ import annotations

from dataclasses import dataclass

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
