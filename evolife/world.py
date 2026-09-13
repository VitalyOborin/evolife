"""World — the simulation driver.

Owns the organism list, the food list, the RNG, and the tick counter.
Each tick: spawn food, ask each organism for actions, integrate motion,
resolve eat attempts, deduct energy, kill the starving, attempt
reproduction.

Mutation policy: in v0 reproduction mutates weights only and uses an
independent child mutation probability. In v1 it will pull from config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import Brain
from .config import (
    COLLISION_RADIUS,
    EAT_RADIUS,
    FOOD_ENERGY,
    FOOD_SPAWN_RATE,
    FOOD_TARGET,
    IDLE_ENERGY_COST,
    INITIAL_ENERGY,
    INITIAL_POPULATION,
    MAX_AGE,
    MAX_LINEAR_SPEED,
    MAX_TURN_RATE,
    MOVE_ENERGY_COST,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    TURN_ENERGY_COST,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from .genome import Genome
from .mutation import mutate_weights
from .organism import Organism


@dataclass
class Food:
    x: float
    y: float


class World:
    """The simulation world."""

    def __init__(
        self,
        seed: int | None = None,
        width: int = WORLD_WIDTH,
        height: int = WORLD_HEIGHT,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.tick: int = 0
        self.organisms: list[Organism] = []
        self.food: list[Food] = []
        self._next_id: int = 0

        # Bootstrap founders.
        for _ in range(INITIAL_POPULATION):
            self._spawn_founder()

        # Seed initial food.
        while len(self.food) < FOOD_TARGET:
            self._spawn_food()

    # --- spawning ----------------------------------------------------------

    def _next_organism_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _spawn_food(self) -> None:
        self.food.append(
            Food(
                x=float(self.rng.uniform(0, self.width)),
                y=float(self.rng.uniform(0, self.height)),
            )
        )

    def _spawn_founder(self) -> None:
        genome = Brain.make_default_genome()
        brain = Brain(genome)
        self.organisms.append(
            Organism(
                id=self._next_organism_id(),
                x=float(self.rng.uniform(0, self.width)),
                y=float(self.rng.uniform(0, self.height)),
                heading=float(self.rng.uniform(0, 2 * np.pi)),
                energy=INITIAL_ENERGY,
                brain=brain,
                genome=genome,
            )
        )

    # --- main loop ---------------------------------------------------------

    def step(self) -> None:
        """Advance one tick."""
        self.tick += 1

        # 1. Spawn food up to target.
        while len(self.food) < FOOD_TARGET and self.rng.random() < FOOD_SPAWN_RATE:
            self._spawn_food()

        # 2. Each organism acts.
        for org in self.organisms:
            if not org.alive:
                continue
            self._act(org)

        # 3. Resolve eat attempts.
        self._resolve_eat()

        # 4. Passive energy drain + age tick + death.
        survivors: list[Organism] = []
        for org in self.organisms:
            if not org.alive:
                continue
            org.energy -= IDLE_ENERGY_COST
            org.age += 1
            if org.energy <= 0.0 or org.age >= MAX_AGE:
                org.alive = False
                continue
            org.peak_energy = max(org.peak_energy, org.energy)
            survivors.append(org)
        self.organisms = survivors

        # 5. Reproduction.
        self._reproduce()

    # --- per-organism action ----------------------------------------------

    def _act(self, org: Organism) -> None:
        """Run the organism brain for one tick and integrate motion."""
        sensors = self._sensors_for(org)
        assert org.brain is not None
        motors = org.brain.forward(sensors)

        # Motor 0: turn rate in [-1, 1] -> radians per tick.
        turn = float(motors[0]) * MAX_TURN_RATE
        # Motor 1: move speed in [0, 1] -> cells per tick.
        move = float(np.clip(motors[1], 0.0, 1.0)) * MAX_LINEAR_SPEED

        # Energy costs.
        org.energy -= abs(turn) * TURN_ENERGY_COST
        org.energy -= move * MOVE_ENERGY_COST

        # Integrate.
        org.heading = (org.heading + turn) % (2 * np.pi)
        org.x = (org.x + np.cos(org.heading) * move) % self.width
        org.y = (org.y + np.sin(org.heading) * move) % self.height

        # Motors 2 and 3 are eat / reproduce attempts. Stash on the
        # organism so _resolve_eat and _reproduce can read them.
        org._eat_attempt = bool(float(motors[2]) > 0.5)  # type: ignore[attr-defined]
        org._reproduce_attempt = bool(float(motors[3]) > 0.5)  # type: ignore[attr-defined]

    def _sensors_for(self, org: Organism) -> np.ndarray:
        """Compute the v0 sensor vector."""
        if not self.food:
            food_angle = 0.0
            food_dist = 1.0
        else:
            fx = np.array([f.x for f in self.food])
            fy = np.array([f.y for f in self.food])
            # Toroidal distance: smallest of dx, width-dx, etc.
            dx = fx - org.x
            dy = fy - org.y
            dx -= self.width * np.round(dx / self.width)
            dy -= self.height * np.round(dy / self.height)
            dist = np.hypot(dx, dy)
            idx = int(np.argmin(dist))
            food_dist = float(dist[idx]) / float(max(self.width, self.height))
            rel_heading = np.arctan2(dy[idx], dx[idx]) - org.heading
            rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
            food_angle = float(np.sin(rel_heading))

        energy_norm = float(np.clip(org.energy / REPRODUCTION_THRESHOLD, 0.0, 1.0))
        speed = 0.0  # v0: no recurrent speed state yet
        bias = 1.0
        return np.array(
            [food_angle, food_dist, energy_norm, speed, bias],
            dtype=np.float32,
        )

    # --- resolution --------------------------------------------------------

    def _resolve_eat(self) -> None:
        """For each organism that wants to eat, consume the nearest food."""
        if not self.food:
            return
        for org in self.organisms:
            if not org.alive or not getattr(org, "_eat_attempt", False):
                continue
            fx = np.array([f.x for f in self.food])
            fy = np.array([f.y for f in self.food])
            dx = fx - org.x
            dy = fy - org.y
            dx -= self.width * np.round(dx / self.width)
            dy -= self.height * np.round(dy / self.height)
            dist = np.hypot(dx, dy)
            idx = int(np.argmin(dist))
            if float(dist[idx]) <= EAT_RADIUS:
                # Eat: remove the food particle, gain energy.
                self.food.pop(idx)
                org.energy += FOOD_ENERGY

    def _reproduce(self) -> None:
        """Spawn children for any organism that wants to and can."""
        if len(self.organisms) >= POPULATION_CAP:
            return
        new_organisms: list[Organism] = []
        for org in self.organisms:
            if not org.alive:
                continue
            if not getattr(org, "_reproduce_attempt", False):
                continue
            if org.energy < REPRODUCTION_THRESHOLD:
                continue
            if len(self.organisms) + len(new_organisms) >= POPULATION_CAP:
                break

            # Mutate weights for the child.
            assert org.genome is not None
            child_genome = mutate_weights(org.genome, self.rng)

            # Split energy.
            org.energy -= REPRODUCTION_ENERGY
            child = Organism(
                id=self._next_organism_id(),
                x=float(org.x),
                y=float(org.y),
                heading=float(org.heading),
                energy=REPRODUCTION_ENERGY,
                brain=Brain(child_genome),
                genome=child_genome,
                parent_id=org.id,
            )
            new_organisms.append(child)
            org.children += 1
        self.organisms.extend(new_organisms)

    # --- diagnostics -------------------------------------------------------

    def population(self) -> int:
        return sum(1 for o in self.organisms if o.alive)

    def mean_energy(self) -> float:
        alive = [o for o in self.organisms if o.alive]
        if not alive:
            return 0.0
        return float(np.mean([o.energy for o in alive]))
