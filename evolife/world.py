"""World — the simulation driver.

Owns the organism list, the food list, the RNG, the innovation database
and the tick counter. Each tick: spawn food, ask each organism for
actions, integrate motion, resolve eat attempts, deduct energy, kill
the starving, attempt reproduction, run a tournament sweep.

v1 introduces:
- InnovationDatabase shared across organisms.
- Tournament selection: every TOURNAMENT_EVERY ticks, every organism is
  paired with its nearest neighbour inside TOURNAMENT_RADIUS. With
  probability TOURNAMENT_KILL_RATE, the worse of the two dies and the
  better one clones (with mutation) nearby.
- Structural mutations enabled (add_node, add_connection, toggle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .brain import Brain
from .config import (
    ADD_CONNECTION_RATE,
    ADD_NODE_RATE,
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
    TOGGLE_CONNECTION_RATE,
    TOURNAMENT_CLONE_ENERGY,
    TOURNAMENT_EVERY,
    TOURNAMENT_KILL_RATE,
    TOURNAMENT_RADIUS,
    TURN_ENERGY_COST,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from .genome import Genome
from .innovation import InnovationDatabase
from .mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_toggle_connection,
    mutate_weights,
)
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
        self.innovations = InnovationDatabase()
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
        # Seed the innovation database with the founder connections so
        # later add_connection/add_node don't collide with them.
        for c in genome.connections.values():
            self.innovations.innovation_for(c.in_node, c.out_node)
        genome.max_innovation = max(
            (c.innovation for c in genome.connections.values()), default=0
        )
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

        # 5. Reproduction (existing behaviour).
        self._reproduce()

        # 6. Tournament selection (v1).
        if self.tick % TOURNAMENT_EVERY == 0:
            self._tournament()

    # --- per-organism action ----------------------------------------------

    def _act(self, org: Organism) -> None:
        """Run the organism brain for one tick and integrate motion."""
        sensors = self._sensors_for(org)
        assert org.brain is not None
        motors = org.brain.forward(sensors)

        turn = float(motors[0]) * MAX_TURN_RATE
        move = float(np.clip(motors[1], 0.0, 1.0)) * MAX_LINEAR_SPEED

        org.energy -= abs(turn) * TURN_ENERGY_COST
        org.energy -= move * MOVE_ENERGY_COST

        org.heading = (org.heading + turn) % (2 * np.pi)
        org.x = (org.x + np.cos(org.heading) * move) % self.width
        org.y = (org.y + np.sin(org.heading) * move) % self.height

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
        speed = 0.0
        bias = 1.0
        return np.array(
            [food_angle, food_dist, energy_norm, speed, bias],
            dtype=np.float32,
        )

    # --- resolution --------------------------------------------------------

    def _resolve_eat(self) -> None:
        if not self.food:
            return
        for org in self.organisms:
            if not org.alive or not getattr(org, "_eat_attempt", False):
                continue
            if not self.food:
                break
            fx = np.array([f.x for f in self.food])
            fy = np.array([f.y for f in self.food])
            dx = fx - org.x
            dy = fy - org.y
            dx -= self.width * np.round(dx / self.width)
            dy -= self.height * np.round(dy / self.height)
            dist = np.hypot(dx, dy)
            idx = int(np.argmin(dist))
            if float(dist[idx]) <= EAT_RADIUS:
                self.food.pop(idx)
                org.energy += FOOD_ENERGY

    def _reproduce(self) -> None:
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

            assert org.genome is not None
            child_genome = self._mutate(org.genome)

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

    # --- v1: tournament selection -----------------------------------------

    def _tournament(self) -> None:
        """Pair each organism with its nearest neighbour, soft-select.

        With probability TOURNAMENT_KILL_RATE per pair, the worse (by
        peak_energy) dies and the better clones nearby. The clone goes
        through the same mutation pipeline as reproduction.

        Implemented with a coarse spatial bucket grid keyed by
        TOURNAMENT_RADIUS so the per-pair search is O(neighbours) rather
        than O(n). With POPULATION_CAP=500 and a 512x512 world, the
        bucket grid is small (~16x16) and very effective.
        """
        alive = [o for o in self.organisms if o.alive]
        if len(alive) < 2:
            return

        # Build bucket grid.
        cell = TOURNAMENT_RADIUS
        nx = max(1, int(math.ceil(self.width / cell)))
        ny = max(1, int(math.ceil(self.height / cell)))
        buckets: dict[tuple[int, int], list[Organism]] = {}
        for o in alive:
            bx = int(o.x // cell) % nx
            by = int(o.y // cell) % ny
            buckets.setdefault((bx, by), []).append(o)

        # Helper: collect neighbours within radius via toroidal wrap.
        def neighbours(o: Organism) -> list[Organism]:
            cx = int(o.x // cell) % nx
            cy = int(o.y // cell) % ny
            result: list[Organism] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = buckets.get(((cx + dx) % nx, (cy + dy) % ny))
                    if b:
                        result.extend(b)
            return result

        # We must NOT mutate self.organisms while iterating over `alive`,
        # so collect a list of (winner, loser) pairs first.
        pairs: list[tuple[Organism, Organism]] = []
        for a in alive:
            best_dist = float("inf")
            best_other: Organism | None = None
            ax, ay = a.x, a.y
            for b in neighbours(a):
                if b is a:
                    continue
                dx = b.x - ax
                dy = b.y - ay
                if dx > self.width / 2:
                    dx -= self.width
                elif dx < -self.width / 2:
                    dx += self.width
                if dy > self.height / 2:
                    dy -= self.height
                elif dy < -self.height / 2:
                    dy += self.height
                d2 = dx * dx + dy * dy
                if d2 < best_dist:
                    best_dist = d2
                    best_other = b
            if best_other is None:
                continue
            if best_dist > TOURNAMENT_RADIUS * TOURNAMENT_RADIUS:
                continue
            if self.rng.random() >= TOURNAMENT_KILL_RATE:
                continue
            winner, loser = (
                (a, best_other)
                if a.peak_energy >= best_other.peak_energy
                else (best_other, a)
            )
            pairs.append((winner, loser))

        # Apply pairs.
        alive_count = len(alive)
        for winner, loser in pairs:
            if not loser.alive:
                continue  # may have lost already this tournament
            loser.alive = False
            alive_count -= 1
            if alive_count >= POPULATION_CAP:
                continue
            assert winner.genome is not None
            clone_genome = self._mutate(winner.genome)
            clone = Organism(
                id=self._next_organism_id(),
                x=float(winner.x),
                y=float(winner.y),
                heading=float(self.rng.uniform(0, 2 * math.pi)),
                energy=TOURNAMENT_CLONE_ENERGY,
                brain=Brain(clone_genome),
                genome=clone_genome,
                parent_id=winner.id,
            )
            self.organisms.append(clone)
            winner.children += 1
            alive_count += 1

    def _mutate(self, genome: Genome) -> Genome:
        """Run the full v1 mutation pipeline on a genome copy."""
        g = mutate_weights(genome, self.rng)
        g = mutate_add_connection(g, self.rng, self.innovations)
        g = mutate_add_node(g, self.rng, self.innovations)
        g = mutate_toggle_connection(g, self.rng)
        return g

    # --- diagnostics -------------------------------------------------------

    def population(self) -> int:
        return sum(1 for o in self.organisms if o.alive)

    def mean_energy(self) -> float:
        alive = [o for o in self.organisms if o.alive]
        if not alive:
            return 0.0
        return float(np.mean([o.energy for o in alive]))
