"""World — the simulation driver.

v2 (Natural Selection Baseline):

- No tournament. No external fitness function. Selection = "did you eat
  enough to not starve and to afford reproduction?"
- Recurrent brains: each organism owns persistent brain_state across
  ticks. Connections of any topology (including cycles, hidden->sensor,
  motor->hidden) are valid because the executor computes
      values_next = activation(bias + sum_i(values_i * w))
  in a fixed-point-ish way that preserves information across ticks.
- Local smell sensors: three sectors (left/front/right) sampled from a
  precomputed smell field. No GPS, no nearest-food lookup.
- Metabolic cost: each neuron and active connection drains a small
  amount of energy per tick. Bigger brains are more expensive.
- Event log: append-only Birth/Death/Reproduction events so the lineage
  tree can be reconstructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .brain import Brain
from .config import (
    ADD_CONNECTION_RATE,
    ADD_NODE_RATE,
    CONNECTION_METABOLIC_COST,
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
    NEURON_METABOLIC_COST,
    N_SENSORS,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    SMELL_HALF_ANGLE,
    TOGGLE_CONNECTION_RATE,
    TURN_ENERGY_COST,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from .events import EventLog
from .genome import Genome
from .innovation import InnovationDatabase
from .mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_toggle_connection,
    mutate_weights,
)
from .organism import Organism
from .sensors import SmellField


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
        events: EventLog | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.innovations = InnovationDatabase()
        self.events = events if events is not None else EventLog()
        self.tick: int = 0
        self.organisms: list[Organism] = []
        self.food: list[Food] = []
        self.smell = SmellField(width, height)
        self._next_id: int = 0

        for _ in range(INITIAL_POPULATION):
            self._spawn_founder()

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
        for c in genome.connections.values():
            self.innovations.innovation_for(c.in_node, c.out_node)
        genome.max_innovation = max(
            (c.innovation for c in genome.connections.values()), default=0
        )
        brain = Brain(genome)
        oid = self._next_organism_id()
        org = Organism(
            id=oid,
            x=float(self.rng.uniform(0, self.width)),
            y=float(self.rng.uniform(0, self.height)),
            heading=float(self.rng.uniform(0, 2 * np.pi)),
            energy=INITIAL_ENERGY,
            brain=brain,
            genome=genome,
        )
        self.organisms.append(org)
        self.events.record_birth(
            self.tick, org_id=oid, parent_id=None, genome_hash=genome.fingerprint()
        )

    # --- main loop ---------------------------------------------------------

    def step(self) -> None:
        """Advance one tick."""
        self.tick += 1

        # 1. Spawn food up to target.
        while len(self.food) < FOOD_TARGET and self.rng.random() < FOOD_SPAWN_RATE:
            self._spawn_food()

        # 2. Recompute smell field for this tick.
        self.smell.recompute(self.food)

        # 3. Each organism acts.
        for org in self.organisms:
            if not org.alive:
                continue
            self._act(org)

        # 4. Resolve eat attempts.
        eaten = self._resolve_eat()
        for org, food_pos in eaten:
            self.events.record_eat(
                self.tick, org_id=org.id, x=food_pos[0], y=food_pos[1]
            )

        # 5. Passive energy drain + metabolic cost + age + death.
        survivors: list[Organism] = []
        for org in self.organisms:
            if not org.alive:
                continue
            org.energy -= IDLE_ENERGY_COST
            assert org.genome is not None
            n_nodes = len(org.genome.nodes)
            n_conns = sum(
                1 for c in org.genome.connections.values() if c.enabled
            )
            org.energy -= (
                NEURON_METABOLIC_COST * n_nodes
                + CONNECTION_METABOLIC_COST * n_conns
            )
            org.age += 1
            if org.energy <= 0.0 or org.age >= MAX_AGE:
                cause = "starvation" if org.energy <= 0.0 else "old_age"
                org.alive = False
                self.events.record_death(self.tick, org_id=org.id, cause=cause)
                continue
            org.peak_energy = max(org.peak_energy, org.energy)
            survivors.append(org)
        self.organisms = survivors

        # 6. Reproduction.
        self._reproduce()

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
        """Local smell sensors: left, front, right sectors + energy + bias."""
        # Smell field sampling. SmellField.sample returns intensities for
        # sectors at the organism's position.
        smells = self.smell.sample(org.x, org.y, org.heading, SMELL_HALF_ANGLE)
        # smells shape: (3,) in [0, 1].
        energy_norm = float(np.clip(org.energy / REPRODUCTION_THRESHOLD, 0.0, 1.0))
        bias = 1.0
        return np.array(
            [
                float(smells[0]),
                float(smells[1]),
                float(smells[2]),
                energy_norm,
                bias,
            ],
            dtype=np.float32,
        )

    # --- resolution --------------------------------------------------------

    def _resolve_eat(self) -> list[tuple[Organism, tuple[float, float]]]:
        """For each organism that wants to eat, consume the nearest food.

        Returns a list of (organism, food_pos) pairs so the caller can log
        Eat events.
        """
        out: list[tuple[Organism, tuple[float, float]]] = []
        if not self.food:
            return out
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
                eaten_food = self.food.pop(idx)
                org.energy += FOOD_ENERGY
                out.append((org, (eaten_food.x, eaten_food.y)))
        return out

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
            child_id = self._next_organism_id()
            child = Organism(
                id=child_id,
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
            self.events.record_reproduction(
                self.tick,
                parent_id=org.id,
                child_id=child_id,
                child_genome_hash=child_genome.fingerprint(),
            )
        self.organisms.extend(new_organisms)

    def _mutate(self, genome: Genome) -> Genome:
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
