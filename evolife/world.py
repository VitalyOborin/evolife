"""World — the simulation driver.

v2.2 (Viable Replicator):

- No tournament. No external fitness function. Selection = "did you eat
  enough to not starve and to afford reproduction?"
- Proto-brain: 3 smell sensors wired to 2 motors, no hidden neurons
  and no gifted memory. Complexity can arise via structural mutation.
- Local smell sensors: three probe points (left/front/right). No GPS,
  no energy/bias sensors, no nearest-food lookup.
- Metabolic cost: each neuron and active connection drains a small
  amount of energy per tick. Bigger brains are more expensive.
- Food respawns in proportion to how far the field is below target.
- Event log: append-only Birth/Death/Reproduction events so the lineage
  tree can be reconstructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .brain import Brain
from .config import (
    CHILD_DISPERSAL_MAX,
    CHILD_DISPERSAL_MIN,
    CHILD_HEADING_NOISE,
    CONNECTION_METABOLIC_COST,
    EAT_RADIUS,
    FOOD_ENERGY,
    FOOD_REGROWTH_RATE,
    FOOD_TARGET,
    IDLE_ENERGY_COST,
    INITIAL_ENERGY,
    INITIAL_POPULATION,
    MAX_AGE,
    MAX_TURN_RATE,
    MOVE_ENERGY_COST,
    NEURON_METABOLIC_COST,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    SMELL_HALF_ANGLE,
    TURN_ENERGY_COST,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    locomotion_speed,
)
from .archive import Archive, MilestoneKind
from .behavior import note_act
from .events import EventLog
from .genome import Genome
from .innovation import InnovationDatabase
from .speciation import SpeciesManager
from .mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_biases,
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
        self._init_state(seed=seed, width=width, height=height, events=events)
        self._populate_initial()

    def _init_state(
        self,
        *,
        seed: int | None,
        width: int,
        height: int,
        events: EventLog | None,
    ) -> None:
        """Initialise empty world state. Subclasses can extend but should
        call super()._init_state() first.
        """
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.innovations = InnovationDatabase()
        self.species_manager = SpeciesManager()
        self.events = events if events is not None else EventLog()
        self.archive = Archive()
        self.tick: int = 0
        self.organisms: list[Organism] = []
        self.food: list[Food] = []
        self.smell = SmellField(width, height)
        self._next_id: int = 0
        self.tick_births: int = 0
        self.tick_mutations: int = 0

    def _populate_initial(self) -> None:
        """Spawn founders + initial food. Subclasses can override."""
        for _ in range(INITIAL_POPULATION):
            self._spawn_founder()
        self.species_manager.sync(self.organisms, self.tick)
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
        # Pass the world's RNG so each founder has different weights.
        genome = Brain.make_default_genome(rng=self.rng)
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
            generation=0,
            founder_lineage_id=oid,
        )
        org.species_id = self.species_manager.assign(
            genome, self.tick, oid, parent_species_id=None
        )
        self.organisms.append(org)
        self.events.record_birth(
            self.tick, org_id=oid, parent_id=None, genome_hash=genome.fingerprint()
        )

    # --- main loop ---------------------------------------------------------

    def step(self) -> None:
        """Advance one tick."""
        self.tick += 1
        self.tick_births = 0
        self.tick_mutations = 0

        # 1. Regrow food in proportion to the deficit vs FOOD_TARGET.
        missing = FOOD_TARGET - len(self.food)
        if missing > 0:
            n_new = int(self.rng.binomial(missing, FOOD_REGROWTH_RATE))
            for _ in range(n_new):
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

        # 7. Observational taxonomy: counts, extinctions, representatives.
        self.species_manager.sync(self.organisms, self.tick)
        self.species_manager.maybe_refresh_representatives(
            self.organisms, self.tick
        )

        # 8. Archive updates (observability only).
        self._archive_tick_end()

    # --- per-organism action ----------------------------------------------

    def _act(self, org: Organism) -> None:
        """Run the organism brain for one tick and integrate motion."""
        sensors = self._sensors_for(org)
        assert org.brain is not None
        motors = org.brain.forward(sensors)

        # Motors: [turn_drive, locomotion_drive], both in [-1, 1].
        # Rest (locomotion <= 0) is a first-class action: no translation
        # cost. Turning in place is allowed and still costs energy.
        # Basal locomotion is a heritable motor bias, not a forced gait.
        turn = float(motors[0]) * MAX_TURN_RATE
        move = locomotion_speed(float(motors[1]))

        org.energy -= abs(turn) * TURN_ENERGY_COST
        org.energy -= move * MOVE_ENERGY_COST

        org.heading = (org.heading + turn) % (2 * np.pi)
        org.x = (org.x + np.cos(org.heading) * move) % self.width
        org.y = (org.y + np.sin(org.heading) * move) % self.height
        note_act(org, sensors, turn, move, self.width, self.height)

        # Automatic contact eating: any food within EAT_RADIUS is eaten.
        # No motor required.

    def _sensors_for(self, org: Organism) -> np.ndarray:
        """Local smell sensors: left, front, right. Nothing else."""
        smells = self.smell.sample(org.x, org.y, org.heading, SMELL_HALF_ANGLE)
        return np.array(
            [float(smells[0]), float(smells[1]), float(smells[2])],
            dtype=np.float32,
        )

    # --- resolution --------------------------------------------------------

    def _resolve_eat(self) -> list[tuple[Organism, tuple[float, float]]]:
        """Automatic contact eating: nearest food within EAT_RADIUS.

        No brain decision required. The brain only controls turn and
        move; once the organism happens to be close enough to food, it
        eats. This is the v2.x simplification: the brain evolves
        navigation, not the decision to eat.

        Returns a list of (organism, food_pos) pairs so the caller can
        log Eat events.
        """
        out: list[tuple[Organism, tuple[float, float]]] = []
        if not self.food:
            return out
        for org in self.organisms:
            if not org.alive:
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
                org.food_eaten += 1
                if org.time_to_first_food is None:
                    org.time_to_first_food = org.age
                out.append((org, (eaten_food.x, eaten_food.y)))
        return out

    def _reproduce(self) -> None:
        """Automatic energy-based reproduction.

        No brain decision required. Once an organism's energy exceeds
        REPRODUCTION_THRESHOLD it splits into two: parent keeps the
        excess above the threshold, child starts with REPRODUCTION_ENERGY.
        """
        if len(self.organisms) >= POPULATION_CAP:
            return
        new_organisms: list[Organism] = []
        for org in self.organisms:
            if not org.alive:
                continue
            if org.energy < REPRODUCTION_THRESHOLD:
                continue
            if len(self.organisms) + len(new_organisms) >= POPULATION_CAP:
                break

            assert org.genome is not None
            child_genome = self._mutate(org.genome)

            org.energy -= REPRODUCTION_ENERGY
            child_id = self._next_organism_id()
            child_heading = float(
                (org.heading + self.rng.normal(0.0, CHILD_HEADING_NOISE))
                % (2 * np.pi)
            )
            distance = float(
                self.rng.uniform(CHILD_DISPERSAL_MIN, CHILD_DISPERSAL_MAX)
            )
            child_x = (org.x + math.cos(child_heading) * distance) % self.width
            child_y = (org.y + math.sin(child_heading) * distance) % self.height
            child = Organism(
                id=child_id,
                x=child_x,
                y=child_y,
                heading=child_heading,
                energy=REPRODUCTION_ENERGY,
                brain=Brain(child_genome),
                genome=child_genome,
                parent_id=org.id,
                generation=org.generation + 1,
                founder_lineage_id=org.founder_lineage_id,
            )
            child.species_id = self.species_manager.assign(
                child_genome,
                self.tick,
                child_id,
                parent_species_id=org.species_id,
            )
            new_organisms.append(child)
            org.children += 1
            self.events.record_reproduction(
                self.tick,
                parent_id=org.id,
                child_id=child_id,
                child_genome_hash=child_genome.fingerprint(),
            )
            self._check_milestones(child_genome, child_id, parent_genome=org.genome, parent_id=org.id)
        self.organisms.extend(new_organisms)

    def _check_milestones(
        self,
        genome: Genome,
        org_id: int,
        parent_genome: Genome | None = None,
        parent_id: int | None = None,
    ) -> None:
        """Update Archive with newly-evolved structural features.

        This is purely observational: it does not affect fitness,
        reproduction, or survival.

        When `parent_genome` is provided and the child carries a
        hidden<->hidden cycle, both the cycle carrier and its parent
        are stored in `archive.cycle_carriers` so the Behavioral Arena
        can replay them side-by-side. The parent is guaranteed to be
        cycle-free because the cycle just appeared in the child via
        structural mutation.
        """
        has_hidden = any(
            n.type.value == "hidden" for n in genome.nodes.values()
        )
        if has_hidden:
            self.archive.maybe_fire(
                MilestoneKind.FIRST_HIDDEN_NODE,
                self.tick,
                value=1,
                payload={
                    "org_id": org_id,
                    "genome_hash": genome.fingerprint(),
                },
            )
        # Recurrent cycle: two enabled connections A -> B and B -> A
        # both enabled, where both A and B are hidden nodes.
        # Cycles through sensors or motors are trivial wiring
        # artefacts (motor->sensor just feeds back to a sensor input)
        # and do not constitute internal memory.
        node_types = {n.id: n.type for n in genome.nodes.values()}
        edges = {
            (c.in_node, c.out_node)
            for c in genome.connections.values()
            if c.enabled
        }
        has_cycle = False
        for a, b in edges:
            if (b, a) in edges and a != b:
                ta = node_types.get(a)
                tb = node_types.get(b)
                if (
                    ta is not None
                    and tb is not None
                    and ta.value == "hidden"
                    and tb.value == "hidden"
                ):
                    has_cycle = True
                    break
        if has_cycle:
            self.archive.maybe_fire(
                MilestoneKind.FIRST_RECURRENT_CYCLE,
                self.tick,
                value=1,
                payload={
                    "org_id": org_id,
                    "genome_hash": genome.fingerprint(),
                },
            )
            # Capture cycle carrier + parent for later arena comparison.
            if parent_genome is not None and parent_id is not None:
                from .archive import CycleCarrier
                self.archive.cycle_carriers.append(
                    CycleCarrier(
                        tick=self.tick,
                        child_id=org_id,
                        parent_id=parent_id,
                        parent_genome=parent_genome,
                        cycle_genome=genome,
                    )
                )
        # Numeric maxima.
        n_nodes = len(genome.nodes)
        self.archive.record_max(
            MilestoneKind.MAX_BRAIN_NODES,
            self.tick,
            n_nodes,
            payload={
                "org_id": org_id,
                "genome_hash": genome.fingerprint(),
            },
        )

    def _archive_tick_end(self) -> None:
        """Run archive updates that depend on the whole population."""
        alive = [o for o in self.organisms if o.alive]
        if not alive:
            return
        max_gen = max(o.generation for o in alive)
        self.archive.record_max(
            MilestoneKind.MAX_GENERATION_REACHED,
            self.tick,
            max_gen,
        )
        # Largest founder lineage by alive count.
        from collections import Counter
        counts = Counter(o.founder_lineage_id for o in alive)
        if counts:
            top_lineage, top_size = max(counts.items(), key=lambda kv: kv[1])
            self.archive.record_max(
                MilestoneKind.MAX_LINEAGE_SIZE,
                self.tick,
                top_size,
                payload={"lineage_id": int(top_lineage)},
            )

    def _mutate(self, genome: Genome) -> Genome:
        self.tick_births += 1
        g = mutate_weights(genome, self.rng)
        g = mutate_biases(g, self.rng)
        g = mutate_add_connection(g, self.rng, self.innovations)
        g = mutate_add_node(g, self.rng, self.innovations)
        g = mutate_toggle_connection(g, self.rng)
        if g is not genome:
            self.tick_mutations += 1
        return g

    # --- diagnostics -------------------------------------------------------

    def population(self) -> int:
        return sum(1 for o in self.organisms if o.alive)

    def mean_energy(self) -> float:
        alive = [o for o in self.organisms if o.alive]
        if not alive:
            return 0.0
        return float(np.mean([o.energy for o in alive]))

    def max_generation(self) -> int:
        alive = [o for o in self.organisms if o.alive]
        if not alive:
            return 0
        return max(o.generation for o in alive)

    def n_lineages(self) -> int:
        return len(
            {o.founder_lineage_id for o in self.organisms if o.alive}
        )

    def n_species(self) -> int:
        return len(self.species_manager.living())

    def n_established_species(self) -> int:
        return len(self.species_manager.established_living())

    def median_movement_transitions(self) -> float:
        alive = [o.movement_transitions for o in self.organisms if o.alive]
        if not alive:
            return 0.0
        return float(np.median(alive))
