"""Phase 3 - Memory Ecology.

A new kind of world where the same smell observation demands
different actions depending on history. Built as a subclass of
`World` so the legacy Phase 1.5/2.5 single-resource world remains
untouched.

Differences vs `World`:
  - Two food types (FoodA, FoodB) with distinct smells.
  - A global `season` flips reward sign every SEASON_LENGTH ticks.
    Organisms cannot sense season directly.
  - After `eat`, the organism receives an intake_feedback signal
    (+1 for positive reward, -1 for negative) that lasts
    INTAKE_FEEDBACK_DURATION ticks. The signal is exposed as a 7th
    sensor input but NOT as a regular smell sensor - it lives at
    a structurally distinct index and is meant to drive the
    hidden node via a dedicated heritable edge.
  - DualSmellField replaces the single SmellField.

Founder for this world has 7 sensors (6 smell + 1 feedback),
1 hidden, 2 motors, and 1 dedicated feedback->hidden edge with
weight FB_TO_HIDDEN_WEIGHT (initially fixed at 1.0 in v3.0; will
become a heritable gene in v3.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .brain import Brain, _activate_vec
from .config import (
    BIAS_MAX,
    CHILD_DISPERSAL_MAX,
    CHILD_DISPERSAL_MIN,
    CHILD_HEADING_NOISE,
    CONNECTION_METABOLIC_COST,
    EAT_RADIUS,
    FB_TO_HIDDEN_WEIGHT,
    FOOD_A_FRACTION,
    FOOD_A_NEGATIVE_ENERGY,
    FOOD_A_POSITIVE_ENERGY,
    FOOD_B_NEGATIVE_ENERGY,
    FOOD_B_POSITIVE_ENERGY,
    FOOD_REGROWTH_RATE,
    FOOD_TARGET,
    INITIAL_ENERGY,
    INITIAL_LOCOMOTION_BIAS_SIGMA,
    INITIAL_POPULATION,
    INITIAL_WEIGHT_SIGMA,
    INTAKE_FEEDBACK_DURATION,
    IDLE_ENERGY_COST,
    MAX_AGE,
    MAX_TURN_RATE,
    MOVE_DEADZONE,
    MOVE_ENERGY_COST,
    NEURON_METABOLIC_COST,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    SEASON_LENGTH,
    SMELL_HALF_ANGLE,
    TURN_ENERGY_COST,
)
from .events import EventLog
from .genome import Activation, ConnectionGene, Genome, NodeGene, NodeType
from .organism import Organism
from .sensors import DualSmellField
from .world import Food, World


# Phase 3 sensor layout (must match brain.make_default_genome below):
#   0..2 : FoodA smell left/front/right   (continuous in [0,1])
#   3..5 : FoodB smell left/front/right   (continuous in [0,1])
#   6    : intake_feedback                (in {-1, 0, +1})
N_SMELL_CHANNELS = 6
FEEDBACK_INDEX = 6
N_SENSORS_PHASE3 = 7


@dataclass
class FoodA(Food):
    """Food with FoodA smell hue and season-dependent reward."""
    pass


@dataclass
class FoodB(Food):
    """Food with FoodB smell hue and season-dependent reward."""
    pass


def make_phase3_founder(rng: np.random.Generator) -> Genome:
    """Build a Phase 3 founder genome.

    Topology:
      6 smell sensors  -> 1 hidden  (tanh, bias 0)
      1 feedback sensor -> 1 hidden  (fixed weight FB_TO_HIDDEN_WEIGHT)
      1 hidden         -> 2 motors  (tanh)

    Total: 9 connections.
    """
    g = Genome()
    sensor_ids: list[int] = []
    for _ in range(N_SENSORS_PHASE3):
        nid = len(g.nodes)
        g.nodes[nid] = NodeGene(
            id=nid, type=NodeType.SENSOR, activation=Activation.LINEAR
        )
        sensor_ids.append(nid)
    hidden_ids: list[int] = []
    for _ in range(1):  # N_HIDDEN=1 for Phase 3.0
        nid = len(g.nodes)
        g.nodes[nid] = NodeGene(
            id=nid, type=NodeType.HIDDEN, activation=Activation.TANH
        )
        hidden_ids.append(nid)
    motor_ids: list[int] = []
    for i in range(2):
        nid = len(g.nodes)
        if i == 0:
            bias = 0.0
        else:
            bias = float(
                np.clip(
                    rng.normal(0.0, INITIAL_LOCOMOTION_BIAS_SIGMA),
                    -BIAS_MAX,
                    BIAS_MAX,
                )
            )
        g.nodes[nid] = NodeGene(
            id=nid, type=NodeType.MOTOR,
            activation=Activation.TANH, bias=bias,
        )
        motor_ids.append(nid)

    sigma = INITIAL_WEIGHT_SIGMA
    innov = 0
    # 6 smell sensors -> hidden
    for src in sensor_ids[:N_SMELL_CHANNELS]:
        for h in hidden_ids:
            g.connections[innov] = ConnectionGene(
                innovation=innov,
                in_node=src, out_node=h,
                weight=float(rng.normal(0, sigma)),
                enabled=True,
            )
            innov += 1
    # feedback sensor (sensor[6]) -> hidden, fixed weight
    fb_id = sensor_ids[FEEDBACK_INDEX]
    for h in hidden_ids:
        g.connections[innov] = ConnectionGene(
            innovation=innov,
            in_node=fb_id, out_node=h,
            weight=FB_TO_HIDDEN_WEIGHT,
            enabled=True,
        )
        innov += 1
    # hidden -> 2 motors
    for h in hidden_ids:
        for m in motor_ids:
            g.connections[innov] = ConnectionGene(
                innovation=innov,
                in_node=h, out_node=m,
                weight=float(rng.normal(0, sigma)),
                enabled=True,
            )
            innov += 1
    g.max_innovation = innov
    return g


class MemoryEcologyWorld(World):
    """Two-resource + hidden-season + intake_feedback world.

    This subclass leaves the legacy World.__init__ flow intact
    (single smell field, single food list) for the *constructor*, then
    immediately swaps in dual-smell / dual-food / season state.
    """

    def __init__(
        self,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
        events: EventLog | None = None,
    ) -> None:
        # Run legacy constructor; we'll immediately overwrite smell and food.
        super().__init__(seed=seed, width=width, height=height, events=events)

        # Replace single SmellField with DualSmellField.
        self.dual_smell = DualSmellField(width, height)
        # Keep `self.smell` as a reference to field_a so legacy code paths
        # that read `self.smell` don't crash (e.g. Arena). It just won't
        # be the source of truth for MemoryEcologyWorld.
        self.smell = self.dual_smell.field_a

        # Rebuild food lists.
        self.food_a: list[FoodA] = []
        self.food_b: list[FoodB] = []
        # Clear legacy self.food; nothing should read it in this subclass.
        self.food = []  # type: ignore[assignment]

        # Season state.
        self.season: int = 0
        self.ticks_in_season: int = 0

        # Re-spawn founders with the Phase 3 genome (7 sensors, feedback edge).
        self.organisms = []
        self._next_id = 0
        for _ in range(INITIAL_POPULATION):
            self._spawn_founder()
        self.species_manager.sync(self.organisms, self.tick)

        # Initial food split per FOOD_A_FRACTION.
        n_a = int(FOOD_TARGET * FOOD_A_FRACTION)
        n_b = FOOD_TARGET - n_a
        for _ in range(n_a):
            self._spawn_food_a()
        for _ in range(n_b):
            self._spawn_food_b()

    # --- spawning ----------------------------------------------------------

    def _spawn_founder(self) -> None:
        """Phase 3 founder with 7 sensors and feedback edge."""
        import math

        genome = make_phase3_founder(rng=self.rng)
        for c in genome.connections.values():
            self.innovations.innovation_for(c.in_node, c.out_node)
        genome.max_innovation = max(
            (c.innovation for c in genome.connections.values()), default=0
        )
        oid = self._next_organism_id()
        org = Organism(
            id=oid,
            x=float(self.rng.uniform(0, self.width)),
            y=float(self.rng.uniform(0, self.height)),
            heading=float(self.rng.uniform(0, 2 * math.pi)),
            energy=INITIAL_ENERGY,
            brain=Brain(genome),
            genome=genome,
        )
        self.organisms.append(org)
        self.events.record_birth(
            self.tick, org_id=oid, parent_id=None,
            genome_hash=genome.fingerprint(),
        )

    def _spawn_food_a(self) -> None:
        self.food_a.append(
            FoodA(x=float(self.rng.uniform(0, self.width)),
                  y=float(self.rng.uniform(0, self.height)))
        )

    def _spawn_food_b(self) -> None:
        self.food_b.append(
            FoodB(x=float(self.rng.uniform(0, self.width)),
                  y=float(self.rng.uniform(0, self.height)))
        )

    # --- main loop ---------------------------------------------------------

    def step(self) -> None:
        """Advance one tick. Override to use dual smell + dual food + season."""
        self.tick += 1

        # 0. Season flip if needed.
        if self.ticks_in_season >= SEASON_LENGTH:
            self.season = 1 - self.season
            self.ticks_in_season = 0
        self.ticks_in_season += 1

        # 1. Regrow food per resource (binomial deficit vs target split).
        a_target = int(FOOD_TARGET * FOOD_A_FRACTION)
        b_target = FOOD_TARGET - a_target
        missing_a = max(0, a_target - len(self.food_a))
        missing_b = max(0, b_target - len(self.food_b))
        if missing_a > 0:
            n_new = int(self.rng.binomial(missing_a, FOOD_REGROWTH_RATE))
            for _ in range(n_new):
                self._spawn_food_a()
        if missing_b > 0:
            n_new = int(self.rng.binomial(missing_b, FOOD_REGROWTH_RATE))
            for _ in range(n_new):
                self._spawn_food_b()

        # 2. Recompute dual smell field.
        self.dual_smell.recompute_split(self.food_a, self.food_b)

        # 3. Decay intake feedback for living organisms.
        for org in self.organisms:
            if org.intake_feedback_ttl > 0:
                org.intake_feedback_ttl -= 1
                if org.intake_feedback_ttl == 0:
                    org.intake_feedback = 0.0

        # 4. Each organism acts.
        for org in self.organisms:
            if not org.alive:
                continue
            self._act(org)

        # 5. Resolve eat attempts (override for signed rewards).
        self._resolve_eat_phase3()

        # 6. Drain + death + reproduction + archive (same as legacy).
        survivors: list[Organism] = []
        for org in self.organisms:
            if not org.alive:
                continue
            org.energy -= IDLE_ENERGY_COST
            assert org.genome is not None
            n_nodes = len(org.genome.nodes)
            n_conns = sum(1 for c in org.genome.connections.values() if c.enabled)
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

        # 7. Reproduction.
        self._reproduce()

        # 8. Taxonomy + archive.
        self.species_manager.sync(self.organisms, self.tick)
        self.species_manager.maybe_refresh_representatives(
            self.organisms, self.tick
        )
        self._archive_tick_end()

    # --- per-organism action -------------------------------------------------

    def _sensors_for(self, org: Organism) -> NDArray[np.float32]:
        """6-vector smell (from dual field) + 1-element feedback channel.

        Order:
          [a_left, a_front, a_right,
           b_left, b_front, b_right,
           intake_feedback]
        """
        smell6 = self.dual_smell.sample(org.x, org.y, org.heading, SMELL_HALF_ANGLE)
        fb = float(org.intake_feedback) if org.intake_feedback_ttl > 0 else 0.0
        arr = np.empty(N_SENSORS_PHASE3, dtype=np.float32)
        arr[:N_SMELL_CHANNELS] = smell6
        arr[FEEDBACK_INDEX] = fb
        return arr

    def _resolve_eat_phase3(self) -> None:
        """Override _resolve_eat to use season-dependent rewards and to
        set intake_feedback on the organism."""
        for org in self.organisms:
            if not org.alive:
                continue
            for food_list in (self.food_a, self.food_b):
                if not food_list:
                    continue
                fx = np.array([f.x for f in food_list], dtype=np.float32)
                fy = np.array([f.y for f in food_list], dtype=np.float32)
                dx = fx - org.x
                dy = fy - org.y
                dx -= self.width * np.round(dx / self.width)
                dy -= self.height * np.round(dy / self.height)
                dist = np.hypot(dx, dy)
                idx = int(np.argmin(dist))
                if float(dist[idx]) <= EAT_RADIUS:
                    eaten = food_list.pop(idx)
                    # Season-dependent reward.
                    if isinstance(eaten, FoodA):
                        reward = (
                            FOOD_A_POSITIVE_ENERGY
                            if self.season == 0
                            else FOOD_A_NEGATIVE_ENERGY
                        )
                    else:
                        reward = (
                            FOOD_B_POSITIVE_ENERGY
                            if self.season == 1
                            else FOOD_B_NEGATIVE_ENERGY
                        )
                    org.energy += reward
                    org.food_eaten += 1
                    if org.time_to_first_food is None:
                        org.time_to_first_food = org.age
                    # Set intake_feedback signal.
                    if reward > 0:
                        org.intake_feedback = +1.0
                    else:
                        org.intake_feedback = -1.0
                    org.intake_feedback_ttl = INTAKE_FEEDBACK_DURATION
                    org.last_intake_feedback = org.intake_feedback
                    self.events.record_eat(
                        self.tick, org_id=org.id,
                        x=eaten.x, y=eaten.y,
                    )
