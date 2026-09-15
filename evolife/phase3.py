"""Phase 3 - Memory Ecology.

A new kind of world where the same smell observation demands
different actions depending on history. Built as a subclass of
`World` so the legacy Phase 1.5/2.5 single-resource world remains
untouched.

Three modes (`mode` argument to MemoryEcologyWorld):
  "static_dual"     - two food types, no season flip. Control arm
                      for "is recurrence selected just by having two
                      food types?".
  "visible_season"  - two food types + season flip + season sensor.
                      Control arm for "is recurrence selected just by
                      having a season flip visible to the brain?".
  "hidden_season"   - two food types + season flip + no season sensor.
                      Only post-eat feedback reveals the reward sign.
                      This is the actual hypothesis test.

Phase 3.1 design changes from 3.0:
  - A/B smell are *distinct* sensor channels (6 smell + 1 feedback =
    7 sensors). Organism can actually choose A vs B.
  - Founder is built from a warm-start Phase 1.5 evolved genome
    (good at navigation); old `sensor -> hidden` edges are split
    into `A_sensor -> hidden` and `B_sensor -> hidden` with the
    same weight, so the cold-start brain still navigates.
  - MemoryEcologyWorld._init_state / _populate_initial override
    cleanly: empty state then Phase 3 population. No "create legacy
    world then throw it away".
  - FB_TO_HIDDEN_WEIGHT starts at 0.2, not 1.0. The channel exists
    but does not dominate the navigation reflex.

Phase 3.1 deliverables:
  - scripts/run_phase3.py supports --mode {static_dual, visible_season,
    hidden_season}.
  - scripts/arena_memory_advantage.py ablates only recurrent edges
    (SCC-based), not full brain state. Runs through real world.step().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .brain import Brain
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
    IDLE_ENERGY_COST,
    PHASE3_GRACE_PENALTY,
    PHASE3_GRACE_TICKS,
    INITIAL_LOCOMOTION_BIAS_SIGMA,
    INITIAL_POPULATION,
    INITIAL_WEIGHT_SIGMA,
    INTAKE_FEEDBACK_DURATION,
    MAX_AGE,
    MOVE_DEADZONE,
    NEURON_METABOLIC_COST,
    PHASE3_FOOD_A_NEGATIVE_ENERGY,
    PHASE3_FOOD_B_NEGATIVE_ENERGY,
    PHASE3_FOOD_TARGET,
    PHASE3_INITIAL_ENERGY,
    PHASE3_REPRODUCTION_THRESHOLD,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    SEASON_LENGTH,
    SMELL_HALF_ANGLE,
)
from .events import EventLog
from .genome import Activation, ConnectionGene, Genome, NodeGene, NodeType
from .innovation import InnovationDatabase
from .organism import Organism
from .sensors import DualSmellField
from .speciation import SpeciesManager
from .world import Food, World


# Phase 3.1 sensor layout. Distinct A/B channels + feedback. 7 sensors
# total in hidden-season mode; 8 in visible-season mode.
#
#   0: a_left
#   1: a_front
#   2: a_right
#   3: b_left
#   4: b_front
#   5: b_right
#   6: intake_feedback  (in {-1, 0, +1})
#   7: season           (in {0, 1})   -- only if PHASE3_VISIBLE_SEASON_SENSOR
N_SMELL_A = 3  # a_left, a_front, a_right
N_SMELL_B = 3  # b_left, b_front, b_right
FEEDBACK_INDEX = 6
SEASON_INDEX = 7


@dataclass
class FoodA(Food):
    """Food with FoodA smell hue and season-dependent reward."""


@dataclass
class FoodB(Food):
    """Food with FoodB smell hue and season-dependent reward."""


def _smell_channel(world: World, org: Organism) -> NDArray[np.float32]:
    """Read both smell fields and concatenate as 6-vector."""
    a = world.dual_smell.field_a.sample(
        org.x, org.y, org.heading, SMELL_HALF_ANGLE
    )
    b = world.dual_smell.field_b.sample(
        org.x, org.y, org.heading, SMELL_HALF_ANGLE
    )
    return np.concatenate([a, b]).astype(np.float32)


def make_phase3_founder_from_warmstart(
    rng: np.random.Generator,
    warm_genome: Genome | None = None,
    visible_season_sensor: bool = False,
) -> Genome:
    """Build a Phase 3 founder.

    If `warm_genome` is provided (a Phase 1.5 3-smell-sensor genome),
    we graft the new sensors by duplicating the smell edges: each
    old "smell_left -> hidden" edge becomes both "a_left -> hidden"
    and "b_left -> hidden" with the same weight. The cold-start
    brain behaves like the warm genome on A+B (both treated as
    "food"), but evolution can independently tune A vs B.

    If `warm_genome` is None, a fresh random founder is built with
    all weights drawn from INITIAL_WEIGHT_SIGMA. This is the cold
    start; it will likely go extinct, which is itself useful
    evidence that warm-start is needed.
    """
    g = Genome()
    n_smell_sensors = N_SMELL_A + N_SMELL_B
    n_extra_sensors = 1 + (1 if visible_season_sensor else 0)
    n_total_sensors = n_smell_sensors + n_extra_sensors

    sensor_ids: list[int] = []
    for _ in range(n_total_sensors):
        nid = len(g.nodes)
        g.nodes[nid] = NodeGene(
            id=nid, type=NodeType.SENSOR, activation=Activation.LINEAR
        )
        sensor_ids.append(nid)

    hidden_ids: list[int] = []
    n_hidden = 1
    for _ in range(n_hidden):
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

    innov = 0
    sigma = INITIAL_WEIGHT_SIGMA

    if warm_genome is not None:
        # Warm-start: take the 3 smell edges from the warm genome's
        # hidden node and split each into a_left / b_left etc.
        warm_hidden_id = None
        warm_sensor_weights: dict[int, float] = {}
        for n in warm_genome.nodes.values():
            if n.type.value == "hidden":
                warm_hidden_id = n.id
                break
        if warm_hidden_id is None:
            warm_hidden_id = 3  # default fallback for 3-sensor founder
        # Collect old sensor->hidden weights from warm genome by index.
        for c in warm_genome.connections.values():
            if (
                c.enabled
                and warm_genome.nodes[c.in_node].type.value == "sensor"
                and c.out_node == warm_hidden_id
            ):
                warm_sensor_weights[c.in_node] = c.weight
        # The 3-sensor founder has sensor ids 0,1,2 in left/front/right
        # order. Map them to A_left, A_front, A_right (0..2) and
        # B_left, B_front, B_right (3..5) with the same weight.
        sensor_role_to_warm_id = {
            0: 0,  # a_left   <- warm_left
            1: 1,  # a_front  <- warm_front
            2: 2,  # a_right  <- warm_right
            3: 0,  # b_left   <- warm_left
            4: 1,  # b_front  <- warm_front
            5: 2,  # b_right  <- warm_right
        }
        for new_sensor_idx in range(n_smell_sensors):
            warm_id = sensor_role_to_warm_id[new_sensor_idx]
            w = warm_sensor_weights.get(warm_id, float(rng.normal(0, sigma)))
            for h in hidden_ids:
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=sensor_ids[new_sensor_idx], out_node=h,
                    weight=w, enabled=True,
                )
                innov += 1
        # hidden -> motor edges: copy weights from warm genome if
        # available, else random.
        warm_motor_weights: dict[int, float] = {}
        for c in warm_genome.connections.values():
            if (
                c.enabled
                and warm_genome.nodes[c.in_node].type.value == "hidden"
                and warm_genome.nodes[c.out_node].type.value == "motor"
            ):
                warm_motor_weights[c.out_node] = c.weight
        # motor_ids here is a list of node ids (int), not NodeGene.
        # Find the warm-genome motor by index to copy weight.
        warm_motor_ids = sorted(
            nid for nid, n in warm_genome.nodes.items()
            if n.type.value == "motor"
        )
        for h in hidden_ids:
            for mi, m_id in enumerate(motor_ids):
                warm_mid = warm_motor_ids[mi] if mi < len(warm_motor_ids) else None
                w = (
                    warm_motor_weights.get(warm_mid, float(rng.normal(0, sigma)))
                    if warm_mid is not None
                    else float(rng.normal(0, sigma))
                )
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=h, out_node=m_id, weight=w, enabled=True,
                )
                innov += 1
    else:
        # Cold start: random weights on all smell->hidden and hidden->motor.
        for src in sensor_ids[:n_smell_sensors]:
            for h in hidden_ids:
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=src, out_node=h,
                    weight=float(rng.normal(0, sigma)),
                    enabled=True,
                )
                innov += 1
        for h in hidden_ids:
            for m in motor_ids:
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=h, out_node=m,
                    weight=float(rng.normal(0, sigma)),
                    enabled=True,
                )
                innov += 1

    # Feedback edge: fixed weight, small.
    fb_sensor_id = sensor_ids[FEEDBACK_INDEX]
    for h in hidden_ids:
        g.connections[innov] = ConnectionGene(
            innovation=innov,
            in_node=fb_sensor_id, out_node=h,
            weight=FB_TO_HIDDEN_WEIGHT, enabled=True,
        )
        innov += 1

    # Optional season sensor edge: zero weight, gives evolution a
    # knob to turn on if useful.
    if visible_season_sensor:
        season_sensor_id = sensor_ids[SEASON_INDEX]
        for h in hidden_ids:
            g.connections[innov] = ConnectionGene(
                innovation=innov,
                in_node=season_sensor_id, out_node=h,
                weight=0.0, enabled=True,
            )
            innov += 1

    g.max_innovation = innov
    return g


class MemoryEcologyWorld(World):
    """Two-resource + hidden-season + intake_feedback world.

    Subclasses the legacy World but bypasses World.__init__'s full
    bootstrap: we call _init_state() to set up the bare world state,
    then Phase 3-specific population and food spawning, then
    _populate_initial() (overridden) handles the rest.
    """

    def __init__(
        self,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
        events: EventLog | None = None,
        mode: str = "hidden_season",
        warm_genome: Genome | None = None,
        visible_season_sensor: bool = False,
    ) -> None:
        if mode not in ("static_dual", "visible_season", "hidden_season"):
            raise ValueError(f"unknown phase 3 mode: {mode}")

        self.mode = mode
        self.visible_season_sensor = visible_season_sensor
        self.season_enabled = mode in ("visible_season", "hidden_season")
        # Initialise empty world state via the base helper.
        self._init_state(
            seed=seed, width=width, height=height, events=events,
        )
        # Phase 3 specific state.
        self.dual_smell = DualSmellField(width, height)
        # For backwards compat with code that reads self.smell.
        self.smell = self.dual_smell.field_a
        self.food_a: list[FoodA] = []
        self.food_b: list[FoodB] = []
        # Clear legacy self.food (kept as empty list to satisfy
        # metrics code that may iterate it).
        self.food = []
        # Warm-start genome (used by _spawn_founder_override below).
        self._warm_genome = warm_genome
        # Season state.
        self.season: int = 0
        self.ticks_in_season: int = 0
        # Phase 5 grace period: ticks remaining with soft negative
        # penalty after a season flip. Set to PHASE3_GRACE_TICKS on
        # every season change, decremented each tick. 0 disables the
        # mechanism entirely (Phase 3/4 backward compat).
        self.grace_ticks_remaining: int = 0
        # Phase 3 reproduction threshold (separate from the global one
        # used by the legacy World).
        self._reproduction_threshold = PHASE3_REPRODUCTION_THRESHOLD
        # Spawn founders + initial food.
        self._populate_initial()

    def _reproduce(self) -> None:
        """Phase 3 reproduction: same logic as World._reproduce, but
        uses PHASE3_REPRODUCTION_THRESHOLD so the founder's first wave
        doesn't overshoot the food budget.
        """
        if len(self.organisms) >= POPULATION_CAP:
            return
        new_organisms: list[Organism] = []
        for org in self.organisms:
            if not org.alive:
                continue
            if org.energy < self._reproduction_threshold:
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
            # Phase 4.1: parent commits its lifetime trace before the
            # child starts its own episode. The child inherits the
            # parent's post-commit weights (already mirrored to its
            # genome by commit_episode) and starts fresh.
            if org.brain is not None:
                org.brain.commit_episode()
            if child.brain is not None:
                child.brain.begin_episode()
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

    def _populate_initial(self) -> None:
        """Phase 3 founder population + dual-resource food spawn.

        Uses Phase 3 specific demographic constants: PHASE3_FOOD_TARGET
        (more food than Phase 1.5 so the founder population survives
        its blind-eats while waiting for evolution to grow recurrence),
        and the standard INITIAL_POPULATION.
        """
        for _ in range(INITIAL_POPULATION):
            self._spawn_founder()
        self.species_manager.sync(self.organisms, self.tick)

        n_a = int(PHASE3_FOOD_TARGET * FOOD_A_FRACTION)
        n_b = PHASE3_FOOD_TARGET - n_a
        for _ in range(n_a):
            self._spawn_food_a()
        for _ in range(n_b):
            self._spawn_food_b()

    def _spawn_founder(self) -> None:
        """Phase 3 founder with 6 distinct A/B smell sensors."""
        genome = make_phase3_founder_from_warmstart(
            rng=self.rng,
            warm_genome=self._warm_genome,
            visible_season_sensor=self.visible_season_sensor,
        )
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
            energy=PHASE3_INITIAL_ENERGY,
            brain=Brain(genome),
            genome=genome,
        )
        # Phase 4.1: each new lifetime starts with a fresh eligibility
        # trace and zeroed episode reward accumulator. Reward baseline
        # is preserved across episodes for long-lived organisms.
        if org.brain is not None:
            org.brain.begin_episode()
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
        """Advance one tick. Phase 3 dual-resource + optional season."""
        self.tick += 1
        self.tick_births = 0
        self.tick_mutations = 0

        # 0. Season flip if enabled.
        if self.season_enabled and self.ticks_in_season >= SEASON_LENGTH:
            self.season = 1 - self.season
            self.ticks_in_season = 0
            # Phase 5: open a grace window where negative food is
            # soft-penalised so the brain can adapt to the new sign
            # of reward without immediate mass starvation.
            if PHASE3_GRACE_TICKS > 0:
                self.grace_ticks_remaining = PHASE3_GRACE_TICKS
        if self.season_enabled:
            self.ticks_in_season += 1
            if self.grace_ticks_remaining > 0:
                self.grace_ticks_remaining -= 1

        # 1. Regrow food per resource.
        a_target = int(PHASE3_FOOD_TARGET * FOOD_A_FRACTION)
        b_target = PHASE3_FOOD_TARGET - a_target
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

        # 3. Decay intake feedback.
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

        # 5. Resolve eat (signed reward, feedback).
        self._resolve_eat_phase3()

        # 5b. Phase 4 lifetime synaptic plasticity. Apply reward-
        # modulated Hebbian updates to each organism's brain using the
        # most recent forward-pass activations and the current
        # intake_feedback signal. No-op when PLASTICITY_ALPHA and
        # PLASTICITY_BETA are both 0 (Phase 3 default).
        for org in self.organisms:
            if not org.alive or org.brain is None:
                continue
            org.brain.apply_plasticity(float(org.intake_feedback))

        # 6. Drain + death.
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
                # Phase 4.1: end-of-life commit. The lifetime trace
                # shapes weights that won't be inherited (organism is
                # dead) but this matters for population-level reward
                # baseline statistics and any cross-generation memory
                # tests. Mirrors the reproduction-side commit semantics.
                if org.brain is not None:
                    org.brain.commit_episode()
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
        """7-vector (or 8 with season sensor).

        Order:
          [a_left, a_front, a_right,
           b_left, b_front, b_right,
           intake_feedback,
           (optional) season in {0, 1}]
        """
        smell = _smell_channel(self, org)
        fb = float(org.intake_feedback) if org.intake_feedback_ttl > 0 else 0.0
        if self.visible_season_sensor:
            arr = np.empty(N_SMELL_A + N_SMELL_B + 2, dtype=np.float32)
            arr[:N_SMELL_A + N_SMELL_B] = smell
            arr[FEEDBACK_INDEX] = fb
            arr[SEASON_INDEX] = float(self.season)
        else:
            arr = np.empty(N_SMELL_A + N_SMELL_B + 1, dtype=np.float32)
            arr[:N_SMELL_A + N_SMELL_B] = smell
            arr[FEEDBACK_INDEX] = fb
        return arr

    def _resolve_eat_phase3(self) -> None:
        """Override _resolve_eat to use season-dependent rewards + feedback.

        In static_dual mode both food types are always positive: there
        is no season and no resource to avoid. Only in visible_season
        and hidden_season does the sign of the reward flip with season.

        One organism can eat at most one particle per tick, even
        when both food_a and food_b are within EAT_RADIUS. This keeps
        energy budgets comparable to Phase 1.5.
        """
        for org in self.organisms:
            if not org.alive:
                continue
            # Pick the single nearest particle across both resources.
            best_dist = float("inf")
            best_target: tuple | None = None
            for food_list, is_a in ((self.food_a, True), (self.food_b, False)):
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
                d = float(dist[idx])
                if d <= EAT_RADIUS and d < best_dist:
                    best_dist = d
                    best_target = (food_list, is_a, idx)
            if best_target is None:
                continue
            food_list, is_a, idx = best_target
            eaten = food_list.pop(idx)
            if not self.season_enabled:
                # static_dual: both resources are always +FOOD_*_POSITIVE_ENERGY.
                reward = (
                    FOOD_A_POSITIVE_ENERGY
                    if is_a
                    else FOOD_B_POSITIVE_ENERGY
                )
            elif is_a:
                reward = (
                    FOOD_A_POSITIVE_ENERGY
                    if self.season == 0
                    else PHASE3_FOOD_A_NEGATIVE_ENERGY
                )
            else:
                reward = (
                    FOOD_B_POSITIVE_ENERGY
                    if self.season == 1
                    else PHASE3_FOOD_B_NEGATIVE_ENERGY
                )
            # Phase 5: grace period after season flip. While in the
            # grace window, negative food gives a softened penalty
            # that interpolates linearly from PHASE3_GRACE_PENALTY
            # (at flip) back to the full negative_energy (at the end
            # of the window). Positive food is unchanged.
            if (
                reward < 0
                and PHASE3_GRACE_TICKS > 0
                and self.grace_ticks_remaining > 0
            ):
                # 1.0 immediately after flip, 0.0 at end of window.
                t = self.grace_ticks_remaining / PHASE3_GRACE_TICKS
                reward = PHASE3_GRACE_PENALTY * t + reward * (1.0 - t)
            org.energy += reward
            org.food_eaten += 1
            if org.time_to_first_food is None:
                org.time_to_first_food = org.age
            if reward > 0:
                org.intake_feedback = +1.0
            else:
                org.intake_feedback = -1.0
            org.intake_feedback_ttl = INTAKE_FEEDBACK_DURATION
            org.last_intake_feedback = org.intake_feedback
            if reward > 0:
                org.positive_eats = getattr(org, "positive_eats", 0) + 1
            else:
                org.negative_eats = getattr(org, "negative_eats", 0) + 1
            # Phase 4.1: feed the *signed* reward into the lifetime
            # episode accumulator. The actual plastic update is committed
            # at episode boundaries (starvation or reproduction), not
            # per-tick. sign(reward) keeps the magnitude small so a
            # single bad bite doesn't dominate the trace commit.
            if org.brain is not None:
                org.brain.accumulate_episode_reward(
                    1.0 if reward > 0 else -1.0
                )
            self.events.record_eat(
                self.tick, org_id=org.id,
                x=eaten.x, y=eaten.y,
            )
