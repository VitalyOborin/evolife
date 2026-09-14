"""Behavioral Arena — passive benchmark for frozen genomes.

The arena is a controlled evaluation environment: a single frozen
genome (no mutation, no reproduction) is dropped into a world whose
food layout is fully specified by a `Scenario`. We measure its
behavior across multiple seeds and aggregate four indicator families:

  Efficiency     : food_rate, distance_per_food, time_to_first_food,
                   energy_efficiency (eaten_energy / energy_spent).
  Reactivity     : steering_alignment, mean_abs_turn.
  Conditional    : movement_transitions, mean_rest_bout, mean_move_bout.
  Stateful       : state_dependence (from behavior.descriptors).

Crucially, the arena NEVER affects living evolution. It only reads
genomes — never writes them back. There is no fitness function, no
selection, no hidden scoring. We surface the four indicator families
side by side so a human (or a downstream visualisation) can judge
whether evolution is producing real behavioral improvement.

Scenarios (A–G):
  A. uniform       : same as living world.
  B. ahead         : food cluster placed in front of the organism.
  C. behind        : food cluster placed behind.
  D. left_right    : two clusters, one to the left and one to the right.
  E. sparse        : very low food density (FOOD_TARGET // 4).
  E'. dense        : very high food density (FOOD_TARGET * 2).
  F. relocating    : food cluster teleports to a new location every
                     RELOCATE_EVERY ticks; tests continual re-acquisition.
  G. smell_blanked : smell signal is silenced for a contiguous 50-tick
                     window starting at SMELL_BLANK_START; tests whether
                     a feed-forward reflex collapses (it should) and a
                     memory-bearing policy holds trajectory.

Replay (scripts/run_arena_replay.py) reuses the same plumbing to
spawn two arenas side by side (ancestor + descendant) with the same
seed and visualise the trajectories.

The arena world is a subclass of World that disables reproduction and
auto-spawning; it shares sensors, brain, lifecycle, and smell logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .behavior import descriptors
from .config import (
    FOOD_RADIUS,
    FOOD_REGROWTH_RATE,
    FOOD_TARGET,
    INITIAL_ENERGY,
    INITIAL_POPULATION,
    MAX_AGE,
    N_HIDDEN,
    N_MOTORS,
    N_SENSORS,
    POPULATION_CAP,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from .genome import Genome
from .organism import Organism
from .sensors import SmellField
from .world import Food, World


# Arena-specific defaults — distinct from living-world values so a
# benchmark stays reproducible.
ARENA_FOOD_TARGET = FOOD_TARGET
ARENA_INITIAL_ENERGY = INITIAL_ENERGY
ARENA_MAX_AGE = MAX_AGE
ARENA_N_TICKS_DEFAULT = 1000
ARENA_N_SEEDS_DEFAULT = 20

# Scenario F — relocate every N ticks.
RELOCATE_EVERY = 200

# Scenario G — silence smell for 50 ticks starting at tick SMELL_BLANK_START.
SMELL_BLANK_START = 300
SMELL_BLANK_DURATION = 50


@dataclass
class ArenaResult:
    """Aggregate behavioral result from one (genome, scenario, seed) run.

    Times and counts are integer ticks; the rest are floats.
    """

    scenario: str
    seed: int
    food_eaten: int
    ticks_to_first_food: int | None
    ticks_alive: int
    final_energy: float
    distance_traveled: float
    mean_speed: float
    mean_speed_while_moving: float
    moving_fraction: float
    movement_transitions: int
    mean_rest_bout: float
    mean_move_bout: float
    longest_rest: int
    longest_move: int
    mean_abs_turn: float
    turns_per_distance: float
    exploration_rate: float
    steering_alignment: float
    state_dependence: float
    bias_over_weights: float
    energy_efficiency: float  # food_eaten * FOOD_ENERGY / (energy_spent+eps)


@dataclass
class ScenarioSpec:
    """One concrete test scenario.

    A scenario is parameterised by a `seed` and is responsible for
    placing food on the arena before the run starts (and, for
    `relocating`, every RELOCATE_EVERY ticks). The same seed always
    yields the same food layout for a given scenario kind.
    """

    kind: str
    seed: int
    food: list[Food] = field(default_factory=list)
    width: int = WORLD_WIDTH
    height: int = WORLD_HEIGHT


# ---------------------------------------------------------------------------
# Scenario constructors.
# ---------------------------------------------------------------------------


def uniform_scenario(seed: int) -> ScenarioSpec:
    """A. Uniform food at FOOD_TARGET density — the living-world baseline."""
    rng = np.random.default_rng(seed)
    spec = ScenarioSpec(kind="uniform", seed=seed)
    for _ in range(ARENA_FOOD_TARGET):
        spec.food.append(
            Food(
                x=float(rng.uniform(0, spec.width)),
                y=float(rng.uniform(0, spec.height)),
            )
        )
    return spec


def ahead_scenario(seed: int) -> ScenarioSpec:
    """B. Food clustered in front of the starting organism (heading=0)."""
    spec = ScenarioSpec(kind="ahead", seed=seed)
    rng = np.random.default_rng(seed)
    cx, cy = spec.width / 2, spec.height / 2
    for _ in range(ARENA_FOOD_TARGET):
        # Place ahead of (cx, cy): dx in [40, 200], dy in [-80, 80].
        dx = float(rng.uniform(40, 200))
        dy = float(rng.uniform(-80, 80))
        spec.food.append(Food(x=(cx + dx) % spec.width, y=(cy + dy) % spec.height))
    return spec


def behind_scenario(seed: int) -> ScenarioSpec:
    """C. Food clustered behind the starting organism."""
    spec = ScenarioSpec(kind="behind", seed=seed)
    rng = np.random.default_rng(seed)
    cx, cy = spec.width / 2, spec.height / 2
    for _ in range(ARENA_FOOD_TARGET):
        dx = float(rng.uniform(-200, -40))
        dy = float(rng.uniform(-80, 80))
        spec.food.append(Food(x=(cx + dx) % spec.width, y=(cy + dy) % spec.height))
    return spec


def left_right_scenario(seed: int) -> ScenarioSpec:
    """D. Two food clusters, one to the left and one to the right."""
    spec = ScenarioSpec(kind="left_right", seed=seed)
    rng = np.random.default_rng(seed)
    cy = spec.height / 2
    half = ARENA_FOOD_TARGET // 2
    for _ in range(half):
        dx = float(rng.uniform(60, 180))
        dy = float(rng.uniform(-60, 60))
        spec.food.append(Food(x=(spec.width / 2 + dx) % spec.width, y=(cy + dy) % spec.height))
    for _ in range(ARENA_FOOD_TARGET - half):
        dx = float(rng.uniform(-180, -60))
        dy = float(rng.uniform(-60, 60))
        spec.food.append(Food(x=(spec.width / 2 + dx) % spec.width, y=(cy + dy) % spec.height))
    return spec


def sparse_scenario(seed: int) -> ScenarioSpec:
    """E. Sparse food — quarter of the baseline density."""
    spec = ScenarioSpec(kind="sparse", seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(max(1, ARENA_FOOD_TARGET // 4)):
        spec.food.append(
            Food(
                x=float(rng.uniform(0, spec.width)),
                y=float(rng.uniform(0, spec.height)),
            )
        )
    return spec


def dense_scenario(seed: int) -> ScenarioSpec:
    """E'. Dense food — double the baseline density."""
    spec = ScenarioSpec(kind="dense", seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(ARENA_FOOD_TARGET * 2):
        spec.food.append(
            Food(
                x=float(rng.uniform(0, spec.width)),
                y=float(rng.uniform(0, spec.height)),
            )
        )
    return spec


def relocating_scenario(seed: int) -> ScenarioSpec:
    """F. Food cluster relocates every RELOCATE_EVERY ticks.

    Initial cluster is placed ahead (heading=0). Each subsequent
    cluster is placed in a uniformly random direction at distance
    100..200 from the world centre.
    """
    spec = ScenarioSpec(kind="relocating", seed=seed)
    _populate_relocating(spec, seed)
    return spec


def _populate_relocating(spec: ScenarioSpec, episode_seed: int) -> None:
    """Place a single dense cluster for the relocating scenario.

    Cluster shape: ~ARENA_FOOD_TARGET food items in a 60x60 square
    around a centre placed 100..200 ahead/behind/left/right of the
    arena centre. Episode_seed picks the centre deterministically.
    """
    spec.food.clear()
    rng = np.random.default_rng(episode_seed ^ 0x5151)
    angle = float(rng.uniform(0, 2 * math.pi))
    distance = float(rng.uniform(100, 200))
    cx = (spec.width / 2 + math.cos(angle) * distance) % spec.width
    cy = (spec.height / 2 + math.sin(angle) * distance) % spec.height
    for _ in range(ARENA_FOOD_TARGET):
        spec.food.append(
            Food(
                x=float((cx + rng.uniform(-30, 30)) % spec.width),
                y=float((cy + rng.uniform(-30, 30)) % spec.height),
            )
        )


def smell_blanked_scenario(seed: int) -> ScenarioSpec:
    """G. Same as uniform but smell is silenced in a window of the run."""
    spec = uniform_scenario(seed)
    spec.kind = "smell_blanked"
    return spec


SCENARIO_BUILDERS: dict[str, Callable[[int], ScenarioSpec]] = {
    "uniform": uniform_scenario,
    "ahead": ahead_scenario,
    "behind": behind_scenario,
    "left_right": left_right_scenario,
    "sparse": sparse_scenario,
    "dense": dense_scenario,
    "relocating": relocating_scenario,
    "smell_blanked": smell_blanked_scenario,
}


# ---------------------------------------------------------------------------
# ArenaWorld — World subclass that disables reproduction and uses a
# scenario's food layout.
# ---------------------------------------------------------------------------


class ArenaWorld(World):
    """World subclass that hosts a single frozen genome in a Scenario.

    - No reproduction: `_reproduce` is a no-op.
    - No auto-spawning of founders: the caller provides `frozen_genome`
      and `frozen_position` (start position and heading).
    - Food layout is taken from the Scenario; the `relocating` scenario
      refills food every RELOCATE_EVERY ticks.
    - For `smell_blanked`, we zero out the smell field in the
      SMELL_BLANK_START..SMELL_BLANK_START+SMELL_BLANK_DURATION window.
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        frozen_genome: Genome,
        start_position: tuple[float, float] | None = None,
        start_heading: float = 0.0,
    ) -> None:
        # Skip World.__init__ — it auto-spawns founders. We do a manual
        # minimal init to get the smell field and RNG.
        self.width = scenario.width
        self.height = scenario.height
        self.rng = np.random.default_rng(scenario.seed)
        # Avoid the heavy SpeciesManager/Archive/EventLog machinery; the
        # arena only cares about behavior descriptors and final stats.
        from .archive import Archive
        from .events import EventLog
        from .innovation import InnovationDatabase
        from .speciation import SpeciesManager

        self.innovations = InnovationDatabase()
        self.species_manager = SpeciesManager()
        self.events = EventLog()
        self.archive = Archive()
        self.tick = 0
        self.organisms = []
        self.food = list(scenario.food)
        self.smell = SmellField(self.width, self.height)
        self._next_id = 0

        # Spawn exactly one frozen organism.
        from .brain import Brain

        brain = Brain(frozen_genome)
        pos = start_position or (
            self.width / 2,
            self.height / 2,
        )
        oid = self._next_organism_id()
        org = Organism(
            id=oid,
            x=pos[0],
            y=pos[1],
            heading=start_heading,
            energy=ARENA_INITIAL_ENERGY,
            brain=brain,
            genome=frozen_genome,
            generation=0,
            founder_lineage_id=0,
        )
        self.organisms.append(org)

    # Override reproduction — arena never reproduces.
    def _reproduce(self) -> None:  # type: ignore[override]
        return

    # Relocating scenario refills food periodically.
    def _maybe_relocate_food(self, scenario: ScenarioSpec) -> None:
        if scenario.kind != "relocating":
            return
        if self.tick == 0 or (self.tick % RELOCATE_EVERY) != 0:
            return
        # Use a seed derived from tick so each relocation is
        # deterministic but varies.
        _populate_relocating(scenario, scenario.seed + self.tick)

    # Smell blanking scenario zeroes out smell in a window.
    def _maybe_blank_smell(self, scenario: ScenarioSpec) -> None:
        if scenario.kind != "smell_blanked":
            return
        if (
            SMELL_BLANK_START
            <= self.tick
            < SMELL_BLANK_START + SMELL_BLANK_DURATION
        ):
            # Replace the smell field with zeros for this window.
            self.smell.grid.fill(0.0)

    def step(self) -> None:  # type: ignore[override]
        """Run one tick: refills food, recomputes smell, acts, eats,
        applies deaths. NO reproduction. NO auto-founders."""
        self.tick += 1
        # Regrow food at the baseline rate so sparse/dense scenarios
        # stay balanced over long runs.
        target = self._scenario_target_food()
        missing = target - len(self.food)
        if missing > 0:
            n_new = int(self.rng.binomial(missing, FOOD_REGROWTH_RATE))
            for _ in range(n_new):
                self.food.append(
                    Food(
                        x=float(self.rng.uniform(0, self.width)),
                        y=float(self.rng.uniform(0, self.height)),
                    )
                )
        # Relocating scenarios re-seed food layout every RELOCATE_EVERY
        # ticks — handled by step caller via _maybe_relocate_food.

        # Recompute smell from current food, then maybe blank it.
        self.smell.recompute(self.food)
        # The blanking hook needs the scenario; we stash it as an
        # attribute when the arena world is created.
        scenario = getattr(self, "_scenario", None)
        if scenario is not None:
            self._maybe_blank_smell(scenario)
        for org in self.organisms:
            if not org.alive:
                continue
            self._act(org)
        self._resolve_eat()
        # Death is handled inline in World.step; arena's overridden
        # step() inlines the same logic below.
        from .config import (
            CONNECTION_METABOLIC_COST,
            IDLE_ENERGY_COST,
            MAX_AGE,
            NEURON_METABOLIC_COST,
        )

        survivors = []
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
                org.alive = False
                continue
            org.peak_energy = max(org.peak_energy, org.energy)
            survivors.append(org)
        self.organisms = survivors

    def _scenario_target_food(self) -> int:
        scenario = getattr(self, "_scenario", None)
        if scenario is None:
            return ARENA_FOOD_TARGET
        if scenario.kind == "sparse":
            return max(1, ARENA_FOOD_TARGET // 4)
        if scenario.kind == "dense":
            return ARENA_FOOD_TARGET * 2
        if scenario.kind == "relocating":
            return ARENA_FOOD_TARGET
        return ARENA_FOOD_TARGET


# ---------------------------------------------------------------------------
# Run a single arena episode.
# ---------------------------------------------------------------------------


def _bias_over_weights(genome: Genome) -> float:
    """Re-export the private helper from metrics.py for arena use."""
    from .metrics import _locomotion_bias_over_weights

    return _locomotion_bias_over_weights(genome)


def run_episode(
    genome: Genome,
    scenario_kind: str,
    seed: int,
    n_ticks: int = ARENA_N_TICKS_DEFAULT,
    start_position: tuple[float, float] | None = None,
    start_heading: float = 0.0,
) -> ArenaResult:
    """Run one (genome, scenario, seed) episode and aggregate metrics.

    Returns an ArenaResult. If the organism dies before n_ticks, the
    result reports partial statistics over the ticks it actually lived.
    """
    if scenario_kind not in SCENARIO_BUILDERS:
        raise ValueError(f"unknown scenario: {scenario_kind}")
    spec = SCENARIO_BUILDERS[scenario_kind](seed)
    arena = ArenaWorld(spec, genome, start_position=start_position, start_heading=start_heading)
    arena._scenario = spec  # stash for hooks.

    food_eaten = 0
    energy_spent = 0.0
    for _ in range(n_ticks):
        if not arena.organisms or not arena.organisms[0].alive:
            break
        e_before = arena.organisms[0].energy
        arena.step()
        # For relocating scenarios, refit food after the tick.
        arena._maybe_relocate_food(spec)
        org = arena.organisms[0]
        energy_spent += max(0.0, e_before - org.energy)
        food_eaten = org.food_eaten

    org = arena.organisms[0] if arena.organisms else None
    if org is None or not org.alive:
        # Organism died or never existed.
        return ArenaResult(
            scenario=scenario_kind,
            seed=seed,
            food_eaten=0,
            ticks_to_first_food=None,
            ticks_alive=0,
            final_energy=0.0,
            distance_traveled=0.0,
            mean_speed=0.0,
            mean_speed_while_moving=0.0,
            moving_fraction=0.0,
            movement_transitions=0,
            mean_rest_bout=0.0,
            mean_move_bout=0.0,
            longest_rest=0,
            longest_move=0,
            mean_abs_turn=0.0,
            turns_per_distance=0.0,
            exploration_rate=0.0,
            steering_alignment=0.0,
            state_dependence=0.0,
            bias_over_weights=0.0,
            energy_efficiency=0.0,
        )
    d = descriptors(org)
    from .config import FOOD_ENERGY

    return ArenaResult(
        scenario=scenario_kind,
        seed=seed,
        food_eaten=org.food_eaten,
        ticks_to_first_food=int(d["time_to_first_food"]) if d["time_to_first_food"] is not None else None,
        ticks_alive=org.age,
        final_energy=org.energy,
        distance_traveled=org.distance_sum,
        mean_speed=d["mean_speed"],
        mean_speed_while_moving=d["mean_speed_while_moving"],
        moving_fraction=d["moving_fraction"],
        movement_transitions=org.movement_transitions,
        mean_rest_bout=d["mean_rest_bout"],
        mean_move_bout=d["mean_move_bout"],
        longest_rest=int(d["longest_rest"]),
        longest_move=int(d["longest_move"]),
        mean_abs_turn=d["mean_abs_turn"],
        turns_per_distance=d["turns_per_distance"],
        exploration_rate=d["exploration_rate"],
        steering_alignment=d["steering_alignment"],
        state_dependence=d["state_dependence"],
        bias_over_weights=_bias_over_weights(genome),
        energy_efficiency=(
            (org.food_eaten * FOOD_ENERGY) / max(energy_spent, 1e-6)
        ),
    )


def aggregate(results: list[ArenaResult]) -> dict[str, float]:
    """Aggregate a list of per-seed ArenaResults into mean/std per metric."""
    import statistics

    if not results:
        return {}
    keys = [
        "food_eaten",
        "ticks_to_first_food",
        "ticks_alive",
        "final_energy",
        "distance_traveled",
        "mean_speed",
        "mean_speed_while_moving",
        "moving_fraction",
        "movement_transitions",
        "mean_rest_bout",
        "mean_move_bout",
        "longest_rest",
        "longest_move",
        "mean_abs_turn",
        "turns_per_distance",
        "exploration_rate",
        "steering_alignment",
        "state_dependence",
        "bias_over_weights",
        "energy_efficiency",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [getattr(r, k) for r in results if getattr(r, k) is not None]
        if not vals:
            out[f"{k}_mean"] = 0.0
            out[f"{k}_std"] = 0.0
            continue
        out[f"{k}_mean"] = float(statistics.mean(vals))
        out[f"{k}_std"] = (
            float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        )
    return out


# ---------------------------------------------------------------------------
# Genome sampling by generation.
# ---------------------------------------------------------------------------


def sample_genomes_by_generation(
    world: World,
    generations: list[int],
    *,
    prefer: str = "oldest",
) -> dict[int, Genome]:
    """Pick one genome per requested generation from a living world.

    For each requested `g`, find the oldest living organism whose
    `generation == g`. If none, find the closest generation <= g.
    Returns a dict {generation: genome}. Skips generations that have
    no living representatives.
    """
    by_gen: dict[int, list[Organism]] = {}
    for org in world.organisms:
        if not org.alive or org.genome is None:
            continue
        by_gen.setdefault(org.generation, []).append(org)
    out: dict[int, Genome] = {}
    for g in generations:
        if g in by_gen and by_gen[g]:
            out[g] = by_gen[g][0].genome
            continue
        # Find nearest lower generation with a living organism.
        lower = [gg for gg in by_gen if gg <= g and by_gen[gg]]
        if lower:
            nearest = max(lower)
            out[g] = by_gen[nearest][0].genome
    return out


# ---------------------------------------------------------------------------
# Brain-shape descriptor.
# ---------------------------------------------------------------------------


def brain_shape(genome: Genome) -> tuple[int, int]:
    """Return (n_nodes, n_connections) for a genome."""
    return (
        N_SENSORS + N_HIDDEN + N_MOTORS,
        len(genome.connections),
    )
