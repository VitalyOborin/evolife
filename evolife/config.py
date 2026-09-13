"""All EvoLife v0 constants in one place.

Anything tunable lives here. Modules import names; they never inline numbers.
This is the single point to evolve when v0 graduates to v1 (real evolution,
topology mutation enabled, speciation active, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass


# --- World -----------------------------------------------------------------

WORLD_WIDTH: int = 512
WORLD_HEIGHT: int = 512

# How many food particles exist at any time. World respawns food up to this
# cap after each tick if any was eaten.
FOOD_TARGET: int = 400

# Probability per cell per tick that a new food particle appears at random.
# We use sparse spawning on top of FOOD_TARGET for a soft trickle even when
# the field is empty.
FOOD_SPAWN_RATE: float = 0.02

# Energy gained from eating one food particle.
FOOD_ENERGY: float = 25.0

# Energy passively drained per tick just for being alive.
IDLE_ENERGY_COST: float = 0.05

# Energy cost per unit of forward motion per tick.
MOVE_ENERGY_COST: float = 0.10

# Energy cost per radian of turning per tick.
TURN_ENERGY_COST: float = 0.02


# --- Organism --------------------------------------------------------------

# Maximum number of simultaneous organisms in v0.
# Soft cap: world tries to keep around this many. When over, no reproduction.
POPULATION_CAP: int = 500

# Initial organism count when the world boots.
INITIAL_POPULATION: int = 200

# Energy a newborn organism starts with. Equal to FOOD_ENERGY so a fresh
# organism has one meal of buffer against starvation.
INITIAL_ENERGY: float = FOOD_ENERGY

# Energy threshold above which an organism may reproduce. Energy is split
# with the offspring, so both parent and child get REPRODUCTION_ENERGY.
REPRODUCTION_THRESHOLD: float = 60.0

# Energy transferred to each side (parent + child) on reproduction.
REPRODUCTION_ENERGY: float = 30.0

# Maximum age in ticks. Pure safety net to prevent immortal lineages from
# monopolising the world even if their energy stays positive forever.
MAX_AGE: int = 50_000


# --- Brain -----------------------------------------------------------------

# Sensor layout for v0. Keep them raw and cheap; we can add rays later.
# Order matters: brain input vector is laid out in this exact order.
# Index 0: food_angle in [-1, 1] (sin of relative angle, normalised).
# Index 1: food_distance in [0, 1] (1 at zero distance, ~0 at WORLD_WIDTH).
# Index 2: own_energy in [0, 1] (energy / REPRODUCTION_THRESHOLD, clipped).
# Index 3: own_speed in [0, 1] (linear speed / MAX_LINEAR_SPEED).
# Index 4: bias (constant 1.0, like a Perceptron bias neuron).
N_SENSORS: int = 5

# Motor outputs:
# Index 0: turn_rate in [-1, 1] -> multiplied by MAX_TURN_RATE.
# Index 1: move_speed in [0, 1] -> multiplied by MAX_LINEAR_SPEED.
# Index 2: eat_attempt in [0, 1]. > 0.5 means try to eat whatever is at
#          the front of the organism (within EAT_RADIUS).
# Index 3: reproduce_attempt in [0, 1]. > 0.5 means try to reproduce if
#          energy threshold is met and POPULATION_CAP is not exceeded.
N_MOTORS: int = 4

# Hidden layer size in the v0 fixed-topology brain. v1 will read this from
# the genome.
N_HIDDEN: int = 4

# Maximum forward speed in cells per tick.
MAX_LINEAR_SPEED: float = 2.0

# Maximum turn rate in radians per tick.
MAX_TURN_RATE: float = 0.3

# Radius within which an eat_attempt consumes the nearest food particle.
EAT_RADIUS: float = 4.0

# Radius within which organisms cannot overlap (soft collision: they push
# each other apart; this is a v0 hack, real collision response comes later).
COLLISION_RADIUS: float = 3.0


# --- Mutation --------------------------------------------------------------

# Per-birth probability of mutating weights.
WEIGHT_MUTATION_RATE: float = 1.0  # v0: always mutate weights.

# Probability per weight of being perturbed by gaussian noise.
WEIGHT_PERTURB_RATE: float = 0.9

# Standard deviation of gaussian weight perturbation.
WEIGHT_PERTURB_SIGMA: float = 0.5

# Hard clamp on absolute weight magnitude, post-mutation.
WEIGHT_MAX: float = 5.0

# Per-birth probability of structural mutations. v0 keeps these at 0 by
# default; flip them on when we graduate to v1.
ADD_NODE_RATE: float = 0.0
ADD_CONNECTION_RATE: float = 0.0
TOGGLE_CONNECTION_RATE: float = 0.0


# --- Visualisation ---------------------------------------------------------

PIXELS_PER_CELL: int = 1  # logical units == pixels at 1x.
FPS_TARGET: int = 60

# Organism body radius in pixels.
ORGANISM_RADIUS: int = 3

# Food particle radius in pixels.
FOOD_RADIUS: int = 2


# --- Metrics ---------------------------------------------------------------

# Ticks between per-organism snapshots (lifetime, energy, descendants).
METRICS_ORGANISM_EVERY: int = 100

# Ticks between per-world snapshots (population, mean energy, complexity).
METRICS_WORLD_EVERY: int = 50

# SQLite file path for metrics. Relative to caller working directory unless
# absolute.
METRICS_DB_PATH: str = "evolife_metrics.sqlite"


@dataclass(frozen=True)
class V0Summary:
    """A read-only summary of v0 constants. Useful for logging and tests."""

    n_sensors: int = N_SENSORS
    n_hidden: int = N_HIDDEN
    n_motors: int = N_MOTORS
    population_cap: int = POPULATION_CAP
    initial_population: int = INITIAL_POPULATION
    world_size: tuple[int, int] = (WORLD_WIDTH, WORLD_HEIGHT)
