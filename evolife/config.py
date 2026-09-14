"""All EvoLife constants in one place.

Anything tunable lives here. Modules import names; they never inline numbers.
v2.2 (Viable Replicator) is a bootstrap stage: a proto-brain that can
persist for tens of generations so natural selection has time to act.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- World -----------------------------------------------------------------

WORLD_WIDTH: int = 512
WORLD_HEIGHT: int = 512

# --- Phase 3: world resource mode ---
# "single" = legacy Phase 1.5 world (one resource, uniform smell, FOOD_ENERGY).
# "two_resources" = two distinct food types (FoodA / FoodB) with season-dependent
#                   reward sign, designed to make feedforward reflex insufficient.
WORLD_RESOURCE_MODE: str = "two_resources"

# Total food particles across both resources when mode = "two_resources".
# (Equivalent to FOOD_TARGET in single mode.)
FOOD_TARGET: int = 200

# How many of the FOOD_TARGET go to FoodA vs FoodB on initial spawn / respawn.
# Roughly balanced so a founder does not see one resource disproportionately.
FOOD_A_FRACTION: float = 0.5

# Per-tick binomial probability that each missing food particle respawns.
# Expected new food = (FOOD_TARGET - current) * FOOD_REGROWTH_RATE.
# At half-stock (200/400) this is ~0.4 food/tick; near full, ~0.01/tick.
FOOD_REGROWTH_RATE: float = 0.002

# Energy gained from eating one food particle.
FOOD_ENERGY: float = 40.0

# Phase 3: edge weight from the feedback sensor into the (single) hidden
# node. Held constant at 1.0 in v3.0; planned to become a heritable gene
# in v3.1. The feedback sensor carries the recent intake signal
# (intake_feedback, in {-1, 0, +1}).
FB_TO_HIDDEN_WEIGHT: float = 0.2

# Phase 3.1 world mode. One of:
#   "static_dual"      : two food types, no season (control)
#   "visible_season"   : two food types, season changes, season visible
#                        via a dedicated sensor (control)
#   "hidden_season"    : two food types, season changes, no sensor for
#                        season, only post-eat feedback (the actual test)
PHASE3_DEFAULT_MODE: str = "hidden_season"

# Whether to expose season as an extra dedicated sensor. Only used by
# "visible_season" mode. Hidden-season tests must keep this False.
PHASE3_VISIBLE_SEASON_SENSOR: bool = False

# Phase 3: signed reward magnitudes. Eating FoodA in season 0 gives +25; in
# season 1 the *same* smell yields -3. Energy delta is what the organism
# experiences — it cannot sense season directly, only via intake feedback.
# The negative penalty is large enough to discourage random feeding but
# small enough that founders can survive an early mistake. Phase 3.0 is
# cold-start: founders have to learn a 7-sensor navigation reflex *and*
# a season-dependent selection rule from scratch, so the net reward
# over many random eats must stay positive.
FOOD_A_POSITIVE_ENERGY: float = 25.0
FOOD_A_NEGATIVE_ENERGY: float = -3.0
FOOD_B_POSITIVE_ENERGY: float = 25.0
FOOD_B_NEGATIVE_ENERGY: float = -3.0

# Phase 3: how long a season lasts in ticks before flipping sign of rewards.
SEASON_LENGTH: int = 2000

# Phase 3: how many ticks after `eat` the intake_feedback signal remains
# non-zero in the organism. 1 means it disappears next tick; 2 means it
# lingers. Short window is closer to CANON "single-tick intake signal".
INTAKE_FEEDBACK_DURATION: int = 2

# Energy passively drained per tick just for being alive.
IDLE_ENERGY_COST: float = 0.02

# Energy cost per unit of forward motion per tick.
MOVE_ENERGY_COST: float = 0.05

# Energy cost per radian of turning per tick.
TURN_ENERGY_COST: float = 0.05


# --- Organism --------------------------------------------------------------

# Maximum number of simultaneous organisms in v0.
# Soft cap: world tries to keep around this many. When over, no reproduction.
POPULATION_CAP: int = 500

# Initial organism count when the world boots.
INITIAL_POPULATION: int = 50

# Energy a newborn organism starts with. Equal to FOOD_ENERGY so a fresh
# organism has one meal of buffer against starvation.
INITIAL_ENERGY: float = FOOD_ENERGY*2

# Phase 3 founder buffer. Founders must survive several wrong eats while
# learning the season/feedback structure. Used by MemoryEcologyWorld
# instead of the global INITIAL_ENERGY.
PHASE3_INITIAL_ENERGY: float = FOOD_ENERGY * 2  # matched to Phase 1.5

# Phase 3 reproduction threshold: raise above the Phase 1.5 default so the
# initial over-reproduction wave doesn't overshoot the food budget.
PHASE3_REPRODUCTION_THRESHOLD: float = 128.0

# Phase 3 food density: total A + B particles maintained on the grid.
# Doubled from Phase 1.5's FOOD_TARGET so a founder population at ~50%
# net reward rate still has food headroom.
PHASE3_FOOD_TARGET: int = 400

# Phase 3 reward magnitudes. +POSITIVE_ENERGY is unchanged from Phase 1.5.
# -NEGATIVE_ENERGY magnitudes the cost of being wrong about the season.
# 0 makes the founder blind to season flip (degenerates to static_dual);
# higher values make the founder starve faster.
PHASE3_FOOD_A_NEGATIVE_ENERGY: float = -3.0
PHASE3_FOOD_B_NEGATIVE_ENERGY: float = -3.0

# Energy threshold above which an organism reproduces. After one successful
# meal a proto-organism should already be close to replication.
REPRODUCTION_THRESHOLD: float = 64.0

# Energy transferred to the child on reproduction (subtracted from parent).
REPRODUCTION_ENERGY: float = 24.0

# Offspring spawn this many world units away from the parent, along a
# slightly jittered heading, so parent and child do not compete on the
# same food particle from the same pose.
CHILD_DISPERSAL_MIN: float = 1.0
CHILD_DISPERSAL_MAX: float = 4.0
CHILD_HEADING_NOISE: float = 0.2

# Maximum age in ticks. Pure safety net to prevent immortal lineages from
# monopolising the world even if their energy stays positive forever.
MAX_AGE: int = 50_000


# --- Brain -----------------------------------------------------------------

# Phase 3 sensor layout: 3 directions (left, front, right) x 2 resources
# (FoodA, FoodB). Order:
# Index 0: a_left  Index 1: a_front  Index 2: a_right
# Index 3: b_left  Index 4: b_front  Index 5: b_right
# Each value in [0, 1] (max smell strength normalised).
# The intake_feedback channel in MemoryEcologyWorld is the 7th sensor,
# set by that subclass's _sensors_for. There is no plain "Phase 3
# sensors" sensor here: legacy world still has 3 (Phase 1.5), and the
# MemoryEcologyWorld subclass overrides everything.
N_SENSORS: int = 3

# Motor outputs (both tanh, zero-centered):
# Index 0: turn_drive      in [-1, 1] -> MAX_TURN_RATE (signed).
# Index 1: locomotion_drive in [-1, 1]; drive <= 0 is rest, drive > 0
#          is forward speed. Basal activity comes from NodeGene.bias,
#          not from a forced sigmoid(0)=0.5 or a locked-zero rest.
N_MOTORS: int = 2

# Founders start with no hidden neurons. Complexity (hidden nodes,
# recurrence) can appear later via structural mutation.
N_HIDDEN: int = 1

# Std of initial connection weights. Strong enough that smell can
# override the weak locomotion prior and flip rest ↔ move.
INITIAL_WEIGHT_SIGMA: float = 0.20

# Founder locomotion bias ~ N(0, this). A weak prior, not a locked
# gait: turn bias stays 0 so founders do not spin.
INITIAL_LOCOMOTION_BIAS_SIGMA: float = 0.10

MAX_LINEAR_SPEED: float = 3.0
MAX_TURN_RATE: float = 0.5
# Drives at or below this are rest. 0 means "non-positive = sit";
# a tiny value (≈0.01) can be used later if numerical jitter crawls.
MOVE_DEADZONE: float = 0.0
EAT_RADIUS: float = 4.0
COLLISION_RADIUS: float = 3.0


# --- Sensors: smell field ---------------------------------------------------

# Radius (in cells) of the smell diffusion kernel. Larger radius ->
# longer-range smell but more compute per tick.
SMELL_FIELD_RADIUS: int = 24

# Half-angle of each smell sector in radians. 3 sectors of width
# 2*half_angle cover (3 * 2 * half_angle) radians; for pi/3 (=60deg)
# half-angle each, this gives full 360 coverage with overlap.
SMELL_HALF_ANGLE: float = 1.05  # ~60deg


# --- Metabolic cost (v2) ----------------------------------------------------

# Per-tick energy drain per neuron. Bigger brains are more expensive.
NEURON_METABOLIC_COST: float = 0.005

# Per-tick energy drain per active connection.
CONNECTION_METABOLIC_COST: float = 0.001


# --- Mutation --------------------------------------------------------------

# Per-birth probability of applying the weight-mutation operator.
WEIGHT_MUTATION_RATE: float = 0.8

# Probability per enabled connection of a small gaussian nudge.
WEIGHT_PERTURB_RATE: float = 0.10

# Standard deviation of the small gaussian nudge.
WEIGHT_PERTURB_SIGMA: float = 0.10

# Probability per enabled connection (if not perturbed) of a rare
# large jump: the weight is redrawn from N(0, WEIGHT_REPLACE_SIGMA).
WEIGHT_REPLACE_RATE: float = 0.02
WEIGHT_REPLACE_SIGMA: float = 1.0

# Hard clamp on absolute weight magnitude, post-mutation.
WEIGHT_MAX: float = 5.0

# Bias mutation (NodeGene.bias). This is how basal locomotion can
# evolve: without it a zero-smell net is stuck at tanh(0)=0 forever.
BIAS_MUTATION_RATE: float = 0.2
BIAS_PERTURB_SIGMA: float = 0.05
BIAS_REPLACE_RATE: float = 0.01
BIAS_REPLACE_SIGMA: float = 0.20
BIAS_MAX: float = 2.0

# Per-birth probability of structural mutations.
#
# Phase 0 baseline (kept rare until food-seeking is stable):
#   ADD_NODE_RATE = 0.002, ADD_CONNECTION_RATE = 0.005
# Phase 1: rates bumped 5x to make hidden↔hidden cycles more likely
# to emerge within a single 50k-tick run. We don't expect runaway
# complexity at these rates — the founder is pure feed-forward and
# structural mutations must first build a hidden node, then a
# second, then connect them. 1 in 50 births adds a node and 1 in 20
# adds a connection; over 50k ticks with ~10k births that is
# statistically enough to see the first cycle.
ADD_NODE_RATE: float = 0.01
ADD_CONNECTION_RATE: float = 0.02
TOGGLE_CONNECTION_RATE: float = 0.005


# --- Selection (natural) ---------------------------------------------------

# No external fitness function. Organisms that find food live; those that
# don't die. Reproduction happens automatically when energy exceeds
# REPRODUCTION_THRESHOLD. No tournament, no ranking, no comparison.
# Selection is purely "did you eat enough to not starve and to afford
# reproduction?" Eating is also automatic on contact; the brain evolves
# navigation, not decisions.


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

# Ticks between species / behavior snapshots.
METRICS_SPECIES_EVERY: int = 50
METRICS_BEHAVIOR_EVERY: int = 100


# --- Speciation (observational taxonomy only) ------------------------------

# Compatibility distance weights. Excess/disjoint are NOT divided by
# genome size: one new connection should matter on these small brains.
COMPAT_C_EXCESS: float = 1.0
COMPAT_C_DISJOINT: float = 1.0
COMPAT_C_WEIGHT: float = 0.4
COMPAT_C_BIAS: float = 0.2

# Child joins parent species if distance to its representative is
# at most this; otherwise we search other living species or originate.
SPECIES_THRESHOLD: float = 3.0

# How often to refresh each living species' representative genome.
SPECIES_REPRESENTATIVE_EVERY: int = 500

# Spatial bin size (cells) for exploration_rate.
EXPLORE_BIN: int = 16

# |turn| above this (radians/tick) counts as TURN_LEFT / TURN_RIGHT
# rather than FORWARD for state-dependence.
TURN_ACTION_THRESHOLD: float = 0.05

# A living species is "established" only after it has persisted long
# enough and grown a real population. One-off mutants stay "newborn".
ESTABLISHED_MIN_AGE: int = 1000
ESTABLISHED_MIN_PEAK: int = 10


@dataclass(frozen=True)
class V0Summary:
    """A read-only summary of v0 constants. Useful for logging and tests."""

    n_sensors: int = N_SENSORS
    n_hidden: int = N_HIDDEN
    n_motors: int = N_MOTORS
    population_cap: int = POPULATION_CAP
    initial_population: int = INITIAL_POPULATION
    world_size: tuple[int, int] = (WORLD_WIDTH, WORLD_HEIGHT)


def locomotion_speed(drive: float) -> float:
    """Map a tanh locomotion drive in [-1, 1] to forward speed.

    Non-positive drive (and anything at or below MOVE_DEADZONE) is rest.
    Positive drive is speed proportional to the drive. No backward motion:
    waiting is the cheap alternative to roaming, not reversing.
    """
    if drive <= MOVE_DEADZONE:
        return 0.0
    return drive * MAX_LINEAR_SPEED
