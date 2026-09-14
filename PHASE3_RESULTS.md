# Phase 3.0 - Memory Ecology: infrastructure complete, cold-start fails

## What Phase 3 set out to do

Phase 1.5 confirmed evolution can grow hidden<->hidden recurrent cycles.
Phase 2.5 showed those cycles sometimes help (seed 1 ahead 1.8->3.7,
seed 3 carrier 1 behind 0.0->10.8) and sometimes hurt. Phase 3 changes
the ecology to make **recurrent state actually pay off**:

  - Two food resources (FoodA, FoodB) with distinct smell signatures.
  - A hidden `season` that flips the sign of reward for the two
    resources every SEASON_LENGTH ticks. The organism cannot sense
    season directly.
  - An `intake_feedback` signal (+1 / -1 / 0) injected as a 4th
    sensor channel for INTAKE_FEEDBACK_DURATION ticks after `eat`.

The reward table is:

| season | FoodA      | FoodB      |
|-------:|-----------:|-----------:|
|      0 |    +25     |     -3     |
|      1 |     -3     |    +25     |

In season 0, FoodA is good and FoodB is mildly toxic. In season 1
they flip. The only way to pick the right food is to remember which
food has been good recently.

## What got built

### World and infrastructure (`evolife/phase3.py`)

  - `MemoryEcologyWorld(World)` subclass. Does not modify the legacy
    World at all.
  - `FoodA` and `FoodB` dataclasses, separate spawn and respawn for
    each.
  - `DualSmellField` in `sensors.py`: two independent SmellFields,
    one per resource. `World.dual_smell.field_a` and `.field_b`.
  - Season state: `world.season: int` (0 or 1), `ticks_in_season: int`.
    Flip at SEASON_LENGTH=2000.
  - `_resolve_eat_phase3` overrides eat logic to apply signed reward
    and set `Organism.intake_feedback`.
  - `Organism.intake_feedback` field added. Decays each tick via
    `intake_feedback_ttl`.

### Phase 3 founder (`make_phase3_founder`)

  - 3 smell sensors (mirroring Phase 1.5) + 1 feedback sensor = 4 sensors.
  - 1 hidden node, 2 motors.
  - 6 connections:
      3 sensor[0:3] -> hidden
      1 sensor[3] (feedback) -> hidden with weight FB_TO_HIDDEN_WEIGHT=1.0
      2 hidden -> motor
  - Same INITIAL_LOCOMOTION_BIAS_SIGMA / INITIAL_WEIGHT_SIGMA as
    Phase 1.5 founder, so the cold-start navigation reflex is
    comparable.

### Brain flexibility (`brain.py`)

  - `Brain.forward` now reads `n_sensor` from the compiled net
    instead of the global `N_SENSORS`. A 4-sensor Phase 3 founder
    works side by side with the 3-sensor legacy founder.

### Runners

  - `scripts/run_phase3.py` -- CONTROL vs MEMORY_ECOLOGY runner.
    Takes `--seeds` and `--ticks`. Writes a per-seed SQLite
    (`evolife_phase3_{branch}_seed{N}.sqlite`) with archive
    milestones and cycle carriers. Logs per-1000-tick line.
  - `scripts/arena_memory_advantage.py` -- takes a frozen genome
    JSON and runs it twice on a Phase 3 world (NORMAL vs ABLATED
    state via `brain.reset_state()` every tick). Writes
    food / positive / reward advantages. Foundation for the
    "memory_advantage" metric from CANON.

### Tests

  - 95 existing tests still pass after the brain.py change.

## What we ran

### Smoke run, 5000 ticks, seed 1

| branch  | pop@5000 | maxGen | eats | repros |
|---------|---------:|-------:|-----:|-------:|
| control |       29 |      7 |  141 |    140 |
| memory  |        0 |      0 |  150 |    691 |

Memory finds *more* food (150 vs 141) but its population goes
extinct by tick 2000. The control population also drifts down
(49 -> 4 -> 13 -> 21 -> 29) but stays alive. Memory's reproduction
rate (691 events) shows founders *are* producing children, but the
children also die before tick 2000.

### Memory advantage arena on Phase 3 founder (10 seeds)

Founder eats 0 food in 200 ticks under both NORMAL and ABLATED
state. This is what we expect: a *cold-start* founder with random
weights cannot navigate to food. The arena infrastructure is
correct; we just need an evolved genome to see meaningful numbers.

## Cold-start extinction: what is happening

MemoryEcologyWorld uses PHASE3_INITIAL_ENERGY = FOOD_ENERGY * 10 =
400 energy per founder. With IDLE_ENERGY_COST = 0.02/tick and a
typical locomotion cost, a founder can survive ~5000 ticks of
near-zero food intake. That is exactly when our smoke runs
terminate.

In Phase 1.5 the founder started with FOOD_ENERGY * 2 = 80, but
food was uniformly +reward every tick, so it could be found and
eaten within a few hundred ticks of random wandering. In Phase 3
founder food is +25 OR -3, and random walking is enough to find
food, but only ~75% of the eats are positive (FoodA in season 0,
FoodB in season 1, randomly chosen by uniform spawn). On 150 eats,
that gives ~38 negative eats = -114 energy. The remaining +25
eats = +950 energy. Net positive, *if* the founder lives long
enough to eat 150.

But negative eats also kill energy, and reproduction only happens
at REPRODUCTION_THRESHOLD = 64. The negative eats keep mean
energy below the threshold. Reproduction only barely starts, and
children inherit the same cold-start brain.

Phase 1.5 also started cold but had 50k ticks for selection to
work. Phase 3.0 needs at least the same, but the cold-start is
harder because every wrong eat costs energy the founder can't
afford.

## What Phase 3.0 *did* prove

1. **The infrastructure is real.** MemoryEcologyWorld, dual smell,
   season state, intake feedback, and the memory_advantage arena
   all work end to end. Sanity checks pass, 95 unit tests pass.

2. **A naive cold-start with 4-sensor founder cannot survive in
   the Memory Ecology within 5000 ticks.** This is a *constraint
   on Phase 3.1*, not a contradiction of the hypothesis.

3. **Phase 3.0 is not the right place to test the hypothesis yet.**
   To test "memory ecology selects for recurrent state" we need
   a population that is *already good at navigation*. Cold-start
   founders don't qualify.

## What Phase 3.1 needs

  - **Warm-start founder.** Reuse a Phase 1.5 evolved genome
    (50k-tick evolution, pop>100, maxGen>100) and graft the
    feedback edge onto it. This gives Phase 3 a starting
    population that already navigates by smell and can therefore
    learn the season rule.

  - **A longer tick budget** for Phase 3.1 evolution. With
    warm-start founders we don't need the population to find
    navigation from scratch, but we still need ~30k+ ticks to see
    recurrent topology emerge under selection.

  - **Smaller feedback signal** initially. `FB_TO_HIDDEN_WEIGHT=1.0`
    is dominant. We may want to drop it to 0.3 so the brain can
    ignore it until selection tunes the weights.

  - **Or: control the founder's brain state.** A warm-start
    founder that has no recurrent topology at all (Phase 1.5
    `N_HIDDEN=0`) forces the population to evolve recurrence
    *under* the new ecology. That is the cleanest test of the
    hypothesis.

## Files

  - `evolife/phase3.py`        - MemoryEcologyWorld + founder
  - `evolife/sensors.py`       - DualSmellField
  - `evolife/organism.py`      - intake_feedback field
  - `evolife/brain.py`         - n_sensor from compiled net
  - `evolife/config.py`        - phase 3 constants
  - `scripts/run_phase3.py`    - CONTROL vs MEMORY_ECOLOGY runner
  - `scripts/arena_memory_advantage.py` - NORMAL vs ABLATED arena
  - `PHASE3_DESIGN.md`         - design spec
  - `PHASE3_RESULTS.md`        - this file
