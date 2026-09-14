# Phase 3 — Memory Ecology: Design Spec

## Goal

Make recurrent state pay off. Replace the current (near-Markov) world
with one where the same current observation requires *different*
actions depending on history. If evolution still finds recurrent
topology here, that is a real signal — not a statistical accident.

## World changes (vs current `evolife/world.py`)

### Two resources, same physics

Replace single `Food(x, y)` with two typed classes:

```
FoodA(x, y)   — neutral resource, light smell (e.g. orange hue)
FoodB(x, y)   — neutral resource, distinct smell (e.g. blue hue)
```

Same SmellField machinery, but **two grids** (`smell_a`, `smell_b`).
`_sensors_for(org)` returns a 6-vector:

```
sensors = [a_left, a_front, a_right, b_left, b_front, b_right]
```

No sensor for season, no sensor for energy, no sensor for "I just ate".
The brain sees smell only.

### Hidden season with switching rewards

Global state on `World`:

```
self.season: int            # 0 or 1
self.ticks_in_season: int
```

`SEASON_LENGTH = 3000` ticks (configurable). When `ticks_in_season` hits
`SEASON_LENGTH`, swap season and reset counter. When season swaps,
**all currently-living food particles flip their reward sign on next
eat**.

Reward table:

| season | eat FoodA | eat FoodB |
|-------:|----------:|----------:|
|      0 |    +25    |    -10    |
|      1 |    -10    |    +25    |

The energy delta is what the organism *experiences*. The smell is the
*same* both seasons. The only way to pick the right food is to
remember whether the last several eats were positive or negative.

### Intake feedback into hidden state

After `eat`, write a single short-lived signal into the **brain's
hidden state**. Implementation:

- New `Organism.intake_feedback: float = 0.0`
- Set to `(+1.0 if reward > 0 else -1.0)` for one tick only.
- The brain's first hidden node gets a **feedback input edge** that
  injects `intake_feedback` directly into its activation:
  `delta[hidden_0] += feedback_weight * intake_feedback`
  where `feedback_weight` starts at `1.0` and is a heritable gene.

This is the "internal reward signal" — it does NOT appear in the
sensor array but it IS available to the brain via a dedicated edge.

**Why hidden-only and not sensor?** Because CANON says no reward
sensor in policy. The feedback is wired into state, not into perception.

### World math remains

- Tor 512×512, same movement costs, same reproduction threshold,
  same energy budget. Don't tune the package here — only the
  resource / season layer changes.
- 200 initial founders, ~50/50 over FoodA and FoodB initial spawns.

### Founder (stays N_HIDDEN=1 for now)

`make_default_genome` with `N_SENSORS=6`, `N_HIDDEN=1`, `N_MOTORS=2`:

- 6 sensor→hidden edges (small random weights)
- 2 hidden→motor edges (small random weights)
- 1 feedback→hidden edge with weight 1.0 (fixed, NOT heritable
  initially — this is the "nervous system has a reward channel")
- 9 connections total

We keep `N_HIDDEN=1` here only as a tractability shortcut, exactly as
we did in Phase 1.5. The plan is to **revert to N_HIDDEN=0** after
Memory Ecology demonstrably selects for recurrent state.

## Memory advantage metric (causal)

For any genome, run the Behavioral Arena twice on the same scenarios
with the same seeds:

1. **NORMAL**: brain state persists across ticks.
2. **ABLATED**: `brain.reset_state()` is called before every tick.

```
memory_advantage(genome, scenario) =
    food_eaten_mean(NORMAL) - food_eaten_mean(ABLATED)
```

If state doesn't help, `memory_advantage ≈ 0`. If state matters,
positive.

Track over evolution:

- `recurrent_organisms` (fraction of living organisms whose brain has
  hidden↔hidden edges)
- `memory_advantage` of the lineage-median genome at gen 10, 40, 80,
  120, 200
- `correct_food_choice` (proxy: ratio of positive-reward eats to total
  eats over a window, computed only in CONTROL-vs-MEMORY mode)

## Control experiment

Two identical-line runs, only the world differs:

| branch             | world                                          |
|--------------------|------------------------------------------------|
| CONTROL            | current static food world (single resource)    |
| MEMORY_ECOLOGY     | new two-resource + season world                |

Identical: mutation rates, brain costs, population cap, initial RNG
seed, founder. We run **5 seeds × 100k ticks** for each branch.

Compare:
- recurrent topology frequency over time
- memory_advantage of sampled genomes
- lineage persistence
- mean food_eaten per individual
- max generation reached

## Safety / liveness checks

- CONTROL must still hit Phase 1.5-like numbers (alive, eating,
  reproducing). If it diverges, the new founder broke something.
- MEMORY_ECOLOGY extinction at start is possible (sign flipping
  early disorients founders) — that's a result, not a bug. But we
  want at least 1/5 seeds alive by tick 50k.

## What this phase does NOT add

- No contact recombination yet (CANON §4.3 — comes after Memory
  Ecology works).
- No synaptic plasticity (η, A, B, C, D). Comes after.
- No colonies / kin selection. Comes after.
- No control arm without jitter, without reward, etc. — those are
  diagnostic checks AFTER Memory Ecology selects for state.

## Files touched

- `evolife/config.py` — new constants (N_SENSORS=6, SEASON_LENGTH,
  FOOD_A_ENERGY / FOOD_B_ENERGY, reward sign mapping).
- `evolife/world.py` — `FoodA`/`FoodB` types, two smell grids,
  season state, intake feedback hook.
- `evolife/sensors.py` — `SmellField` extended to two grids OR
  `World` holds two SmellFields; `_sensors_for` returns 6-vector.
- `evolife/organism.py` — `intake_feedback` field.
- `evolife/brain.py` — `make_default_genome` wires feedback edge
  into hidden.
- `scripts/run_v3_*.py` — CONTROL vs MEMORY_ECOLOGY runner, ABLATION
  arena.
- `tests/test_*.py` — genome shape, brain wiring, season transitions.

## Open risks

1. **6 sensors × 1 hidden = 6 weights**. Default sigma=0.05 means
   initial hidden output ≈ 0, motor bias drives everything. Phase 1.5
   test `test_smell_can_stop_a_weak_roamer` will need re-tooling.
2. **feedback edge weight is fixed** to 1.0 in v1. If it dominates
   and evolution never tunes hidden→motor away, brain is essentially
   "follow the last reward". We accept this for Phase 3.0; in 3.1
   we make the weight heritable (mutate like any other).
3. **Season length 3000 ticks** is a guess. If extinction dominates
   the first half-cycle we may need to shorten or initialise food
   patches with `H=1` so founders see something. This is a parameter
   sweep, not a redesign.
