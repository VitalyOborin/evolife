# EvoLife

A digital organism simulation. The question this experiment asks is not
"can we train a model to do X" but:

> Can a minimal digital world, given energy costs, reproduction and a
> mutable genome, generate pressure that makes nervous systems more
> complex on their own?

## Status

### V0 — scaffold (committed)

10 000 headless ticks, seed 42, no evolution: population collapses to
zero as expected (frozen random brains cannot reliably find food).
Throughput: 614 ticks/s.

### V1 — microbial tournament + structural mutations (committed)

20 000 headless ticks, seed 42, evolution on:

```
ticks=20000  elapsed=183.88s  rate=108.8 ticks/s
final_pop=2  final_mean_energy=2308.73
```

**Observation:** population collapses from 200 to 2 around tick 5000.
The two survivors are NOT the result of selection — they survived
because no rival was within TOURNAMENT_RADIUS. Mean energy of the two
climbs monotonically because they have the world to themselves. Genome
size grows modestly: avg connections 36.6, max 44 (start = 36).

**Diagnosis:** tournament kills faster than reproduction replaces. With
TOURNAMENT_KILL_RATE=0.8 and TOURNAMENT_EVERY=50 ticks, every 50 ticks
half the population loses a member, while reproduction only fires when
an organism's brain outputs reproduce_attempt > 0.5 AND it has
REPRODUCTION_THRESHOLD=60 energy. Brains rarely hit both.

**Next levers for v2:**
1. Lower REPRODUCTION_THRESHOLD so more organisms reproduce.
2. Lower TOURNAMENT_KILL_RATE so tournament is softer.
3. Add speciation (NEAT-style compatibility distance) so similar
   organisms don't compete directly and structural diversity persists.
4. Vectorise tournament and reproduction with numpy/PyTorch for
   10x+ throughput.

## Hard constraints (do not break)

- **No LLM, no backprop, no RL, no datasets, no human labels.** Evolution
  is the only learning signal.
- **No explicit fitness function.** Fitness emerges as the number of
  surviving descendants.
- **Single discrete tick.** No `time.sleep`, no event loop driving sim.
- **Deterministic via seed.** `numpy.random.default_rng(seed)` lives in
  `World`. No global RNG.

## Layout

```
evolife/
  config.py          # all v0/v1 constants
  genome.py          # NEAT-shaped Genome (nodes, connections, innovations)
  innovation.py      # global innovation counter (v1)
  brain.py           # feedforward net built from a Genome
  organism.py        # position, energy, age, brain, genome
  world.py           # tick(), spawn_food(), reproduce(), tournament()
  mutation.py        # weights, add_node, add_connection, toggle
  speciation.py      # compatibility distance (v0: one bucket)
  metrics.py         # per-organism and per-world metrics into SQLite
  visualization.py   # Pygame top-down renderer
scripts/
  run_visual.py      # run with Pygame
  run_headless.py    # run N ticks headless, write metrics
tests/
  test_innovation.py
  test_genome.py
  test_brain.py
  test_world.py
  test_mutation.py
```

## Run

```
pip install -e .
python scripts/run_visual.py
python scripts/run_headless.py --ticks 20000 --seed 42
pytest -q
```
