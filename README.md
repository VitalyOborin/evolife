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

Tournament killed faster than reproduction replaced. Diagnosis: a
hidden fitness function (peak_energy comparison + clone-energy injection)
contradicted the "no explicit fitness" principle.

### V2 — Natural Selection Baseline (current)

Five independent seeds × 50 000 ticks. Removes tournament entirely;
recurrent brains with persistent state; local smell-only perception;
metabolic cost on neurons and connections; append-only event log.

Question:

> Can a population of random minimal recurrent brains sustain itself
> across generations when perception is purely local, with no external
> fitness function beyond "did you find food"?

Headless smoke test at seed=42, 2000 ticks:

```
ticks=2000  elapsed=9.58s  rate=208.8 ticks/s
final_pop=0  events=412  births=200  deaths=200  eats=12
```

Throughput 208 ticks/s — about 2x v1, despite the smell field and
recurrent iterations. The 12 eats in 2000 ticks mean random founders
occasionally stumble onto food; the population cannot yet sustain.

Full 5-seed × 50k experiment running; results in the report below.

## Hard constraints (do not break)

- **No LLM, no backprop, no RL, no datasets, no human labels.** Evolution
  is the only learning signal.
- **No explicit fitness function.** Fitness emerges as the number of
  surviving descendants. No tournament, no ranking, no comparison.
- **Single discrete tick.** No `time.sleep`, no event loop driving sim.
- **Deterministic via seed.** `numpy.random.default_rng(seed)` lives in
  `World`. No global RNG.
- **Local perception only.** Organisms have no GPS; they sense smell
  intensity in three sectors.

## Layout

```
evolife/
  config.py          # all constants
  genome.py          # NEAT-shaped Genome (nodes, connections, fingerprint)
  innovation.py      # global innovation counter
  brain.py           # recurrent, persistent-state brain
  organism.py        # position, energy, age, brain, genome
  world.py           # tick, food, eat, reproduce, metabolic cost
  mutation.py        # weights, add_node, add_connection, toggle
  speciation.py      # compatibility distance (v0: one bucket)
  events.py          # append-only Birth/Death/Reproduction/Eat
  metrics.py         # SQLite snapshots + event flush
  sensors.py         # SmellField: per-tick diffusion grid
  visualization.py   # Pygame top-down renderer
scripts/
  run_visual.py
  run_headless.py
  run_v2_experiment.py
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
python scripts/run_headless.py --ticks 2000 --seed 42
python scripts/run_v2_experiment.py --ticks 50000 --seeds 1 2 3 4 5
pytest -q
```
