# EvoLife

A digital organism simulation. The question this experiment asks is not
"can we train a model to do X" but:

> Can a minimal digital world, given energy costs, reproduction and a
> mutable genome, generate pressure that makes nervous systems more
> complex on their own?

## V0 (this iteration)

A scaffold only. No evolution yet — just a world where organisms with
fixed-topology brains eat, move, reproduce and die, so we can see the
plumbing work before we let selection run.

- 2D continuous world, 512×512, food particles respawn up to a cap.
- ~500 organisms, Pygame top-down visualiser, NumPy on CPU.
- Genome is NEAT-shaped from day one (innovation numbers, structural
  mutations defined) but v0 only mutates weights.
- Only food and starvation. No poison, no day/night cycle. Pressure on
  complexity comes from metabolic cost of the brain itself, not from
  external tricks.

## Baseline (this commit)

10 000 headless ticks, seed 42, no evolution (founder brains have
frozen random weights):

```
ticks=10000  elapsed=16.28s  rate=614.2 ticks/s
final_pop=0  final_mean_energy=0.00
```

Population collapses to zero somewhere between tick 50 and tick 5050.
This is expected and is the v0 baseline we want: without selection,
random brains cannot reliably find food, and founder energy drains
faster than replenishment. Once we flip the evolution switch in v1, we
expect this curve to invert.

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
  config.py          # all v0 constants
  genome.py          # NEAT-shaped Genome (nodes, connections, innovations)
  brain.py           # feedforward net built from a Genome
  organism.py        # position, energy, age, brain, genome
  world.py           # tick(), spawn_food(), reproduce()
  mutation.py        # all four mutations defined; v0 uses only weight perturb
  speciation.py      # compatibility distance (v0: always one species)
  metrics.py         # per-organism and per-world metrics into SQLite
  visualization.py   # Pygame top-down renderer
scripts/
  run_visual.py      # run with Pygame
  run_headless.py    # run N ticks headless, write metrics
tests/
  test_genome.py
  test_brain.py
  test_world.py
  test_mutation.py
```

## Run

```
pip install -e .
python scripts/run_visual.py
python scripts/run_headless.py --ticks 10000 --seed 42
pytest -q
```
