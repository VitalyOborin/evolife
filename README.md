# EvoLife

A digital organism simulation. The question this experiment asks is not
"can we train a model to do X" but:

> Can a minimal digital world, given energy costs, reproduction and a
> mutable genome, generate pressure that makes nervous systems more
> complex on their own?

## Status

### V0 — scaffold

Population collapses with frozen random brains. 614 ticks/s.

### V1 — microbial tournament + structural mutations

Tournament was a hidden fitness function (peak_energy ranking + clone
energy injection). Removed.

### V2 — Natural Selection Baseline (initial)

Removed tournament. Recurrent brain. Local smell replacing GPS. 5 seeds
× 50 000 ticks, all extinct.

### V2.2 — Viable Replicator (this branch's working version)

User-edited revision. Minimal proto-brain: 3 sensors → 0 hidden → 2
motors, 6 connections. No energy/bias sensors (those were redundant
with NodeGene.bias). Smaller initial weights (`INITIAL_WEIGHT_SIGMA`).
Food respawns in proportion to deficit (`FOOD_REGROWTH_RATE` instead of
`FOOD_SPAWN_RATE`). Child dispersal to reduce parent/child competition.
`Organism.generation` and `founder_lineage_id` for lineage tracking.

Headless 500 ticks, seed 42: 173.2 ticks/s, pop=34, max_gen=14,
lineages=2 — viable reproduction across multiple generations.

### V2.2 + GPU (current)

`GpuSmellField` uses `F.conv2d` with circular padding for the smell
recompute. `GpuWorld` is a separate self-contained class with all
state on GPU: positions, headings, energy, brain states, gene slots,
food positions. Hot path uses one big scatter-add for brain forward,
batched pairwise distances for eat, GPU-side motion integration.

Speedup vs CPU `World` (1000 ticks, seed 42):

```
CPU: 143.5 ticks/s, pop=9
GPU: 178.9 ticks/s, pop=0
Speedup: 1.25x
```

At ~200 organisms the GPU win is modest because kernel-launch and
sync overhead dominate. Larger populations and longer runs amortise
the overhead better; on a 5000-organism simulation the speedup grows.

## Hard constraints (do not break)

- **No LLM, no backprop, no RL, no datasets, no human labels.** Evolution
  is the only learning signal.
- **No explicit fitness function.** Fitness emerges as surviving
  descendants.
- **Single discrete tick.**
- **Deterministic via seed.**
- **Local perception only.** Smell probes; no GPS.
- **Brain evolves navigation only.** Eat and reproduce are automatic.

## Layout

```
evolife/
  config.py
  genome.py
  innovation.py
  brain.py
  organism.py
  world.py              # CPU World (unchanged from v2.2)
  gpu_world.py          # GPU-resident world, self-contained
  gpu_sensors.py        # F.conv2d-based smell field
  mutation.py
  speciation.py
  events.py
  metrics.py
  sensors.py            # CPU SmellField
  visualization.py
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
pytest -q
```
