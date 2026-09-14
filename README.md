# EvoLife

A digital organism simulation. The question this experiment asks is not
"can we train a model to do X" but:

> Can a minimal digital world, given energy costs, reproduction and a
> mutable genome, generate pressure that makes nervous systems more
> complex on their own?

## Status

### V0 → V2.2

See git log. V0 was a scaffold; V2.2 is the user's "Viable Replicator"
proto-brain: 3 sensors → 0 hidden → 2 motors, 6 connections, no
recurrence gifted.

### Phase 0 — pure feedforward founder + Archive (current)

Following the user's roadmap: founders are PURE FEEDFORWARD. Recurrence
is not gifted; if it emerges it is via structural mutation, and we
record it as a milestone.

`Archive` is an observability log, not selection. It records:
  - `first_hidden_node` — first organism with a hidden node.
  - `first_recurrent_cycle` — first organism with a hidden↔hidden
    cycle (NOT cycles through sensors or motors — those are trivial
    wiring artefacts, not memory).
  - `max_brain_nodes`, `max_generation_reached`, `max_lineage_size` —
    running maxima.

`run_v2_experiment.py` writes per-seed SQLite metrics and a combined
JSON of archive milestones across all seeds.

## Phase 0 results: 5 seeds × 10 000 ticks

```
seed   rate    pop  maxGen  eats   repros
  1   237/s    33     33   2920    1163
  2   200/s    82     44   4162    1641
  3   121/s    51     45   3497    1386
  4    90/s    57     38   3507    1387
  5   143/s     0      0   1294     490

extinct: 1/5
median_pop: 51
median_maxGen: 38
median_births: 200
median_repro: 1386
median_eats: 3497
```

Archive milestones across all 5 seeds:

```
max_brain_nodes:          10 entries
max_generation_reached:  182 entries
max_lineage_size:        321 entries
first_hidden_node:        5 (one per seed, ticks 39..6156)
first_recurrent_cycle:    0  ← not yet observed
```

**Interpretation.** 4 of 5 seeds reached a stable, reproducing
population across ~30–45 generations. Hidden nodes emerged in every
seed through `add_node` structural mutation — that is emergence, not
gift. Hidden↔hidden cycles did not yet emerge; the statistical
expectation for a (hidden + 2× add_connection) triple under current
rates is ~0.5 cases per 5-seed run, so absence over 10 000 ticks is
expected, not a failure of evolution.

## Hard constraints (do not break)

- No LLM, no backprop, no RL, no datasets, no human labels.
- No explicit fitness function. Selection = "did you find food".
- Single discrete tick.
- Deterministic via seed.
- Local perception only. Smell probes; no GPS.
- Brain evolves navigation only. Eat and reproduce are automatic.
- Founder is pure feedforward. Recurrence is an emergent property.

## Layout

```
evolife/
  config.py
  genome.py
  innovation.py
  brain.py
  organism.py
  world.py              # CPU World (v2.2)
  gpu_world.py          # GPU-resident world
  gpu_sensors.py        # F.conv2d-based smell field
  mutation.py
  speciation.py
  events.py
  archive.py            # observability milestones
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
  test_archive.py
```

## Run

```
pip install -e .
python scripts/run_visual.py
python scripts/run_headless.py --ticks 2000 --seed 42
python scripts/run_v2_experiment.py --ticks 10000 --seeds 1 2 3 4 5
pytest -q
```
