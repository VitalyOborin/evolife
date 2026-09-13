# EvoLife

A digital organism simulation. The question this experiment asks is not
"can we train a model to do X" but:

> Can a minimal digital world, given energy costs, reproduction and a
> mutable genome, generate pressure that makes nervous systems more
> complex on their own?

## Status

### V0 — scaffold

10 000 headless ticks, seed 42, no evolution: population collapses to
zero (frozen random brains cannot reliably find food). Throughput 614
ticks/s.

### V1 — microbial tournament + structural mutations

20 000 headless ticks, seed 42, evolution on. Tournament killed faster
than reproduction replaced because it was effectively a hidden fitness
function (peak_energy ranking + clone-energy injection).

### V2 — Natural Selection Baseline (initial)

Removed tournament. Recurrent brain with persistent state. Local smell
sensors replacing GPS. Metabolic cost on neurons and connections.
Event log.

5 seeds × 50 000 ticks, all extinct by tick 50 000. Negative result:
the experiment was correctly framed but the simulation had bootstrap
issues that needed separate fixing.

### V2.x — restore viable bootstrap

13 independent seeds × 10 000 ticks. Five concrete fixes (identical
founder weights, saturated smell, eat/reproduce motors, fake recurrence,
toroidal wrap). Founders moved, ate, reproduced — but median 77 births
on 200 founders (`R ≈ 0.4`) and 13/13 extinctions by tick 10 000.
Natural selection never had generations to work with.

### V2.2 — Viable Replicator (current)

The previous stage asked "can a random dense RNN find enough food to
not go extinct?" The answer was no. This stage does not add mechanisms.
It makes the simplest ancestor a viable replicator and gives selection
tens of generations.

| Change | Why |
|---|---|
| 3 sensors (left/front/right); drop energy + bias | Bias=1 forced constant turn; energy made networks "wake up" near death |
| Founder: 0 hidden, 6 connections, weights `N(0, 0.05)`, motor bias 0 | No smell → walk roughly straight (`move ≈ 0.5`), not circle |
| `_ITERATIONS = 1` | Internal state tracks world time, not an intra-tick attractor |
| Weight mutation `0.8 / 0.10 / 0.10` + rare replace | Child inherits ~95% of parent behaviour |
| Structural mutation ~10× rarer | First prove the proto-brain weights can evolve |
| `REPRODUCTION_THRESHOLD=48`, `REPRODUCTION_ENERGY=24` | One successful meal puts a good organism near replication |
| Food binomial on deficit (`p=0.002`) | Field actually replenishes toward `FOOD_TARGET` |
| Child dispersal 1–4 units, heading noise 0.2 | Parent and child do not compete on the same food |
| `generation` + `founder_lineage_id` | So we can see whether evolution had generations at all |

Success criterion (not "smartness"): 20 seeds × 50 000 ticks, median
run not extinct, median `max_generation > 30`, thousands of births,
population neither pinned to the cap nor collapsing to 0.

Preliminary (not the full 20 × 50 000 criterion):

```
3 seeds × 8 000 ticks:
  extinct 0/3   median pop 70   median maxGen 35   median repro 1065

1 seed × 20 000 ticks:
  pop 108   maxGen 78   reproductions 2767   1 surviving founder lineage
```

Compare to v2.x: median 77 reproductions, 13/13 extinct by tick 10 000,
`max_generation` was not even measured. The population now persists
and a single lineage can run for dozens of generations. Full 20 × 50 000
is the actual v2.2 gate; after that, mutations-off vs mutations-on.

## Hard constraints (do not break)

- **No LLM, no backprop, no RL, no datasets, no human labels.** Evolution
  is the only learning signal.
- **No explicit fitness function.** Fitness emerges as the number of
  surviving descendants. No tournament, no ranking, no comparison.
- **Single discrete tick.** No `time.sleep`, no event loop driving sim.
- **Deterministic via seed.** `numpy.random.default_rng(seed)` lives in
  `World`. No global RNG.
- **Local perception only.** Organisms have no GPS; they sense smell
  intensity at three probe points ahead.
- **Brain evolves navigation only.** Eat and reproduce are automatic.

## Layout

```
evolife/
  config.py
  genome.py
  innovation.py
  brain.py
  organism.py
  world.py
  mutation.py
  speciation.py
  events.py
  metrics.py
  sensors.py
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
python scripts/run_v2_experiment.py --ticks 50000 --seeds 1 2 3
pytest -q
```
