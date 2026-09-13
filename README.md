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

### V2.x — restore viable bootstrap (current)

13 independent seeds × 10 000 ticks. Five concrete fixes:

| Bug | Symptom | Fix |
|---|---|---|
| Identical founder weights | All 200 founders walk in lockstep circles | Pass `rng` into `make_default_genome` |
| Saturated smell sensor | `[1,1,1]` everywhere; no gradient | Replace sector sums with 3 probe-point samples at distance 8 with soft saturation `1-exp(-k*v)` |
| Eat/reproduce from brain | Required learning two extra decisions before navigation | Eat on contact; reproduce at energy threshold automatically |
| Fake recurrent start | `hidden0 → hidden1` was a single feed-forward edge | Two opposing edges: `hidden0 ↔ hidden1` |
| Toroidal sample wrap | Edge organisms got distorted perception | Wrap probe points mod (width, height) |

Brain now only controls `turn` and `move`. The 2-motor brain is 13
neurons and 37 connections (with the new mutual recurrent pair).

Headless 13 seeds × 10 000 ticks:

```
seeds = 13
eats          median=308  mean=313  range=[209..441]
reproductions median=77   mean=79   range=[47..123]
final pop     0/13 (extinction by tick 10000)
throughput    207-534 ticks/s (faster as population shrinks)
```

Total: 4 070 eats, 1 027 reproductions across 130 000 organism-ticks.
Compare to v2 (before fixes): 80 eats, 1 reproduction across 250 000
organism-ticks — ~50x improvement in eats, ~1000x in reproductions.

**Interpretation:** the experiment is now correctly bootstrapped.
Founder-ы движутся, иногда едят, иногда размножаются — но естественного
отбора пока недостаточно для устойчивой популяции за 10 000 тиков.
Это уже не сломанная симуляция; это давление среды, которое выше
текущей способности маленьких случайных мозгов.

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
python scripts/run_v2_experiment.py --ticks 10000 --seeds 1 2 3
pytest -q
```
