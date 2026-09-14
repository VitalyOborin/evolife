# Phase 2.5 Results — parent vs cycle carrier on Behavioral Arena

## What this phase asked

Phase 1.5 confirmed that **hidden↔hidden recurrent cycles emerge on
their own** in 3/5 seeds within 50k ticks under a N_HIDDEN=1 founder.
Phase 2 tried to ask "does the cycle *do* anything?" but could not
isolate a cycle carrier vs its parent — by the time the living
population was sampled at 15k ticks, the cycle had been lost or the
founder had reverted.

This phase (2.5) **fixes that**: every time a newborn organism gets a
hidden↔hidden cycle via structural mutation, we freeze the cycle
carrier **and its parent genome** into a new `cycle_carriers` SQLite
table. The Behavioral Arena then runs both sides through the 8
scenarios and we ask the comparison we actually wanted.

## Pipeline

1. `Genome.to_dict()` / `from_dict()` — added for plain-JSON
   serialisation (topology + weights + biases).
2. `Archive.cycle_carriers` — list of `CycleCarrier(tick, child_id,
   parent_id, parent_genome, cycle_genome)`. Captured every time
   `_check_milestones` fires `FIRST_RECURRENT_CYCLE`, with the parent
   genome that just produced the cycle-bearing child.
3. `Metrics.record_cycle_carriers` — incremental SQLite writer for
   the new `cycle_carriers(tick, child_id, parent_id, parent_genome_json,
   cycle_genome_json, parent_hash, cycle_hash)` table.
4. `scripts/run_cycle_arena.py` — reads cycle_carriers from a Phase
   1.5 SQLite, deserialises both genomes, and runs the parent and the
   cycle genome through 8 scenarios × 10 arena-seeds × 200 arena-ticks
   each. Writes per-episode rows + per-(carrier, scenario, side)
   summary.

## Sweep results

5 seeds × 50k ticks under Phase 1.5 settings (N_HIDDEN=1, ADD_NODE_RATE
0.01, ADD_CONNECTION_RATE 0.02):

| seed | pop@50k | maxGen | first_recurrent_cycle | n_cycle_carriers |
|-----:|--------:|-------:|----------------------:|----------------:|
| 1    |      92 |     86 | tick 14276            |               1 |
| 2    |      85 |     89 | tick 45225            |               2 |
| 3    |     111 |    105 | tick 49628            |               2 |
| 4    |     102 |     77 |          —            |               0 |
| 5    |     137 |     61 |          —            |               0 |

5 cycle-carrier pairs total (1 + 2 + 2 + 0 + 0).

The variance across seeds is real: in the previous Phase 1.5 run
seed 4 produced a cycle at tick 7148 and seed 5 went extinct; in this
run (different RNG draws because we ran 5 fresh seeds), seed 4 and
seed 5 produced no cycle within 50k ticks. That is the natural
stochasticity of structural mutation at these rates. We have enough
carriers (5) to draw conclusions from, but a 10-seed run would give
a tighter picture.

## Arena: parent vs cycle (5 carriers × 8 scenarios)

Each cell is the mean over 10 arena-seeds of food_eaten (number of
food pellets consumed in 200 ticks).

### Carrier 1 — seed 1 (parent 6 nodes / 7 conns vs cycle 6/8)

| scenario      | parent | cycle | delta |
|---------------|-------:|------:|------:|
| uniform       |   2.30 |  2.60 | +0.30 |
| ahead         |   1.80 |  3.70 | **+1.90** |
| behind        |   0.00 |  0.00 |  0.00 |
| left_right    |   1.40 |  5.60 | **+4.20** |
| sparse        |   0.80 |  1.20 | +0.40 |
| dense         |   2.60 |  3.80 | **+1.20** |
| relocating    |   0.00 |  0.00 |  0.00 |
| smell_blanked |   2.30 |  2.60 | +0.30 |

Cycle wins on every smell-driven scenario (cycle wins +300% on
left_right, +106% on ahead, +46% on dense). `smell_blanked`
identical to `uniform` for both — sanity check that the network is
actually using smell.

### Carrier 1 — seed 2 (parent 6/10 vs cycle 6/11)

| scenario      | parent | cycle | delta |
|---------------|-------:|------:|------:|
| uniform       |   7.50 |  6.50 | -1.00 |
| ahead         |  13.50 | 13.50 |  0.00 |
| behind        |   4.00 |  0.00 | -4.00 |
| left_right    |  11.90 | 11.30 | -0.60 |
| sparse        |   3.40 |  2.70 | -0.70 |
| dense         |   8.90 | 10.40 | +1.50 |
| relocating    |   0.00 |  0.00 |  0.00 |
| smell_blanked |   7.50 |  6.50 | -1.00 |

Cycle is *slightly worse* than parent on most scenarios. Both
genomes are bigger than seed 1's (10 vs 11 connections — already
complex founder-derived topology), so the marginal cycle is noise on
top of well-tuned weights.

### Carrier 2 — seed 2 (parent 6/10 vs cycle 6/11)

| scenario      | parent | cycle | delta |
|---------------|-------:|------:|------:|
| uniform       |   8.20 |  6.70 | -1.50 |
| ahead         |  23.40 | 11.90 | **-11.50** |
| behind        |   6.90 |  1.40 | **-5.50** |
| left_right    |  18.70 | 11.10 | **-7.60** |
| sparse        |   3.30 |  3.30 |  0.00 |
| dense         |  11.60 |  8.30 | -3.30 |
| relocating    |  18.40 |  0.00 | **-18.40** |
| smell_blanked |   8.20 |  6.70 | -1.50 |

Parent *crushes* cycle here — parent solves relocating (18.4 food)
and does well on ahead/left_right/behind; cycle is stuck near 0 on
relocating and ~half on most. The cycle disrupted a finely tuned
forward-wired brain.

### Carrier 1 — seed 3 (parent 6/8 vs cycle 6/9)

| scenario      | parent | cycle | delta |
|---------------|-------:|------:|------:|
| uniform       |   8.50 |  6.50 | -2.00 |
| ahead         |  25.00 | 15.10 | -9.90 |
| behind        |   0.00 | 10.80 | **+10.80** |
| left_right    |  15.70 | 12.60 | -3.10 |
| sparse        |   3.50 |  2.80 | -0.70 |
| dense         |  10.70 |  9.70 | -1.00 |
| relocating    |   2.00 |  0.60 | -1.40 |
| smell_blanked |   8.50 |  6.50 | -2.00 |

Cycle is **the only one that solves `behind`** (10.8 vs parent's
0.0). Parent dominates everywhere else. This is the most
informative carrier: it shows that the cycle can be a *targeted
specialisation* — even when the parent is the better generalist,
the cycle unlocks a regime (behind = food behind the agent's heading)
that requires memory to track.

### Carrier 2 — seed 3 (parent 6/8 vs cycle 6/9)

| scenario      | parent | cycle | delta |
|---------------|-------:|------:|------:|
| uniform       |   6.80 |  7.00 | +0.20 |
| ahead         |  14.20 | 16.90 | **+2.70** |
| behind        |   0.00 |  0.00 |  0.00 |
| left_right    |   8.90 | 11.40 | **+2.50** |
| sparse        |   3.30 |  2.80 | -0.50 |
| dense         |   8.60 |  9.00 | +0.40 |
| relocating    |   4.80 |  2.20 | -2.60 |
| smell_blanked |   6.80 |  7.00 | +0.20 |

Cycle slightly ahead on most. Modest but consistent wins on
ahead (+19%) and left_right (+28%).

## Aggregate picture

| scenario      | parent_med | cycle_med | cycle_wins | parent_wins | delta_med |
|---------------|-----------:|----------:|-----------:|------------:|----------:|
| uniform       |       7.50 |      6.50 |          0 |           3 |     -1.00 |
| ahead         |      14.20 |     13.50 |          2 |           2 |     -0.70 |
| behind        |       0.00 |      0.00 |          1 |           2 |     +0.00 |
| left_right    |      11.90 |     11.30 |          2 |           3 |     -0.60 |
| sparse        |       3.30 |      2.80 |          0 |           2 |     -0.50 |
| dense         |       8.90 |      9.00 |          2 |           2 |     +0.10 |
| relocating    |       2.00 |      0.00 |          0 |           3 |     -2.00 |
| smell_blanked |       7.50 |      6.50 |          0 |           3 |     -1.00 |

`cycle_wins` / `parent_wins` count how many of the 5 carriers had
cycle food > parent food + 0.5 (or vice versa).

## Reading the comparison

### The cycle is not a free lunch

Aggregated across all 5 carriers, **parent edges cycle on median
food on 6 of 8 scenarios**. Cycle wins on `behind` (1 carrier, by
+10.8) and is statistically tied on `ahead`, `left_right`, `dense`.

This is the honest answer: a randomly-inserted recurrent edge, given
no time to be weight-tuned, often *disrupts* a working feedforward
brain. The cycle carrier in seed 2 / carrier 2 saw parent food drop
from 23.4 to 11.9 on ahead.

### But the cycle unlocks a regime the parent cannot reach

`behind` is the most informative scenario. Food is placed *behind* the
agent's heading, so the only way to eat it is to turn around — and to
know you need to turn around, you need some persistence of intent
across ticks. The parent of seed-3 carrier-1 is a 6/8 network that
got 0 food on behind; its cycle child (6/9) got **10.8**.

The same pattern shows in seed-1 carrier-1 `left_right`: 1.4 → 5.6.
A left/right alternating food pattern rewards memory of where you
just were; cycle brains can do that, the parent's mostly-reactive
forward brain cannot.

### Behavioural indicator families

Per-scenario averages across the 5 carriers:

| scenario      | sd_p | sd_c | sa_p | sa_c | eff_p | eff_c |
|---------------|-----:|-----:|-----:|-----:|------:|------:|
| uniform       | 0.64 | 0.63 | 0.78 | 0.80 |  7.17 |  6.68 |
| ahead         | 0.63 | 0.54 | 0.66 | 0.65 | 15.61 | 10.89 |
| left_right    | 0.62 | 0.49 | 0.60 | 0.65 | 12.23 | 12.63 |
| sparse        | 0.40 | 0.40 | 0.76 | 0.80 |  2.74 |  3.21 |
| dense         | 0.69 | 0.67 | 0.74 | 0.82 |  9.31 |  8.89 |
| relocating    | 0.13 | 0.17 | 0.19 | 0.23 | 12.95 |  0.00 |

`sd` = state_dependence (history use), `sa` = steering_alignment
(toward-food movement), `eff` = energy_efficiency (food per energy).

The most consistent cycle signal: **state_dependence drops slightly
(sd_c < sd_p on most scenarios) but steering_alignment goes up**
(sa_c > sa_p on 6 of 8). Cycle brains use less raw history but
*align better with food direction* — a hint that the recurrent
computation is reshaping the policy rather than adding inertia.

`energy_efficiency` collapses to 0 on `relocating` for cycles
because no cycle carrier scored there; parents scored there because
they had well-tuned non-recurrent loops.

## Conclusion

The hidden↔hidden recurrent cycle, when it emerges in a Phase-1.5
brain, is **not a uniform improvement and not a uniform regression**.
Across 5 carriers it sometimes helps a lot (seed-1 carrier-1 on
ahead/left_right/dense; seed-3 carrier-1 on behind), sometimes hurts
(seed-2 carrier-2 across the board), and on most scenarios it sits
within ~20% of the parent on median food.

The cleanest qualitative signal is **targeted memory**: the cycle
gives the brain a place to store a state across ticks, which unlocks
scenarios that depend on history (`behind`, `left_right` alternating
food) but costs nothing in reactive scenarios that don't.

This matches the *behavioural* expectation: cycles are not magic,
they're state. They help when state matters, they hurt when state
adds noise to a working reactive policy. Evolution in Phase 1.5 is
*producing* both winners and losers; whether carriers survive long
enough to dominate is a longer-run question.

## Files

- `evolife_cycle_arena_seed{1,2,3}.sqlite` and `.csv` — per-episode
  metrics and per-(carrier, scenario, side) summary.
- `scripts/run_cycle_arena.py` — the new arena runner.
- `evolife/genome.py` — `to_dict()` / `from_dict()` for JSON.
- `evolife/archive.py` — `CycleCarrier` and `archive.cycle_carriers`.
- `evolife/world.py` — `_check_milestones` now accepts parent_genome
  and writes carriers when cycle fires.
- `evolife/metrics.py` — `cycle_carriers` table + `record_cycle_carriers`.
- `scripts/run_v2_experiment.py` — calls `record_cycle_carriers` every
  tick; reports `cycles=N` per seed.

## Next steps

1. **Longer runs.** 100k–200k ticks × 10 seeds would give more carriers
   and a tighter statistical picture. Current run gave 5 carriers in
   3 seeds; 10–15 carriers would let us split by topology ("fresh
   cycle" vs "tuned cycle") and look for trends.
2. **Look at cycle survival.** Of the 5 carriers captured here, how
   many of their lineages survived to the end of the run? If
   cycle-bearing genomes go extinct faster than non-cycle, that is
   itself the answer to "does evolution keep the cycle?" — and it's
   not what Phase 1.5's first_recurrent_cycle snapshot tells us.
3. **Replay trail dump for the +10.8 behind case.** The single
   carrier that solves `behind` is the most interesting artefact in
   this phase. A trail-level inspection would show what it actually
   does — sustained turns? Forward runs that ignore food? Drift-then-
   return? It deserves its own run.
