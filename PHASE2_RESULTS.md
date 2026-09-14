# Phase 2 Results — Behavioral Arena on evolving genomes (seed 4)

## What this phase asked

Phase 1.5 showed that **3/5 seeds evolve a hidden↔hidden recurrent cycle
within 50k ticks**. The natural next question is *what does the cycle
do for the organism?* The Behavioral Arena (c68bcb8) was built exactly
for this: keep the genome frozen, run it on a battery of controlled
scenarios, aggregate behavioural indicators, and compare ancestor vs
descendant.

## Why the original plan didn't fire

The original plan was: take a seed where `first_recurrent_cycle`
fired (seeds 2/3/4), pull a pre-cycle and a post-cycle frozen genome
from `world.organisms`, run them through the arena. The first attempt
(seed 4, 50k evo-ticks, generations 0..237) returned the same genome
for every generation ≥ 75 — only five distinct sampled genomes out of
eleven requested. The reason: at the end of a 50k run, the living
population is concentrated in late generations; the founder (gen 0) is
long dead and intermediate generations have no survivors.

We re-ran with evo-ticks=15000 so that the population still carries
the earlier generations alive. This gave four distinct frozen genomes
at generations 20, 30, 50, 75 (max generation reached was 34 at that
point — generations 50/75 fall back to the highest surviving one).

## What we actually compared

The 15k-tick Arena run produced four distinct brain shapes:

| gen | nodes | connections | shape description                       |
|----:|------:|------------:|----------------------------------------|
|  20 |     6 |           6 | founder + 1 add_connection (no cycle) |
|  30 |     6 |           5 | founder (no structural mutation kept)  |
|  50 |     6 |           5 | founder                                |
|  75 |     6 |           5 | founder                                |

None of the sampled genomes in this run have a hidden↔hidden cycle. The
archive shows that seed 4 produced `first_recurrent_cycle @ tick 7148`,
but by tick 15000 the lineage carrying the cycle has died out and
later lineages reverted to founder topology (or never picked up a
cycle in the first place — we cannot distinguish without lineage
tracking).

So this phase did **not** end up comparing "with cycle" vs "without
cycle". It compared two founder topologies:

- **gen 20 (6 nodes, 6 connections)**: founder (3 sensor → 1 hidden →
  2 motor, 5 connections) plus one `add_connection` event — one extra
  weight landed somewhere in the network. We don't know exactly where
  without diffing lineage, but it is structurally distinct from the
  founder and from later generations.
- **gen 30 (6 nodes, 5 connections)**: textbook founder.

This is still a useful comparison. It's the closest thing to "did the
one structural mutation that survived into the population help?".

## Arena results — per scenario, per generation

(`food` = mean over 5 arena-seeds of `food_eaten`, `state_dep` =
state_dependence_mean, `eff` = energy_efficiency_mean)

### gen 20 (founder + 1 add_connection)

| scenario      | food | state_dep | eff  |
|---------------|-----:|----------:|-----:|
| uniform       |  2.4 |     0.557 |  3.55 |
| ahead         |  0.0 |     0.107 |  0.00 |
| behind        |  0.0 |     0.117 |  0.00 |
| left_right    |  0.0 |     0.128 |  0.00 |
| sparse        |  1.2 |     0.081 |  1.90 |
| dense         |  3.0 |     0.510 |  4.20 |
| relocating    |  0.0 |     0.005 |  0.00 |
| smell_blanked |  2.4 |     0.557 |  3.55 |

### gen 30 (founder)

| scenario      | food | state_dep | eff  |
|---------------|-----:|----------:|-----:|
| uniform       |  2.2 |     0.008 |  3.00 |
| ahead         |  4.8 |     0.041 |  6.91 |
| behind        |  0.0 |     0.000 |  0.00 |
| left_right    |  4.8 |     0.053 |  6.91 |
| sparse        |  0.2 |     0.000 |  0.30 |
| dense         |  7.0 |     0.122 |  8.97 |
| relocating    |  4.0 |     0.000 |  6.47 |
| smell_blanked |  2.2 |     0.008 |  3.00 |

### gen 50/75 (founder)

| scenario      | food | state_dep | eff  |
|---------------|-----:|----------:|-----:|
| uniform       |  1.6 |     0.211 |  1.96 |
| ahead         |  5.6 |     0.041 |  8.29 |
| left_right    |  5.4 |     0.039 |  7.51 |
| sparse        |  0.8 |     0.056 |  1.19 |
| dense         |  5.6 |     0.320 |  6.28 |
| relocating    |  4.6 |     0.004 |  7.30 |
| smell_blanked |  1.6 |     0.211 |  1.96 |

## Reading the comparison

### 1. The +1-connection variant (gen 20) is *less* efficient than founder (gen 30)

This is the surprise. Adding one extra connection actually hurts
behavioural indicators on most scenarios:

- **dense**: gen20 food=3.0, eff=4.20 vs gen30 food=7.0, eff=8.97
- **ahead**: gen20 food=0, eff=0 vs gen30 food=4.8, eff=6.91
- **left_right**: gen20 food=0, eff=0 vs gen30 food=4.8, eff=6.91

The +1 connection might be putting noise into the network that the
1-tick-lagged hidden layer cannot compensate for.

### 2. But state_dependence (history signal) is highest in gen 20

- **uniform**: gen20 state_dep=0.557 vs gen30 state_dep=0.008
- **dense**: gen20 state_dep=0.510 vs gen30 state_dep=0.122

So the +1-connection variant *does* carry signal from one tick to the
next — but it uses that signal badly: it sits longer in `move`
state, which costs energy, and gets worse food. Without knowing
exactly where the connection landed, we can't say whether it's a
self-loop on hidden or a stray motor→hidden edge. Either way, more
state is not automatically a fitness gain.

### 3. Founder at gen 30 vs gen 50/75 — pure behavioural drift

| scenario      | gen30 food | gen50 food | Δ |
|---------------|-----------:|-----------:|--:|
| uniform       |        2.2 |        1.6 | -0.6 |
| ahead         |        4.8 |        5.6 | +0.8 |
| dense         |        7.0 |        5.6 | -1.4 |
| smell_blanked |        2.2 |        1.6 | -0.6 |

Same topology (6/5 founder), but weights and biases drifted through
~15k ticks of weight/bias mutation. Net effect: roughly equivalent
across scenarios, with founder-tuned gen 30 doing better on dense
food patches and gen 50/75 doing better on rare-food scenarios. This
is the typical "founder → later drift" picture: no architecture
change, just knob-twiddling.

### 4. The unreachable comparison: with cycle vs without

The original hypothesis — that a recurrent hidden↔hidden cycle is
behaviourally distinct from feedforward founder — remains untested
by direct arena comparison. The cycle appeared in the archive once
(seed 4, tick 7148) but its lineage did not survive to tick 15000.
To test the hypothesis we need either:

1. **Lineage-aware snapshotting**: when `first_recurrent_cycle`
   fires, freeze that genome (and its parent) for arena use.
2. **A founder with N_HIDDEN=2** so the cycle is one add_connection
   event away from birth, increasing the chance the cycle survives.
3. **Replay across many seeds** until at least one survivor with a
   cycle can be found.

## Recommendation

The Phase 2 question — *does a hidden↔hidden cycle help?* — is the
right next question, but this run did not answer it. The most efficient
path to an answer is option (1): freeze the cycle carrier at the
moment of `first_recurrent_cycle` and arena-test it against its
parent (which, by definition, has no cycle).

That requires a small change to the metrics pipeline: when a
`FIRST_RECURRENT_CYCLE` archive event fires, write the cycle carrier's
genome to a dedicated SQLite table. The phase 1.5 metrics already
write `org_id` and `genome_hash` into the archive payload — adding a
serialised genome alongside would close the loop.

## Files

- `evolife_arena_seed4_short.sqlite` and `.csv` — 4 sampled generations
  × 8 scenarios × 5 arena-seeds × 200 arena-ticks.
- `evolife_arena_replay_seed4_gen20_gen30.json` — trail + food payload
  for the gen 20 vs gen 30 side-by-side comparison.
- `evolife_arena_replay_log.txt` — captured stdout of the replay.
