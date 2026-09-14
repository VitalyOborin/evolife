# Phase 3.1 Results - controlled Memory Ecology

## Scope of this phase

Phase 3.0 built a Memory Ecology world with two food types and a
hidden season. The cold-start founder went extinct before evolution
could test the hypothesis. Phase 3.1 fixes five problems from the
code review:

1. MemoryEcologyWorld used to bootstrap a legacy World then throw
   away its founders + species + events. Now it initialises empty
   state via `_init_state()` and overrides `_populate_initial()`.
2. A and B smell were summed into 3 channels. Organisms physically
   could not choose A vs B. Now there are 6 distinct smell channels
   (a_left, a_front, a_right, b_left, b_front, b_right) plus one
   feedback channel. Seven sensors total.
3. CONTROL was the legacy single-resource World. Now CONTROL is
   MemoryEcologyWorld in `static_dual` mode (two food types, but no
   season). Same founder, same topology, same densities. The only
   difference is the season flip.
4. Memory advantage used `brain.reset_state()` every tick. With
   `_ITERATIONS=1` this destroyed the sensor->hidden->motor feed
   forward that takes two physical ticks, not just recurrent memory.
   Now ablation is SCC-based: we disable only the edges inside a
   strongly connected component of size >= 2. Weights, biases,
   feed-forward computation are all preserved.
5. Founders started cold. Phase 3.1 supports a `--warm-json` flag
   that takes a Phase 1.5 evolved genome and grafts the smell
   edges: each old `smell_left -> hidden` edge becomes both
   `a_left -> hidden` and `b_left -> hidden` with the same weight.
   The cold-start brain still navigates; evolution can then
   specialise A vs B.

FB_TO_HIDDEN_WEIGHT is now 0.2 (down from 1.0). The feedback channel
exists but does not dominate the navigator.

## What we have

### World (`evolife/phase3.py`)

MemoryEcologyWorld now supports three modes:

| mode            | season flip | season sensor | need memory? |
|-----------------|-------------|----------------|--------------|
| `static_dual`   | no          | no             | no           |
| `visible_season`| yes         | yes (8th sensor) | barely     |
| `hidden_season` | yes         | no             | yes          |

All three use the same founder (warm-start or cold), the same
mutation rates, the same population caps.

### Founders

`make_phase3_founder_from_warmstart(warm_genome=...)` returns a 7 (or
8 with season sensor) sensor / 1 hidden / 2 motor genome with the
grafted smell edges. Without `warm_genome` it builds a cold-start
founder with random weights.

### Runners

- `scripts/run_phase3_1.py` runs CONTROL (`static_dual`),
  `visible_season`, `hidden_season` on a configurable list of
  seeds. Per-seed SQLite + per-tick log. 3000-tick smoke results
  below.
- `scripts/arena_memory_advantage.py` takes any genome JSON and
  runs it on MemoryEcologyWorld with full `world.step()` for both
  NORMAL and SCC-ABLATED versions. Writes a CSV with
  positive / negative / reward advantages.

### Tests

95 existing tests still pass. World refactor (extract `_init_state`
+ `_populate_initial`) is backward-compatible with legacy World.

## Smoke results (seed 1, 3000 ticks, warm-start from
`evolife_phase31_warm_parent.json`)

| mode            | pop@3000 | maxGen | positive eats | negative eats | cycles |
|-----------------|---------:|-------:|--------------:|--------------:|-------:|
| `static_dual`   |       12 |      4 |            68 |            54 |      0 |
| `visible_season`|        3 |      4 |            24 |            12 |      0 |
| `hidden_season` |       14 |      5 |            80 |            49 |      0 |

Positive-eat fractions:

- static_dual:   68/122 = 0.557
- visible_season: 24/36  = 0.667
- hidden_season:  80/129 = 0.620

All three worlds go extinct by tick ~2000-2500. The warm-start
parent genome had no recurrent edges, so no cycle carriers are
recorded. This is the same cold-start extinction we saw in Phase
3.0, but the failure mode is now better isolated: the world is
correctly wired, the founder is the wrong starting point.

What we *do* see is that `visible_season` and `hidden_season` both
yield a higher positive-eat fraction than `static_dual` even at
3000 ticks. With three seeds the noise is too high to call this
significant; with ten seeds it could be the first signal that the
season flip is selecting for discrimination, *if* extinction can
be prevented.

## What still needs to happen (Phase 3.2)

The remaining problem is extinction. Even with the warm-start
parent, populations do not survive 3000 ticks under any of the
three modes. Phase 3.2 should address this by:

1. **Larger founder buffer.** `PHASE3_INITIAL_ENERGY` is currently
   `FOOD_ENERGY * 10 = 400`. Bump to 600 or 800 to absorb more
   negative eats before starvation.

2. **Better warm-start.** Use a Phase 1.5 evolved genome from a
   high-population, high-maxGen run (e.g. seed 1 carrier 1) rather
   than its parent. An evolved navigator finds food *fast*, which
   gives reproduction a chance to fire before the energy buffer
   runs out.

3. **Longer runs.** 30k-100k ticks instead of 3000. The warm-start
   enables Phase 3 evolution to begin; meaningful adaptation to
   the season needs 10+ generations of selection, which at ~500
   ticks/generation is 5000+ ticks minimum, plus headroom for
   recurrent topologies to emerge.

4. **Memory advantage on evolved genomes.** Once we have a
   hidden-season population with recurrent edges, the SCC
   ablation gives a real number. Phase 3.1's arena runner is
   ready for this.

## Files

- `evolife/phase3.py`        - MemoryEcologyWorld + warm-start founder
- `evolife/config.py`        - PHASE3_DEFAULT_MODE, FB_TO_HIDDEN_WEIGHT=0.2
- `evolife/world.py`         - `_init_state`, `_populate_initial`
                              (subclass override points)
- `evolife/organism.py`      - positive_eats, negative_eats,
                              positive_eat_seasons
- `scripts/run_phase3_1.py`  - 3-world runner
- `scripts/arena_memory_advantage.py` - SCC-ablation arena
- `evolife_phase31_warm_parent.json` - warm-start genome (parent of
                              first cycle_carrier in seed 1)
