# Phase 1 Results — full 5x50k sweep

## Setup

Same as Phase 0 except:

| param                       | Phase 0 | Phase 1 |
|-----------------------------|--------:|--------:|
| ADD_NODE_RATE               |  0.002  |  0.01   |
| ADD_CONNECTION_RATE         |  0.005  |  0.02   |
| TOGGLE_CONNECTION_RATE      |  0.002  |  0.005  |

Per-seed results (read from `archive_milestones` table in each
per-seed SQLite, written incrementally by the new
`Metrics.record_archive_milestones` so they survive any timeout):

| seed | pop@50k | maxGen | eats  | repros | species | est | first_hidden @ tick | first_recurrent |
|-----:|--------:|-------:|------:|-------:|--------:|----:|---------------------:|----------------:|
| 1    |    249  |  156   | 33742 | 18563  |       5 |   1 |                    1 |             0 |
| 2    |    262  |  113   | 34048 | 18898  |       5 |   4 |                 1452 |             0 |
| 3    |    262  |  114   | 34385 | 19051  |       4 |   1 |                    1 |             0 |
| 4    |    118  |  190   | 32055 | 17808  |       4 |   2 |                    1 |             0 |
| 5    |    154  |  188   | 31803 | 17686  |       2 |   1 |                 2005 |             0 |

**All 5 seeds stayed alive.** No extinction in this run, unlike
Phase 0 where 3/5 went extinct early.

## Archive milestones — total

```
first_hidden_node events:    5  (1 per seed)
first_recurrent_cycle:       0  (5/5 seeds)
max_brain_nodes events:     ~4-5 per seed (max 5 nodes/connections)
max_generation_reached:    113-190 per seed
max_lineage_size:          169-284 per seed
```

## Interpretation

### Hidden nodes appear, but cycles don't

- **5/5 seeds** saw at least one `first_hidden_node` event. This means
  structural mutations are firing and producing hidden nodes, sometimes
  very early (ticks 1–2005).
- **0/5 seeds** saw a `first_recurrent_cycle`. Even with rates bumped
  5×, two hidden nodes never connected in a bidirectional cycle within
  50k ticks.

### Why cycles are still hard to hit

For a hidden↔hidden cycle to fire, two hidden nodes need to exist AND
have a connection pair (A→B and B→A). With `N_HIDDEN=0` in the founder,
the first hidden node must be added by `add_node`. Then a second
hidden must also be added by `add_node`. Then two connections between
them must be added — and they have to land in opposite directions.

With our rates:
- ~100 births × `ADD_NODE_RATE=0.01` ≈ 1 add-node per seed per ~50k
  ticks. So *on average* we expect ~1 add-node event per seed, not
  the multiple events required to build two distinct hidden nodes
  plus their connections.
- We do see `first_hidden_node` firing 5/5 — but it captures the very
  first add-node mutation, which is a single hidden. There is no
  archive milestone for "second hidden" so we cannot tell from the
  archive alone whether any seed got past the first hidden.

### Population and reproduction are healthy

Pop size is 118–262 across seeds. Eats are 31k–34k. Repro 17k–19k. So
the food-seeking behavior is sustained and organisms are reproducing
plenty — there's enough evolutionary pressure for structural mutations
to be sampled. The bottleneck is the *combination* of structural events
needed to form a cycle, not the total number of births.

## Recommendation

To push past this without further rate increases (which would start
harming fitness), we have two paths:

1. **Allow hidden nodes in the founder.** With `N_HIDDEN=0`, the world
   starts with no hidden nodes and we must build them. Starting with
   `N_HIDDEN=1` or `N_HIDDEN=2` would mean we only need one (or zero)
   add-node events to reach the cycle configuration. This makes
   "evolution produces cycles" testable rather than a question of
   whether evolution will build the substrate at all.

2. **Increase `ADD_CONNECTION_RATE` only.** The first hidden node
   does appear, but it doesn't get re-wired into a cycle. More
   add-connection events would give more chances for the wiring.

3. **Just run longer.** 100k–200k ticks per seed instead of 50k. The
   first cycles would eventually emerge statistically, but at the
   current rate this would take very long.

The cleanest next experiment is path (1): a controlled Phase 1.5 that
introduces `N_HIDDEN=1` or `2` from the start so the *combination*
question is decoupled from the *substrate* question.
