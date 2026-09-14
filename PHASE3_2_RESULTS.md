# Phase 3.2 — Demographic Stabilisation (in progress)

Phase 3 = "Memory Ecology": three controlled worlds that differ only in
their ecology, identical founders/mutation/density, to test whether
evolution grows recurrence to handle a season flip whose reward sign
is hidden.

```
static_dual     : A and B both always +25.    Control: is recurrence
                  selected just by having two distinct food types?
hidden_season   : A+25 B-3 in season 0; A-3 B+25 in season 1;
                  brain sees no season sensor, only post-eat feedback.
                  Hypothesis test.
```

Phase 3.1 (commit db8e68f): 6 distinct A/B smell channels, FB_TO_HIDDEN_WEIGHT=0.2,
warm-start founder via `make_phase3_founder_from_warmstart`.

Phase 3.2 (commit d65fd1d): static_dual fix in `_resolve_eat_phase3` —
season-dependent rewards were applied even in static_dual mode.

Phase 3.2 followup (commit 33d5a4d): one-eat-per-tick (was eating both
A and B in a single tick when both within EAT_RADIUS, double-feeding).
Comment fix: `+12` → `FOOD_*_POSITIVE_ENERGY`.

## What Phase 3.2 smoke actually shows

The Phase 1.5 founder genome (`evolife_regen_warm_parent.json`, 6 nodes,
5 conns, pure feedforward) was warm-started into Phase 3 hidden_season
under several demographic overrides. Population curve + cumulative
positive/negative eats were logged every 200 ticks. See
`_smoke_phase32_curve.py` and `_smoke_phase32_grid.py`.

### Population overshoot wave (initial_e=400, repro_t=64, pop=50, season=2000)

The original Phase 3 founder started with PHASE3_INITIAL_ENERGY=400.
With 50 founders and reproduction threshold 64, it exploded:

```
t=  101 pop=500  pos=813  neg=697   (cap hit)
t=  401 pop=304  pos=9521 neg=5082  (die-off begins)
t=  601 pop=107
t=  701 pop=69
t=2001 pop=0 (extinct before first season flip)
```

A 50-fold over-reproduction in 100 ticks is the overshoot wave. By the
time the season would flip at t=2000, pop=0.

### Demographic grid (warm-start, 4000 ticks, seed 1)

| Config (init_e, repro, pop, season) | static_dual | hidden_season |
|---|---:|---:|
| (80, 64, 50, 2000)   baseline       | 3   | 0 |
| (80, 64, 50, 8000)   low_energy     | 3   | 0 |
| (80, 128, 50, 8000)  high_threshold | 20  | 0 |
| (80, 64, 20, 8000)   low_pop        | 2   | 0 |

Only `high_threshold` (init_e=80, repro_t=128) stabilises static_dual
at pop=20 — founder reproduces ~3×, settles well below the 500 cap.

**hidden_season extinct under ALL configurations.** The founder's net
reward rate halves once it eats the wrong food (-3 instead of +25),
which collapses the demographic balance.

### Reward asymmetry is the bottleneck

The reward magnitudes are asymmetric on purpose (the negative energy is
the cost of being wrong about the season), but the **founder has no
recurrence and no season sensor** — it eats both A and B equally
(~50/50 split). Net reward per eat in season0:

```
0.5 * 25 + 0.5 * -3 = 11  (positive net, but only half of static_dual's 25)
```

So the founder can sustain roughly half the population of static_dual.
At static_dual's pop=20 equilibrium, hidden_season would need to
support pop≈10. But the founder can't even get there — it goes extinct
during the initial wave before equilibrium.

## Next: reduce reward asymmetry

The reward magnitudes are a *world design choice*, not a fitness
function. Lowering the negative energy from -3 → -1 (or 0) makes the
founder's net rate close to static_dual's, while still making the
right choice strictly better. The hypothesis test still works:
recurrence becomes valuable as soon as the *cost* of being wrong is
nonzero.

`_smoke_phase32_neg.py` runs the high_threshold config with
neg_energy ∈ {-3, -1, 0} on both static_dual and hidden_season for
8000 ticks.

## Open question

Is "lower the negative energy" the right move? Alternative: keep
the asymmetry but **dampen** it differently — e.g. reduce positive
energy too. But lowering only the negative keeps the *signal* of "A
vs B matters" loud without making it lethal.

If hidden_season stabilises at neg_energy=-1, we proceed to the
3-world × N-seed × long-tick sweep. Otherwise: rethink the reward
design or the founder structure.

## Phase 3.2 outcome: stable demography (commit 2a0bade)

The demographic wave (initial pop 50 → cap 500 → crash to 0) was
the proximate cause of extinction. The reward asymmetry is real but
not the bottleneck — the founder dies during the wave, before the
season flip even matters.

Fix: a Phase 3 specific config block that matches Phase 1.5's
demographic regime, plus more food headroom:

```
PHASE3_INITIAL_ENERGY          = FOOD_ENERGY * 2   # 80, was 400
PHASE3_REPRODUCTION_THRESHOLD  = 128.0             # was 64
PHASE3_FOOD_TARGET             = 400               # was 200
PHASE3_FOOD_A_NEGATIVE_ENERGY  = -3.0              # unchanged
PHASE3_FOOD_B_NEGATIVE_ENERGY  = -3.0              # unchanged
```

MemoryEcologyWorld overrides `_reproduce` to honour the Phase 3
threshold (the global `REPRODUCTION_THRESHOLD` is still 64 for the
legacy World and for any future regression tests).

### Smoke results (warm-start, 4000 ticks, seed 1)

```
hidden_season  pop@4000=2   maxGen=0  pos=180k neg=138k  (sustainable)
static_dual    pop@4000=98  maxGen=5  pos=1.8M neg=0      (gen 5 evolved)
```

hidden_season pop curve:

```
t=201   pop=50 (no wave — founder can't reproduce fast enough)
t=1001  pop=44
t=2001  pop=8   (just before season flip)
t=2801  pop=3   (after flip — survived)
t=4000  pop=2   (settled)
```

Small but non-extinct. The founder lineage (gen=0) survives the
season flip. With 8000+ tick runs and more founder numbers we should
see evolution actually do something.

static_dual is healthy: 50 founders → 70 by t=1001 → 98 by t=4000.
Generation 5 has appeared, evolution is operating on top of the
founder.

### What this means

The 3 controlled worlds are now demographically viable. Next step is
a proper 3-world × N-seed × 30k-tick sweep with the arena metrics
already implemented in `scripts/run_phase3_1.py` and
`scripts/arena_memory_advantage.py`.

The hypothesis test still has full force: in hidden_season the
season flips at t=2000 (and every 2000 ticks after) and the founder
has no way to know which season it's in — only the post-eat feedback
signal. If evolution grows recurrence in hidden_season but not in
static_dual or visible_season, that's the signal Phase 3 was built
to detect.