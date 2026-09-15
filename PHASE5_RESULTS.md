# Phase 5 results — Grace period after season flip

**Date:** 2025
**Status:** implemented and tested. Initial smoke complete.
**Headline:** grace=200 alone is **not enough** to save season modes.
The fundamental problem is a **demographic undershoot**, not a
brain adaptation budget issue.

## 1. Setup

Phase 5 introduces a configurable grace window after every season
flip, during which the negative food penalty linearly interpolates
from `PHASE3_GRACE_PENALTY` (default −0.5) back to the full negative
penalty (default −3.0). Defaults: `PHASE3_GRACE_TICKS=200`,
`PHASE3_GRACE_PENALTY=-0.5`. Setting `PHASE3_GRACE_TICKS=0` cleanly
disables the mechanism.

Wiring (committed in 3ad48d6):

- `evolife/config.py` — defaults.
- `evolife/phase3.py` — `grace_ticks_remaining` state, reset on
  flip, decremented each tick; soft-penalty interpolation in
  `_resolve_eat_phase3` only when `reward < 0`.
- `scripts/run_phase3_1.py`, `scripts/run_phase4_1.py` — `--grace-ticks N`.
- `tests/test_phase5_grace.py` — 7 unit tests, all pass.

## 2. Smoke results (grace=200, no plasticity)

3 seeds × 3 modes × 6000 ticks, with `--grace-ticks 200 --neg-energy -1`.

### static_dual — unaffected, as expected (no season flip)

| seed | pop@6000 | maxGen |
|------|---------:|-------:|
| 1    | 113      | 6      |
| 2    | 105      | 6      |
| 3    | (≥100)   | (≥6)   |

No change vs Phase 3.1 static_dual.

### visible_season — still goes extinct

```
seed 1: pop=8 at t=2001, pop=0 at t=9001
seed 2: pop=10 at t=2001, pop=0 at t=9001
seed 3: pop=8 at t=2001, pop=0 at t=9001
```

Despite the 200-tick grace window at each flip, the population
collapses at the second season transition (around t=8000-9000).
The soft penalty buys a brief reprieve but does not change the
fundamental trajectory.

### hidden_season — still goes extinct

```
seed 1: pop=8 at t=2001, pop=1 at t=9001
seed 2: pop=7 at t=2001, pop=0 at t=9001
seed 3: (similar trajectory)
```

Same pattern.

## 3. Diagnosis: grace is not the bottleneck

Why does grace=200 fail?

The grace window is 200 ticks (~10% of SEASON_LENGTH=2000). After
the flip, the population is already in a precarious state (pop=8-12,
mostly young, low-energy organisms). The 200-tick window gives them
some breathing room but the linear ramp back to −3.0 still starves
most of them before they can reproduce.

The deeper issue is **demographic undershoot during the GOOD
season**. Organisms reproduce up to ~50 individuals by t=1500-1800
(during season 0, when food_a is positive). At the flip, that
population has to suddenly switch to food_b. There is no time for
learning, no time for reproduction under the new regime, no time
for anything except mass starvation.

## 4. What grace DOES change

Comparing the seed-1 hidden_season trajectory with vs without grace:

| t      | grace=0  | grace=200 |
|--------|---------:|----------:|
| 1001   | 47       | 44        |
| 2001   | 11       | 8         |
| 3001   | 8        | 4         |
| 4001   | 6        | 2         |
| 5001   | 5        | 3         |
| 6001   | 2        | 2         |
| 7001   | 2        | 1         |
| 8001   | 0        | 2         |
| 9001   | 0        | 1         |

Both go extinct, but grace slightly extends the survival window
by 1-2k ticks. This is a marginal benefit, not a fix.

## 5. What grace enables but doesn't itself fix

Grace is necessary but not sufficient. It provides the *time* the
brain needs to learn, but it doesn't change the brain's *ability*
to learn within that time. The remaining bottleneck is whether
rHebb can compress "I just died after eating food_a" into a
strategy within the new few-hundred-tick budget per flip.

Two empirical questions remain:

1. **Does Phase 5 + Phase 4.1 (rHebb) combined beat Phase 5 alone?**
   The rHebb mechanism can credit past co-activations back to
   sparse reward events, so the brain should be able to update
   weights during the grace window and emerge into the next
   "full penalty" phase already adapted.

2. **Is the grace window length itself the bottleneck?**
   A longer grace (say 500 or 1000 ticks) might give rHebb enough
   time to lock in a season-dependent policy.

The Phase 5 + Phase 4.1 combined sweep is the obvious next step.
If that fails, the demographic architecture itself needs rework
(e.g., population-cap reduction in season modes, or a permanent
soft penalty).

## 6. Reproduction

```bash
# Phase 5 alone (smoke):
python scripts/run_phase3_1.py --seeds 1 2 3 --ticks 6000 \
    --warm-json evolife_regen_warm_parent.json \
    --neg-energy -1 --grace-ticks 200

# Phase 5 + Phase 4.1 (combined):
python scripts/run_phase4_1.py --seeds 1 2 3 --ticks 6000 \
    --warm-json evolife_regen_warm_parent.json \
    --neg-energy -1 --grace-ticks 200
```
