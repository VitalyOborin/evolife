# Phase 5 — Grace period after season flip

**Date:** 2025
**Status:** implemented and tested. Sweep pending.

## Problem

Phase 4 / 4.1 sweeps both showed visible_season and hidden_season
populations crashing at the first season flip (~t=2000) and going
extinct by t=5000-15000. Investigation traced the extinction to
**demographic**, not brain, dynamics:

1. Organisms reproduce up to PHASE3_FOOD_TARGET/2 on the GOOD food
   type (no season-flip awareness).
2. At the season flip, all of that food type becomes NEGATIVE.
3. Population overshoots food budget → mass starvation → extinction.

The plasticity mechanism (Phase 4 / 4.1) is operating correctly and
even shows growing `rec_frac` in static_dual (seed 3, 0.01→0.07), but
the brain has no time to shape a viable strategy before the
demographic wave kills everyone.

## Design

**Linear grace window after each season flip.** For the first
`PHASE3_GRACE_TICKS` ticks after a flip, the negative food penalty
interpolates linearly from `PHASE3_GRACE_PENALTY` (at the flip) back
to the full `PHASE3_FOOD_*_NEGATIVE_ENERGY` (at the end of the window):

```
t = grace_ticks_remaining / PHASE3_GRACE_TICKS       # 1.0 at flip, 0.0 at end
penalty = PHASE3_GRACE_PENALTY * t + full_penalty * (1 - t)
```

After the window closes, the system behaves exactly as Phase 3 / 4.

**Defaults:**

- `PHASE3_GRACE_TICKS = 200` (10% of SEASON_LENGTH=2000)
- `PHASE3_GRACE_PENALTY = -0.5`
- `PHASE3_GRACE_TICKS=0` disables the mechanism entirely (Phase 3/4
  backward compat).

**Biological motivation:** in nature, an organism encountering a
newly-toxic food source (e.g. a seasonal berry) needs multiple
attempts to update its internal model. The grace window is
analogous to that re-calibration time. It preserves selection
pressure (the full penalty eventually returns) but lets the brain
survive long enough to discover the new sign of reward.

## Implementation

- `evolife/config.py` — `PHASE3_GRACE_TICKS=200`,
  `PHASE3_GRACE_PENALTY=-0.5`.
- `evolife/phase3.py` — `grace_ticks_remaining` state, reset on
  season flip, decremented each tick; soft-penalty interpolation in
  `_resolve_eat_phase3` only when `reward < 0`.
- `scripts/run_phase3_1.py` and `scripts/run_phase4_1.py` —
  `--grace-ticks N` CLI override.

## Tests

7 unit tests in `tests/test_phase5_grace.py`:

- defaults match spec (200 ticks / -0.5 penalty)
- grace state starts at 0 (no grace before first flip)
- `--grace-ticks=0` cleanly disables the mechanism
- grace resets to GRACE_TICKS on each flip
- grace decrements each tick
- linear interpolation at endpoints (grace=GRACE → soft, grace=0 → hard, halfway → mid)
- positive food is unaffected by grace

All 7 pass. Full suite: 108 tests passing (without GPU test).

## Expected outcome

- **static_dual** — unaffected (no season flip).
- **visible_season** — should survive the first flip and reach a
  population of 10–30 by t=5000 (vs ~0 without grace).
- **hidden_season** — same as visible_season, plus the rHebb rule
  (if combined with Phase 4.1) finally has time to discover which
  food type is currently positive via the intake_feedback signal.

If grace alone is not enough to keep hidden_season alive long-term,
the next architectural step is a softer negative penalty at all
times (PHASE3_GRACE_PENALTY = full penalty) or a permanently
asymmetric cost structure.

## Reproduction

```bash
# Phase 5 alone (no plasticity, but grace saves the population):
python scripts/run_phase3_1.py --seeds 1 2 3 --ticks 6000 \
    --warm-json evolife_regen_warm_parent.json \
    --neg-energy -1 --grace-ticks 200

# Phase 5 + Phase 4.1 (the full stack):
python scripts/run_phase4_1.py --seeds 1 2 3 --ticks 6000 \
    --warm-json evolife_regen_warm_parent.json \
    --neg-energy -1 --grace-ticks 200
```
