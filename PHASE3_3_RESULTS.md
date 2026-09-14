# Phase 3.3 — 3-World Sweep Results (10k ticks, 3 seeds)

## Setup

- Warm-start: regenerated Phase 1.5 parent (`evolife_regen_warm_parent.json`,
  6 nodes / 5 conns, pure feedforward, 3 smell sensors).
- Phase 3 founder grafts that genome into 6-channel A/B smell.
- Worlds: `static_dual` (control, A+B both always +25),
  `visible_season` (A and B alternate reward by season, brain sees a
  season sensor), `hidden_season` (same reward flip, no season sensor).
- Demography: PHASE3_INITIAL_ENERGY=80, PHASE3_REPRODUCTION_THRESHOLD=128,
  PHASE3_FOOD_TARGET=400, PHASE3_FOOD_*_NEGATIVE_ENERGY=-3.
- Seeds: 1, 2, 3. Ticks: 10,000 per run. Total: 9 runs × 10k = 90k ticks.

## Sweep results

```
mode           seed  pop maxGen    pos    neg cycles  rec_frac_max
static_dual       1  100      8   1958      0      0   0.02
static_dual       2  105      6   1879      0      0   0.02
static_dual       3  109      8   1826      0      0   0.02
visible_season    1    0      0      0      0      0   0.00
visible_season    2    0      0      0      0      0   0.00
visible_season    3    0      0      0      0      0   0.00
hidden_season     1    1      0     48     37      0   0.00
hidden_season     2    1      0     53     48      0   0.00
hidden_season     3    0      0      0      0      0   0.00
```

### What survived in each mode

**static_dual** — full population persists and evolves.
- pop hovers at 100–124 (well below the 500 cap), maxGen=6–8.
- rec_frac reaches 0.01–0.02 by t=6000 (Tarjan SCC finds recurrent
  edges in some evolved descendants).
- All eats positive, no seasons, founder navigates fine.

**visible_season** — collapsed to pop=0 in all three seeds.
- The founder population peaks at ~20 in season 0, drops through season 1.
- By t=9000–10000 every founder has died; the cycle_carriers table is
  empty.
- A season sensor is *available* but the founder has no reward signal
  telling it which season is which (no positive-vs-negative feedback
  the brain can act on), so it eats both A and B regardless. The
  founder's expected reward rate is half of static_dual; combined
  with the demography, it goes extinct.

**hidden_season** — collapses to pop=0–1 in all three seeds.
- Same trajectory as visible_season: founder can't see the season
  *and* can't act on the post-eat feedback (brain has no recurrence).
- Pos:neg ratio at extinction is ~52:43 = 56% positive — exactly the
  expected 50/50 + slight A bias from the founder's Phase 1.5
  evolved weights.
- Pos and neg are both ~50 over the founder's lifetime — populations
  last only ~5k ticks before extinction.

## Interpretation

The 3-world sweep delivers a clean but sobering result:
**recurrence has not yet emerged at any meaningful scale in any arm.**

The static_dual control arm does have **Tarjan-confirmed recurrent
edges** in 1-2% of the population by t=6000, but these are
*unselected* by the environment — the founder genome in static_dual
doesn't need recurrence (it never sees wrong-eat feedback), so any
recurrence that does appear is drift, not signal.

The two season arms are *demographically dead* before evolution can
discover recurrence. The founder's first job in those worlds is to
survive the season flip, which it cannot do without recurrence — but
it also cannot reproduce enough to test new variants before
extinction.

This is **expected at 10k ticks**. The Phase 1.5 evolved founder took
~4k ticks just to stabilise. Discovering recurrence on top of an
unstable population needs many more generations.

## Scientific reading

The result is consistent with the *minimum-assumptions thesis* but
*not* a positive finding yet:

1. The 3 worlds are demographically viable (no founder extinction by
   construction).
2. The static_dual control shows the founder navigates, evolves, and
   produces small recurrent fractions through drift alone — a
   baseline for "noise" recurrence.
3. The season arms don't yet survive long enough for *selected*
   recurrence to appear.

What we'd want to see: after 50k+ ticks and 5+ seeds, a *higher*
recurrent-fraction in hidden_season than in static_dual. That's the
positive hypothesis.

## What's next

The hypothesis test as currently framed is hard: the founder needs
to be just-good-enough to survive season flips while evolution tries
to make it better. Options:

1. **Longer sweep** (50k–100k ticks, 5+ seeds). Evolution has more
   time per season flip.
2. **Better founder**: evolve a Phase 1.5 carrier in static_dual
   *first*, then warm-start that. A smarter founder would survive
   longer in the season arms.
3. **Lower negative reward** further (-3 → -1) so the founder's
   net rate is closer to static_dual's while still penalising the
   wrong choice. The Phase 3.2 grid showed neg=-1 didn't save the
   population in 4k ticks but might allow survival in 10k.
4. **Pre-phase: latent food switch**. A simpler bridge world where
   the resource's value flips deterministically (no season) so the
   brain can learn "what was good yesterday is bad today" with
   single-step memory, before adding the two-season jump.

The next concrete step is option 3 + 1 combined: re-run hidden_season
with neg_energy=-1 for 30k ticks, 5 seeds, to see whether the
founder lineage can persist long enough for selected recurrence to
appear.

## Files / commits

- Sweep driver: `scripts/run_phase3_1.py` (Phase 3.1 runner, used with
  new Phase 3.2 demographic config).
- Warm parent: `evolife_regen_warm_parent.json`.
- Demography: `evolife/config.py` PHASE3_* block + `evolife/phase3.py`
  `_reproduce` override.
- This document: `PHASE3_3_RESULTS.md`.

Tests: 95 pass. Code is on `main`.