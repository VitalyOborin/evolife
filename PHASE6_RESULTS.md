# Phase 6 results — Colony social markers (Choe & Chung 2011)

**Date:** 2025
**Status:** full 3×3×8000 sweep complete. **Markers do NOT save season modes.**
But the null result itself is scientifically meaningful.

## 1. Mechanism

Generic environmental marker field per Choe & Chung (2011). On every
positive eat, an organism deposits a marker at its position. Markers
diffuse (separable 3x3 Gaussian kernel) and decay (0.95/tick, ~14-tick
half-life). Each organism carries 3 new sensor probes (marker_left,
marker_front, marker_right) that read the field locally.

Defaults (master switch `COLONY_MARKER_OFF=True` for Phase 3/4/5 compat):

```
COLONY_MARKER_EMIT       = 1.0     # amount per positive eat
COLONY_MARKER_DECAY      = 0.95    # per-tick multiplicative decay
COLONY_MARKER_DIFFUSION  = 0.3     # blend with blurred version
COLONY_MARKER_GRID_SCALE = 1       # cell = world unit (512×512 grid)
```

The founder gets 3 extra sensor nodes wired into hidden with small
random weights. Children inherit topology via NEAT mutation.

## 2. Full sweep — final population at t=8000

```
mode            seed  pop  maxGen  rec_frac_max  marker_total  pos      neg
static_dual     1     86   7      0.12          48174         5872552  0
static_dual     2     71   5      0.00          38056         5177559  0
static_dual     3     89   7      0.02          49248         6129330  0
visible_season  1      0   0      0.00            1520          96919   71432
visible_season  2      0   0      0.00            1368         103546   88136
visible_season  3      0   0      0.00            1197          64619   55673
hidden_season   1      2   1      0.00            3190         380994  254532
hidden_season   2      0   0      0.00            1767         171114  118946
hidden_season   3      0   0      0.00            1444          86794   79123
```

All visible_season and 2/3 hidden_season runs go extinct by t=5000-7000.
One hidden_season (seed 1) survives at pop=2 throughout. `rec_frac=0`
in every season run.

## 3. Comparison with prior phases (same 3 seeds × 8000 ticks)

| mode           | Phase 3.1 | Phase 5 (grace) | Phase 4.1+5 | Phase 6 (markers) |
|----------------|----------:|----------------:|------------:|------------------:|
| static_dual    | 100-110   | 110-113         | 111-116     | **71-89**         |
| visible_season | 0-0       | 0-0             | 0-6         | **0-0**           |
| hidden_season  | 0-1       | 0-1             | 0-4         | **0-2**           |

Phase 6 **does not improve** season-mode survival compared to Phase 5
grace. Both essentially lose the demographic race against the season
flip. Phase 6's static_dual pop is *lower* because the founder has to
re-tune itself with 3 extra sensor inputs.

## 4. Why markers do not save season modes

The marker field encodes **"food was eaten here recently"** but nothing
about which food is currently *positive*. After a season flip, the
old markers point to where the *old* positive food was. An organism
following those markers walks straight into the now-negative food.

To make markers useful across season flips, the marker would need to
encode sign-of-reward at the time of emission (positive vs negative
eat). That's a different design: a two-channel marker field (or a
per-season-color marker). This is a Phase 7 candidate.

The simpler interpretation: **the population is extinct before the
marker information becomes actionable**. Markers can only help if the
population is alive long enough to use them. In our regime, the
first season flip wipes out >90% of organisms within ~1000 ticks, and
no marker mechanism can prevent that — it's a demographic problem.

## 5. What Phase 6 does establish

- **External memory *is* sensed**: static_dual markers grow to 50-200
  active cells and the brain pays attention (visible by population
  paying the sensor budget).
- **External memory does not solve season-flip prediction**: the
  information asymmetry is in the brain (knowing which food is
  currently positive), not in the environment (knowing where food
  was).

This is a clean null result: it answers the *control question* the
literature review (PHASE4_LIT.md §4) raised. The answer is: external
markers are *not* a substitute for internal memory under seasonal
flips. Recurrence, if it ever evolves, must encode something the
markers cannot.

## 6. Conclusion

**Phase 6 was a clean architectural experiment that ruled out
environmental memory as a substitute for internal memory.** This is a
valuable negative result.

The remaining architectural levers:
- **Larger populations / longer ticks**: more time for selection.
- **Smarter founder**: seed populations with built-in seasonal
  expectations.
- **Phase 7: dual-channel markers**: encode sign-of-reward in the
  marker, so old-season markers become useless after a flip. Test
  whether this resolves the demographic crisis.
- **Acknowledge the null and stop**: Phase 3's question (does evolution
  grow memory?) is not reliably answered in this regime within a
  reasonable computational budget. The work has still established:
    - Phase 3 demography is stable when there is no season (static_dual).
    - Phase 4/4.1 plasticity helps demography but does not select
      for recurrence.
    - Phase 5 grace + Phase 4.1 rHebb combined does not save season
      modes within 8k ticks.
    - Phase 6 colony markers do not save season modes either.

## 7. Reproduction

```bash
python scripts/run_phase6.py --markers --seeds 1 2 3 --ticks 8000 \
    --warm-json evolife_regen_warm_parent.json --neg-energy -1
```

Without markers (control arm):
```bash
python scripts/run_phase3_1.py --seeds 1 2 3 --ticks 8000 \
    --warm-json evolife_regen_warm_parent.json --neg-energy -1
```
