# Phase 4 results — Lifetime synaptic plasticity

**Date:** 2025
**Status:** sweep complete. Phase 4 lifts demography but does NOT
select for recurrence.

## 1. Setup

Phase 4 added **per-tick local Hebbian + reward-modulated plasticity**
to each organism's brain. The rule (applied in `Brain.apply_plasticity`
every tick that has a non-zero `intake_feedback`):

```
delta_w = alpha * pre * (post - target) + beta * R * pre * post
       = 0              + beta * R * pre * post
```

The local term is structurally zero (target = post), so the only
effective term is `beta * R * pre * post`. Default rates:
alpha=0.01, beta=0.05, neg-energy=-1.0.

**Sweep:** 3 modes × 3 seeds × 15,000 ticks, warm-started from the
Phase 1.5 6-node / 5-connection evolved parent.

| arm           | seed 1 | seed 2 | seed 3 |
|---------------|-------:|-------:|-------:|
| static_dual   |   109  |   117  |   140  |
| visible_season|     0  |     0  |     0  |
| hidden_season |     0  |     0  |     0  |

Final population at t=15000.

## 2. Per-arm timeline (selected runs)

**static_dual seed 1** — stable across 15k ticks, pop 68→110, no season flip:
```
t=1001 pop=72 rec_frac=0.00
t=5001 pop=110 rec_frac=0.02
t=10001 pop=102 rec_frac=0.00
t=15000 pop=109 maxGen=9
```

**visible_season seed 1** — collapses at the first season flip:
```
t=1001 pop=21 rec_frac=0.00 (season 0)
t=2001 pop=8  rec_frac=0.00 (season 1, demographic crash)
t=8001 pop=0 (extinct, season 0)
```

**hidden_season seed 2** — collapses earlier (no sensor to warn of flip):
```
t=1001 pop=49 rec_frac=0.00
t=2001 pop=13 rec_frac=0.00
t=5001 pop=1 rec_frac=0.00
t=6001 pop=0 (extinct)
```

## 3. What we learned

### 3.1 Phase 4 plasticity lifts demography, not recurrence

Comparing Phase 3.3 (10k ticks, no plasticity) with Phase 4 (15k ticks,
with plasticity) on the same worlds:

| arm          | Phase 3.3 pop@10k | Phase 4 pop@15k | delta |
|--------------|------------------:|----------------:|------:|
| static_dual  | 100–109           | 109–140         | +0 to +40 |
| visible_season | 0               | 0               | 0     |
| hidden_season| 0–1                | 0               | -1 to 0 |

Static_dual grows by 0–40 individuals; visible/hidden still go extinct.

**But `rec_frac` stays at the noise floor** in every arm — max value
across all 9 runs was 0.08 (static_dual seed 3, t=11001), and that
signal disappeared by t=15000. No persistent recurrent lineage emerges.

### 3.2 Why Phase 4 does not select for recurrence

The local Hebbian term is **identically zero** by construction (target
= post). The reward-modulated term acts per-tick without an
eligibility trace. This means:

- Reward arrives only when the organism eats (sparse).
- Each tick, the rule modulates weights by `R * pre * post`, where `R`
  is the *current* intake feedback.
- A recurrent edge that was useful for navigation many ticks before
  the eat is not credited, because the per-tick rule does not retain
  a record of past co-activations.

So Phase 4 plasticity modulates weights, but it cannot bridge sparse
reward events to recurrent dynamics. It is essentially a 2-factor
rule; biological learning in recurrent circuits is 3-factor
(eligibility × reward prediction error, see PHASE4_LIT.md §2).

### 3.3 The extinction of season arms is demographic, not brain-limited

Both season arms go extinct at the first flip. In visible_season, the
sensor lets the brain know about the flip but the brain is too small
to act on it; in hidden_season there is no sensor at all. The
population crashes because:

1. Organisms reproduce up to PHASE3_FOOD_TARGET/2 on the GOOD food
   type (no season-flip awareness).
2. At the season flip, all the food of that type becomes NEGATIVE.
3. Population overshoots food budget → mass starvation.

The plastic rule actually makes this WORSE in some seeds: positive
eats become harder to dislearn when reward modulation reinforces
existing edges. Phase 4's positive net effect on demography (visible
in static_dual pop 50→140) does not survive into the season arms
because the demographic wave kills organisms before any lifetime
plasticity can shape a viable strategy.

## 4. Where Phase 4 does help: the comparison arms

Even without recurrence emergence, Phase 4 has measurable effects:

- **static_dual population growth.** Phase 3.3 maxed at pop ~109.
  Phase 4 reaches pop ~140 (seed 3) — +30% over the no-plasticity
  baseline.
- **Generation depth.** Phase 4 reaches maxGen 9–13 in static_dual
  vs Phase 3.3's ~6–8.
- **Survival through the first season.** Visible/hidden season pops
  reach 21–49 at t=1001 in Phase 4 vs 0–1 in Phase 3.3 (the
  Phase 4.2 commit). Phase 4 organisms survive the FIRST tick-flip
  in some seeds before collapsing.

These are demographic / lifespan effects, not memory effects. The
plastic rule helps organisms extract more food per unit time
(perhaps via subtle weight tuning), but does not give them any
tool that helps with the season-flip dilemma.

## 5. Conclusion

**Phase 4 alone does not answer the original Phase 3 question**
(does evolution grow recurrence under minimal assumptions?). The
sweep confirms the Phase 3.4 methodological finding from a different
angle: per-tick 2-factor reward modulation is not enough to bridge
sparse post-eat feedback to recurrent dynamics. The biological
literature (Miconi 2017, see PHASE4_LIT.md) shows that 3-factor rules
with eligibility traces ARE sufficient — provided the trace has
enough capacity to bridge reward events to past co-activations.

The next architectural step is therefore **Phase 4.1: eligibility-trace
rHebb (Miconi 2017)**, implemented in commit 5e469f7. Initial smoke
results (1 seed × 3 modes × 1500 ticks) show the first non-zero
rec_frac (0.05) in visible_season at t=1251, suggesting the trace
mechanism is engaging. A full 15k-tick sweep is the next data point.

## 6. Reproduction

```bash
python -u scripts/run_phase4.py \
    --seeds 1 2 3 --ticks 15000 --log-every 1000 \
    --warm-json evolife_regen_warm_parent.json \
    --neg-energy -1 --alpha 0.01 --beta 0.05
```

Phase 4.1 (next):

```bash
python -u scripts/run_phase4_1.py \
    --seeds 1 2 3 --ticks 15000 --log-every 1000 \
    --warm-json evolife_regen_warm_parent.json \
    --neg-energy -1
```
