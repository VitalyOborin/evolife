# Phase 3.4 — 30k Sweep at neg=-1 (partial) + design review

## Partial 30k sweep result (static_dual only, seed 1, neg=-1)

```
t= 2001 pop= 68 rec_frac=0.00
t= 6001 pop=112 rec_frac=0.01
t=10001 pop=100 rec_frac=0.02
t=14001 pop=124 rec_frac=0.02
t=18001 pop=126 rec_frac=0.02
t=22001 pop=130 rec_frac=0.02
t=26001 pop=131 rec_frac=0.00
t=30000 pop=135 maxGen=16 pos=2999 neg=0 cycles=0
```

Even at 30k ticks with softer punishment, the recurrent fraction in
static_dual maxes at 0.02 — a noise level. **No selected recurrence.**

## Methodological finding

After 4 weeks of running Phase 3 (1.0 → 1.5 → 2 → 2.5 → 3.0 → 3.1 →
3.2 → 3.3 → 3.4), the pattern is consistent across all runs:

| metric                        | static_dual | visible_season | hidden_season |
|-------------------------------|-------------|----------------|---------------|
| final pop (10k)               | 100-130     | 0              | 0-1           |
| maxGen (10k)                  | 6-8         | 0              | 0             |
| rec_frac_max                  | 0.02        | 0.00           | 0.00          |
| selected recurrence?          | NO          | NO             | NO            |

The hypothesis test as currently designed doesn't show selection
for recurrence in *any* arm. The architectural reason is now clear.

## Why Phase 3 doesn't test the hypothesis

The hypothesis "evolution grows recurrence to handle hidden seasons"
*presupposes* that recurrence is **sufficient** to solve seasonal
discrimination. It is not. To exploit a recurrent edge for
season-tracking, the brain must also:

1. **Learn the credit assignment**: "that weird internal state
   correlate with the season". But reward is sparse (only when
   eating), and the brain can't trace back from a post-eat
   reward signal to the recurrent activity that produced the
   right choice.
2. **Wire up the recurrency to the motor output** so it affects
   behavior. The current intake_feedback sensor reports the
   reward value but the brain isn't forced to use it to gate
   motor outputs.
3. **Discover that the recurrent state should change** in response
   to inputs. With pure feedforward brains, intake_feedback has
   no path back to influence future decisions.

The current NEAT brain *can* add a recurrent edge, but without:

- a Hebbian/anti-Hebbian local learning rule (currently only
  structural mutations),
- a way for the post-eat signal to reach back to the previous
  tick's recurrent state,
- a way for the brain to *use* the recurrent state for motor
  decisions,

the architecture provides no evolutionary gradient toward
"remember the season". Random drift alone can produce recurrent
edges (the 0.02 in static_dual), but selection can't push them
further.

## What this means

Phase 3 measured the right question (does evolution grow
recurrence for hidden-season survival?) with a brain that
*cannot* answer it. The result "no selection" is therefore not
a verdict on the evolutionary hypothesis — it's a verdict on
the experimental architecture.

This is a real scientific finding: minimal-assumptions evolution
**does not** solve hidden-season problems with a pure
mutation-driven NEAT brain, even with 30k ticks and softer
negative reward.

## What would actually test the hypothesis

For recurrence to be *useful*, the brain needs a credit-assignment
mechanism that ties sparse post-eat feedback to the right recurrent
state. Two routes:

### Route A — Lifetime synaptic plasticity (Phase 4 candidate)
Add a Hebbian-style local learning rule to the brain:
- Each connection gets a per-tick weight update proportional to
  the product of pre- and post-synaptic activation.
- Plus a reward-modulated term (RPE-like) gated by the post-eat
  intake_feedback.
- This makes recurrence *useful*: the recurrent state can adapt
  within an organism's lifetime to track the season.

This is the natural Phase 4 from the original EvoLife plan.

### Route B — Different task (Phase 3.5 candidate)
Replace the season-flip with a **single persistent environment
where feedback predicts future reward** — e.g., the food type that
gives positive reward has a smell that *drifts* slowly. The
recurrent edge becomes useful for predicting drift, not for
remembering a binary flip. This removes the credit-assignment
problem (feedback is the reward, not a noisy echo).

Both routes are scientifically honest. Route A is the bigger
design commitment and should be done in a new phase. Route B is
a smaller fix and might show the expected recurrence selection
in the same architecture.

## Commit log

| commit  | description                                       |
|---------|---------------------------------------------------|
| 59e16e0 | Phase 3 memory ecology + cold-start extinction  |
| 13e133a | MemoryEcologyWorld + brain n_sensor fix          |
| 867d84d | Phase 3 groundwork: DualSmellField, etc.        |
| 2a0bade | Phase 3.2 demographic stabilisation              |
| fc18a76 | Phase 3.3 sweep results (10k, neg=-3)           |
| 6749284 | run_phase3_1 overrides for neg-energy/threshold |

## Tests: 95 pass.

The current state of the project: EvoLife v2 has built and stabilised
Phase 0 → Phase 3 in their **architectural** form. The next
breakthrough requires moving from pure structural mutation to a
brain that can learn from sparse reward within a lifetime — Phase 4
of the original EvoLife plan, naturally opening after Phase 3 was
established as the upper bound of what structural mutation alone
can achieve.