# Phase 5 — Failure report

**Date:** 2025
**Status:** Phase 5 + Phase 4.1 combined does NOT save season modes.
The grace period is a band-aid that doesn't address the root issue.

## Headline

The Phase 5 grace window was implemented and tested (7 unit tests
pass, PHASE5_RESULTS.md). But the combined Phase 5 + Phase 4.1
eligibility-trace rHebb run on 8000 ticks × 3 seeds showed:

```
mode            seed1 seed2 seed3 verdict
static_dual     111   111   116   stable (no flip)
visible_season    1     6     0   all extinct or near-zero
hidden_season     4     1     0   all extinct or near-zero
```

And — more damningly — `rec_frac = 0.00` in every season run. Not a
single useful recurrent edge emerged in 8000 ticks of evolution +
plasticity.

## Why grace alone cannot solve this

The grace period (200 ticks of soft −0.5 penalty) is a *time* fix: it
gives the brain more room to update weights after a flip. But it
does not change the *credit-assignment problem*. Phase 4.1 rHebb
accumulates an eligibility trace so reward can be credited back to
past co-activations. Both of those are necessary but not sufficient:

1. **Time pressure.** Each season flip happens at fixed tick intervals
   (SEASON_LENGTH=2000). The brain has at most 2000 ticks per season
   to (a) survive, (b) reproduce, (c) update weights, (d) be selected.
   That's not enough generations to fix a structural mutation in the
   graph (NEAT add_node/add_connection) when most of the population
   is being culled.

2. **Initial-energy mismatch.** When a flip happens, ~50% of the
   population is in a depleted state (low energy, low reproduction).
   They cannot reproduce faster than they starve, regardless of how
   forgiving the negative penalty is.

3. **Selection on behavior, not structure.** Even if rHebb produces
   a slightly-better edge during the grace window, that edge has
   to make it through a full reproduction cycle to be inherited.
   With population crashing from 50 to 1 between flips, the
   effective selection pressure is mostly noise.

## What this means for the project

Phase 5 was the last "structural plasticity" lever I had. The
following null is now credible:

> *Within the current architectural choices (NEAT structural mutation,
> 5-tick eligibility trace, lifetime rHebb, season length 2000,
> initial energy 80, food target 400), evolution in this simulation
> does not select for recurrence within 8k ticks of any season-mode
> arm.*

Either the architectural choices need to change, or the hypothesis
itself (evolution grows memory under minimal assumptions) is
unfalsifiable in this regime without longer runs.

## Two honest next steps

1. **Long static_dual sweep** at 30k–50k ticks. No seasons, no
   memory task. Just to find the ceiling of `rec_frac` selection
   in the simplest regime where population is stable. If it
   saturates at <0.10 even there, the structural-mutation budget
   is the bottleneck (not season dynamics).

2. **Colonies.** A social architecture where organisms exchange
   local signals (Choe & Chung 2011 — environmental markers). This
   sidesteps the recurrent-edge selection problem entirely: if a
   group of organisms can deposit a scent marker when food is
   found, *any* organism (recurrent or not) can use it to find
   food without internal memory. The memory moves from the brain
   to the environment. This is the natural Phase 6 and respects
   Vitaly's request to drop the workarounds.

We are taking step 2 next. No more band-aids on the season-mode
demographic floor.
