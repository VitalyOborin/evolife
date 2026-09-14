# Phase 4 — Lifetime Synaptic Plasticity (design)

## Why Phase 4

Phase 3 (Memory Ecology) was designed to test whether evolution
*discovers* recurrence under minimal assumptions: pure mutation,
no lifetime learning, identical founders, three worlds identical
except for ecology.

**Result**: across 9 runs (3 seeds × 3 modes × 10k and 30k ticks),
no recurrence was selected in any arm. The maximal recurrent
fraction in static_dual (the control where there's no selection
pressure for recurrence) is 0.02 — pure drift.

The reason: structural mutation alone cannot connect a sparse
post-eat feedback signal back to the recurrent state that
produced the right behavior. **Recurrence is necessary but not
sufficient** — the brain also needs *credit assignment*, which
in biology comes from **lifetime synaptic plasticity**.

Phase 4 introduces lifetime plasticity. Phase 4 is the natural
next step from Vitaly's EvoLife plan, and it directly addresses
the Phase 3 ceiling.

## What Phase 4 adds

A minimal lifetime learning rule on top of the NEAT structural
mutation:

```
Δw_ij = α * a_i * (a_j - target_j)            # local Hebbian
       + β * (reward - baseline) * a_i * a_j  # reward-modulated
```

Where:
- `w_ij` is the weight from neuron `i` to neuron `j`,
- `a_i`, `a_j` are pre/post activations at this tick,
- `target_j` is the neuron's clamped target (input value if
  sensor, current `tanh` output if hidden/motor),
- `reward` is the post-eat feedback (±1 from
  `intake_feedback`),
- `baseline` is a per-organism running average of recent reward,
- `α`, `β` are small learning rates (e.g. 0.001, 0.01).

No backprop. No gradients. Just locally-computable updates that
fire whenever intake_feedback changes.

## Why this should change Phase 3's outcome

Once the brain has a way to *use* recurrence (because the
recurrent state can be tuned within a lifetime to track the
season), selection can act on *whether* the recurrence is present:

- Founder has no recurrence → can't track season → extinction.
- Mutation adds recurrence → brain uses it → survives the
  season flip → reproduces more.
- Brain that uses recurrence efficiently wins → recurrence
  spreads.

The result of Phase 3.4 (no selection) should flip to "selection
for recurrence in hidden_season only" once Phase 4 is wired.

## Implementation plan

1. **Brain.py**: add `plasticity_alpha`, `plasticity_beta`
   per-connection attributes. After each forward pass, apply the
   rule on each edge that has `enabled=True` and was active this
   tick (non-zero activation in either direction).
2. **Organism.py**: apply the rule when `intake_feedback`
   changes. The reward term scales by `(intake_feedback -
   self.reward_baseline)`. Else only the local Hebbian term
   applies.
3. **Phase 3.5 forward-compat**: keep the Phase 3 three-mode
   runner working — just enable/disable plasticity per-phase.
4. **Phase 4 runner**: copy `run_phase3_1.py`, but set
   `plasticity_alpha=0.001, plasticity_beta=0.01` and re-run
   the same 3 modes × 3 seeds × 30k ticks. Expected: static_dual
   and visible_season unchanged; hidden_season should now have
   higher rec_frac.

## Risk: still too small

Even with plasticity, the credit-assignment problem can be too
hard if `intake_feedback` only fires once per eat. The brain
might need a *denser* signal — e.g. a constant feedback value
that follows from "am I in a good patch". Phase 3.5 (continuous
drift) might be the cleaner test even after Phase 4. Plan to
run both.

## Files / status

- Brain.py currently has structural mutation only.
- Phase 4 will require ~50 lines in brain.py + ~20 lines in
  organism.py + a new `scripts/run_phase4.py` (copy of phase3_1).
- Total expected effort: 1-2 days of implementation + a 30k
  sweep per mode (~1 hour per mode).

This is the path forward.