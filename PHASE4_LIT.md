# Phase 4 — Literature review

**Date:** 2025
**Status:** active — informs Phase 4.1 design.

This note records what published work says about how **memory / recurrence
emerge in evolved plastic neural networks**, and maps those findings onto
EvoLife's current architecture. The goal is to identify which mechanism we
are missing that could turn our static-recurrence ceiling into selection
pressure for recurrent dynamics.

## 1. The three learning loops

The clearest framing is **Soltani & Wang / Miconi et al.**, who argue that
any system capable of acquiring novel cognitive tasks needs **three
nested loops**:

1. **Evolution** — outer loop over lifetimes; structural search.
2. **Task / plasticity** — middle loop within a lifetime; weight
   change that lets an animal learn within its lifetime.
3. **Episode** — inner loop within a trial; recurrent dynamics /
   working memory.

> "We operationally define cognitive tasks as those that require some
> within-episode learning and memory, evolving such a cognitive learner
> necessarily involves at least three nested learning/memory loops."
> — Pedersen, Risi, et al., "Learning to acquire novel cognitive tasks
> with evolution, plasticity and meta-meta-learning" (arXiv:2112.08588)

EvoLife state of play:

| Loop              | Present? | Mechanism                          |
|-------------------|----------|------------------------------------|
| Evolution         | yes      | NEAT structural mutation            |
| Task plasticity   | Phase 4  | per-tick local Hebbian + R-mod      |
| Episode (recurrence) | emerging | structural mutation only          |

Loop 3 is precisely what we are trying to grow. Loops 1 and 2 are present
but our **loop 2** is sub-optimal: per-tick reward modulation on a
2-factor rule does not propagate credit through time, which is what
loops 1+3 jointly need to bridge.

## 2. Miconi (2017) — biologically plausible plasticity in RNNs

The most directly relevant result. Miconi (eLife, 2017) showed that a
**reward-modulated Hebbian rule with an eligibility trace** can train
chaotic recurrent networks to solve delayed non-match-to-sample and
sequential-XOR — using only sparse, delayed rewards at end of trial.

Mechanism (rHebb):

```
e_ij(t) = e_ij(t-1) + S(pre_i * (post_j - <post_j>))     # eligibility
Δw_ij   = η * (R - R_b) * e_ij                            # commit
```

with **S(x) = x³** (supralinear) and `R_b` a learned reward baseline.

Three ingredients we do NOT have:

1. **Eligibility trace.** Without it, reward modulation can only act
   on the immediately preceding tick's activations. With it, reward
   delivered at end of trial still reaches synapses that fired hundreds
   of ticks earlier — exactly the credit-assignment bridge we need
   between sparse post-eat feedback and recurrent state.
2. **Supralinear non-linearity** on pre·post. `S(x) = x³` amplifies
   large co-activations and damps small noise-driven ones. Without it,
   the stochasticity of recurrent firing drowns the Hebbian signal.
3. **End-of-trial reward**, not per-tick. Letting eligibility
   accumulate and only committing on actual reward events keeps the
   plasticity signal meaningful.

Our current Phase 4 rule:

```
delta_w = α * pre * (post - post) + β * R * pre * post
       = 0            + β * R * pre * post
```

The local Hebbian term is **identically zero** by construction — the
"target" we subtract is `post` itself. The reward term is the only
effective term, and it acts per tick without an eligibility trace.

This explains the Phase 4 smoke results we observed:

- All three arms gain demography (organisms survive longer).
- `rec_frac` hovers at 0.01–0.02, i.e. **noise floor** — recurrence is
  not being selected for because the plasticity has no mechanism to
  make a recurrent edge *useful*.

## 3. Lakhman & Burtsev (2012) — short-term memory via neuron duplication

> "Evolution discovered two mechanisms for short-term memory. The first
> mechanism is integration of sensory signals and ongoing internal
> neural activity, resulting in emergence of cell groups specialized
> on alternative actions. The second mechanism is slow neurodynamical
> processes that makes possible to code the previous behavioral choice."

Their model duplicates neurons (a structural mutation that yields
topological expansion, similar to NEAT's `add_node`). Memory emerges
through:

- **Integration dynamics** (mixed selectivity, conjunctive coding).
- **Slow internal dynamics** in recurrent loops.

Implication for us: even with pure structural mutation, two regimes
should be testable:

- **Fast regime** — few hidden nodes, light recurrence, dynamics
  dominated by instantaneous sensory input. Memory bounded.
- **Slow regime** — more nodes, recurrent feedback loops with
  intermediate time-constants. Memory can integrate over many ticks.

We currently have NEAT's mutation operators but no explicit **slow
dynamics** bias. Adding a leak / decay constant (or a slower
recurrence weight scale) would let recurrent loops accumulate state
without immediate reset.

## 4. Choe & Chung (2011) — feedforward + environmental markers

> "Even memoryless feedforward networks can evolve behavior that can
> solve tasks requiring memory, when material interaction is allowed."

Two architectures compared:

- Recurrent controller with internal state.
- **Feedforward controller + dropper + detector** that writes and reads
  environmental markers (pheromones, excretions).

The feedforward + markers group matches recurrent-controller performance
on ball-catching and food-foraging tasks. Memory becomes **environmental
state**, not neural state.

Implication for EvoLife: an **external marker field** (organism emits a
scent/pheromone when eating, detects its own scent within a finite
lifetime window) is a feedforward-compatible alternative to recurrence.
If we add this BEFORE introducing recurrence, we can:

- Test whether the *behavior* of recurrent organisms is genuinely
  different from feedforward + markers — i.e. is internal memory
  actually selected, or do organisms just outsource memory to the world?
- Provide a control arm that rules out the "they don't need recurrence"
  null hypothesis.

## 5. NEAT-LSTM (Rawal & Miikkulainen, 2016) and gated memory cells

NEAT extended to discover LSTM cells outperforms vanilla RNN on POMDP
memory tasks. Two reasons:

- LSTM gates allow **explicit memory gating** (input/forget/output).
- NEAT can grow the number of memory cells as needed.

Implication: a Phase 5 architectural step could be a **gated recurrent
unit** node type (input/forget/output gates), evolved via NEAT. This is
a bigger change than Phase 4.1 — it requires:

- New node type (`NodeType.LSTM` or `NodeType.GRU`).
- Forward pass semantics for gated cells.
- Mutation operators that add gated cells (not just plain hidden).

Out of scope for the immediate next step but worth flagging as the
natural Phase 5.

## 6. Synthesis — three concrete gaps in Phase 4

| Gap                                | Source             | Fix in Phase 4.1?       |
|------------------------------------|--------------------|-------------------------|
| No eligibility trace               | Miconi 2017        | yes — accumulate e(t)   |
| 2-factor reward modulation only    | Miconi 2017        | yes — 3-factor rule     |
| Per-tick reward, not end-of-trial  | Miconi 2017        | yes — accumulate R, commit on episode boundary (reproduction or starvation) |
| Local term is structurally zero    | code review        | yes — use running mean  |
| Supralinear non-linearity absent   | Miconi 2017        | yes — `S(x)=x³`         |
| No slow-dynamics bias              | Lakhman-Burtsev 2012 | yes — decay constant   |
| No external-marker control         | Choe 2011          | defer to Phase 5+       |

## 7. Phase 4.1 design — eligibility-trace reward-modulated plasticity

Direct port of Miconi's rHebb.

### 7.1 Per-tick accumulation (cheap, runs every tick)

For each edge (i→j):

```
e_ij(t) = λ * e_ij(t-1) + (1 - λ) * (pre_i * post_j - <pre·post>EMA)
```

where:

- `λ` is the eligibility decay (e.g. 0.95 — keeps trace alive for ~20
  ticks).
- `<pre·post>EMA` is the running mean of pre·post on this edge
  (Miconi's `<post_j>` is approximated by an EMA per-edge).
- The eligibility is **not** committed yet — just accumulated.

### 7.2 Per-episode commit (rare, on starvation or reproduction)

When an episode ends:

```
Δw_ij = η * (R - R_b) * e_ij(T)
```

where:

- `R` is the **episode reward** = total positive eats − |total negative
  eats| during the episode.
- `R_b` is a per-organism running-mean reward baseline (Miconi's
  reward prediction baseline).
- `η` is the plasticity rate (much smaller than per-tick since this
  commits a whole trace).

Then `e_ij = 0` (reset for next episode) and `R_b ← EMA(R_b, R)`.

### 7.3 Episode boundary detection

In EvoLife, episodes correspond naturally to **lifetimes**:

- Start: birth (parent just spawned).
- End: **starvation** (`energy ≤ 0`) or **reproduction** (genome copied
  to offspring). Episode ends, weights commit, eligibility resets.

Reproduction-as-episode-end is biologically clean: the parent has
finished a "lifetime" of trial-and-error and passes on weights shaped
by that lifetime. Starvation-as-end is the other half of the same coin.

### 7.4 What changes in code

- `brain.py` keeps the current per-tick forward pass untouched.
- Add `e_trace` and `r_baseline` to Brain state (not genome — these are
  lifetime variables, like `state`).
- `Brain.begin_episode()` resets e_trace; called by world on spawn.
- `Brain.observe(sensors)` runs forward AND accumulates e_trace in the
  same step (zero extra cost over current loop).
- `Brain.commit_episode(episode_reward)` runs once on starvation or
  reproduction; applies `Δw = η (R - R_b) e`, resets trace, updates
  baseline.

### 7.5 Defaults

```python
ELIGIBILITY_DECAY = 0.95
ELIGIBILITY_BASELINE_EMA = 0.99
PLASTICITY_RATE = 0.001             # per-episode commit
R_BASELINE_EMA = 0.99
SUPERLINEAR_POWER = 3               # Miconi's S(x)=x³
```

All Phase 4 α/β knobs become inactive; Phase 4.1 supersedes Phase 4 but
keeps the same opt-in flag (`--enable-plasticity`).

### 7.6 Expected outcome

- **static_dual**: plasticity should now allow a recurrent mutant that
  *anticipates* food-A scarcity by re-weighting sensors when energy is
  low. Population should grow beyond Phase 4 numbers (which were
  demography-only gains).
- **hidden_season / visible_season**: this is where Miconi's rule is
  designed to win. Eligibility trace lets the network remember the
  previous season's "wrong food" and bias future exploration. Expected
  to see `rec_frac > 0.05` for the first time, with a positive slope
  across generations rather than noise.

If Phase 4.1 still doesn't lift `rec_frac`, the null hypothesis
becomes credible: evolution under our constraints simply doesn't
select for recurrence, regardless of plasticity mechanism.

## 8. References

- Miconi, T. (2017). Biologically plausible learning in recurrent
  neural networks reproduces neural dynamics observed during cognitive
  tasks. eLife 6:e20899. doi:10.7554/eLife.20899
- Pedersen, J.W. & Risi, S. (2021). Learning to acquire novel cognitive
  tasks with evolution, plasticity and meta-meta-learning. arXiv:2112.08588
- Lakhman, K. & Burtsev, M. (2012). Neuroevolution results in emergence
  of short-term memory for goal-directed behavior. arXiv:1204.3221
- Choe, Y. & Chung, J.R. (2011). Emergence of memory in reactive agents
  equipped with environmental markers. IEEE T-AMD.
- Rawal, A. & Miikkulainen, R. (2018). From nodes to networks: Evolving
  recurrent neural networks. arXiv:1803.04439
- Soltoggio, A. et al. (2018). Evolved plastic artificial neural
  networks. arXiv:1703.10371
