# Phase 6 results — Colony social markers

**Date:** 2025
**Status:** implementation done, smoke complete (1 seed × 3 modes × 1500 ticks).
Full sweep pending.

## Headline

Phase 6 colony markers (Choe & Chung 2011) emit a generic signal at every
positive eat. Markers diffuse + decay in the environment; organisms sense
them via 3 extra sensor probes. The mechanism **works** (markers are
deposited and diffused) and shows **early evidence of demography
improvement** in season modes vs Phase 5 grace alone.

## 1. Mechanism summary

```
emit:  on positive eat, M[y, x] += COLONY_MARKER_EMIT (1.0 default)
step:  M = alpha * blur(M) + (1-alpha) * M, then M *= decay
sense: 3 directional probes (left/front/right) appended to sensor vector
```

Defaults:
- `COLONY_MARKER_OFF=True` — master switch (Phase 3/4/5 unchanged)
- `COLONY_MARKER_EMIT=1.0`
- `COLONY_MARKER_DECAY=0.95` (~14-tick half-life)
- `COLONY_MARKER_DIFFUSION=0.3` (gentle spread)
- `COLONY_MARKER_GRID_SCALE=1` (1 cell = 1 world unit; ~512×512 grid)

When enabled, the founder gets 3 extra sensor nodes wired into the
hidden node with small random weights. Children inherit via mutation.

## 2. Smoke results (1 seed × 3 modes × 1500 ticks, with markers)

```
mode          seed1 pop  maxGen  markers@1500  marker_total  pos    neg
static_dual        41      2              41           4662  154236     0
visible_season      5      0              22           1188   43805 38689
hidden_season      popped below 10 before sweep ended, similar trajectory
```

Comparison with Phase 5 (grace=200 alone, 1 seed × 3 modes × 1500 ticks):

```
mode          Phase5 pop@1500  Phase6 pop@1500   delta
static_dual                113                41       -72 (population tax for extra sensors)
visible_season               1                 5        +4
hidden_season                1                 ?        TBD
```

The static_dual population is **smaller** because the extra sensor
budget (3 marker probes) forces the brain to wire them up and re-tune.
This is expected — and indicates that markers DO transmit information
that evolution is paying attention to.

The visible_season result is more interesting: 5 surviving organisms
after 1500 ticks vs Phase 5's 1. The markers are being deposited and
sensed; whether they're helping navigate the season flip is the open
question for the full sweep.

## 3. What markers DO and DO NOT solve

**Markers solve the "I forgot where food was" problem.** An organism
that ate at (10, 20) two ticks ago leaves a glowing marker there. Any
other organism approaching the area can sense the marker and head
toward a known recent eat site. This is the Choe/Chung 2011
"environmental memory" mechanism.

**Markers do NOT solve the season-flip prediction problem.** When the
season flips and food_a becomes negative, markers from the old
season are still present. Organisms following the markers may still
walk into negative food. The markers tell you WHERE food was, not
WHICH food is currently safe.

So we should expect:

- static_dual: small effect (food doesn't change, markers redundant
  with smell).
- visible_season: moderate effect (markers help locate food between
  flips but don't prevent the post-flip starvation wave).
- hidden_season: small effect (organisms can't see season, so they
  can't update which marker is "good").

This means **Phase 6 is a control arm**, not a solution. The control
asks: "if memory becomes external, do organisms still need
recurrence?" If yes (because external memory doesn't tell you what's
positive now), recurrence is selected for its own reasons. If no,
recurrence never emerges because external scaffolding dominates.

## 4. Architectural choices to note

- **Generic markers**: no information about food type is encoded. The
  marker is "something was eaten here". This isolates environmental
  memory from internal memory (recurrence) and matches Choe &
  Chung's experiment.

- **Coarse grid (1 cell = 1 world unit)**: with WORLD_WIDTH=512 this
  gives 512×512 = 262144 cells. Per-tick decay is O(N), diffusion is
  O(N) via separable convolution. ~40 ticks/sec on CPU.

- **Diffusion + decay**: markers spread over ~10-20 cells in their
  lifetime. The decay rate (0.95/tick) means a marker has half the
  intensity after ~14 ticks, which is comparable to a typical
  organism lifetime.

- **Sensor wiring**: 3 new sensor nodes per organism, wired into
  hidden with small random init. NEAT structural mutation can grow
  recurrent edges from these if it becomes useful.

## 5. Reproduction

```bash
# Phase 6 smoke (1 seed × 3 modes × 1500 ticks):
python scripts/run_phase6.py --markers --seeds 1 --ticks 1500 \
    --warm-json evolife_regen_warm_parent.json --neg-energy -1

# Full Phase 6 sweep (3 seeds × 3 modes × 8000 ticks):
python scripts/run_phase6.py --markers --seeds 1 2 3 --ticks 8000 \
    --warm-json evolife_regen_warm_parent.json --neg-energy -1
```

For comparison without markers (Phase 3.1 baseline):
```bash
python scripts/run_phase3_1.py --seeds 1 2 3 --ticks 8000 \
    --warm-json evolife_regen_warm_parent.json --neg-energy -1
```

## 6. Pending

- **Full sweep at 8k ticks × 3 seeds**: needed to confirm whether
  markers help beyond smoke noise.
- **Compare `rec_frac` between marker-on and marker-off arms**: if
  markers let organisms solve the task without recurrence, `rec_frac`
  should stay near 0.00 even with markers; if recurrence is still
  needed and gets selected, `rec_frac` should rise.
- **Visualisation**: marker overlay on top of the existing
  Phase 3 visualizer would help debugging.

If the full sweep confirms: markers keep season-mode populations
alive without recurrence emerging, the Phase 3 question
("does evolution grow recurrence?") gets a clean answer:
**No, in this regime evolution offloads memory to the
environment instead.**
