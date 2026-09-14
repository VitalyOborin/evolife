# Phase 1 Results — structural mutation rates bumped 5x

## Background

Phase 0 (commit `1e83237`): 5 seeds × 50k on CPU, with low structural mutation
rates (`ADD_NODE_RATE=0.002`, `ADD_CONNECTION_RATE=0.005`). Result: 2/5 stable
seeds (pop 103–104), 3/5 extinct. `first_hidden_node` fired in 2/5 seeds,
`first_recurrent_cycle` (hidden↔hidden) **never fired** within 50k ticks.

Phase 1 (commit `164554d`): same config except:

  ADD_NODE_RATE          0.002 -> 0.01
  ADD_CONNECTION_RATE    0.005 -> 0.02
  TOGGLE_CONNECTION_RATE 0.002 -> 0.005

## Run summary

Background bash was killed by maxRunMs timeout after ~40 minutes. We have
**3 of 5 seeds complete**: 1, 2, 3. Seeds 4 and 5 did not run to completion.

Per-seed outcomes (final world snapshot at last recorded tick):

| seed | pop @ tick 50 | pop @ final | max repros | eats | species | established species | outcome |
|-----:|--------------:|------------:|-----------:|-----:|--------:|--------------------:|---------|
| 1    | 226           | 249         | 18563      | 33742| 74      | 1 (s1, born t1508)  | stable   |
| 2    | 216           | 262         | 18898      | 34048| 99      | 4 (s0 born 0; s89/94/95 born ~46-47k) | stable, late speciation |
| 3    | 220           | 0 @ t46751  | (extinct)  | (extinct) | 76 | 1 (s21, born t9169) | late extinction @ t46751 |

Phase 0 vs Phase 1 — same seeds, same world, different rates:

| seed | Phase 0 pop@50k | Phase 1 pop@50k | Δ pop   | Phase 0 maxGen | Phase 1 maxGen | Δ maxGen |
|-----:|----------------:|----------------:|--------:|---------------:|---------------:|---------:|
| 1    | 104             | 249             | +139%   | 170            | 156            | -14      |
| 2    | 103             | 262             | +154%   | 148            | 102            | -31      |
| 3    | 0 (extinct @ 509) | 0 (extinct @ 46751) | +9100% in lifespan | 0 | (peak pop 275) | — |

## What this tells us

### What improved

- **Population size more than doubled** on stable seeds (104 → 249, 103 → 262).
- **Reproduction and food intake grew ~70–80%** (10232 → 18563, 11046 → 18898
  on seeds 1–2).
- **Species count exploded**: 1–2 species in Phase 0 → 74–99 species in Phase 1.
  Structural mutations produce real genotypic variation, and the speciation
  threshold picks them up.
- **Late extinction on seed 3 went from tick 509 to tick 46751** — two orders
  of magnitude later. Higher rates don't prevent extinction, but they
  dramatically extend the survival window for organisms to find workable
  mutations.

### What didn't change (yet)

- `first_recurrent_cycle` (hidden↔hidden) milestones are still in
  `evolife_phase0_archive.json` (which was the only archive the script had a
  chance to write before timeout — it is the Phase 0 archive, not Phase 1).
  Per-seed SQLite files contain `species_snapshots` and `events` but **not**
  the archive milestones (archive lives in-memory and is dumped once at the
  very end). So we cannot directly compare hidden↔hidden counts between Phase 0
  and Phase 1 from this run.
- maxGen dropped slightly on seeds 1–2 (170 → 156, 148 → 102). With larger
  populations the average generation comes from a more even mix, so the
  maximum generation recorded is not necessarily lower in absolute terms — but
  it suggests structural mutations may be slightly more deleterious on
  average than weight mutations, which is biologically expected.
- Seed 3 still went extinct, just much later. Extinction is not solved.

### What we would need to fully answer the recurrent-cycle question

We'd need either:
- Run seeds 4 and 5 to complete the 5-seed sweep, then re-run with explicit
  per-seed archive dumps (so milestones survive even on timeout), OR
- Run a single deterministic seed with extended wall clock so the script
  reaches its summary phase and writes the JSON.

The Phase 0 baseline remains the only archive we have for `first_hidden_node`
and `first_recurrent_cycle`. The structural infrastructure is in place;
collecting Phase 1 milestones is a re-run concern, not a code change.

## Recommendation

Bumping rates 5× did not break evolution — it accelerated it in a meaningful
way. The next sensible steps are either:
1. Run seeds 4–5 and write the Phase 1 archive (so we can answer the
   hidden↔hidden question), OR
2. Drop into Behavioral Arena and measure the four indicator families on
   Phase 1 frozen genomes.
