"""Phase 3.2: try to stabilise hidden_season by reducing reward asymmetry.

Hypothesis: high_threshold (init_e=80, repro_t=128, pop=50, season=8000)
stabilises static_dual at pop=20. Hidden_season still goes extinct at
pop=0 because the founder's net reward rate halves once it eats the
"wrong" food (FOOD_*_NEGATIVE_ENERGY = -3).

If we shrink the negative magnitude, the founder's net rate stays
close to static_dual's. This makes the *task* (which food is positive)
still hard, but doesn't kill the founder before evolution gets a turn.

This smoke varies only neg_energy, holding the high_threshold config.
"""
from __future__ import annotations

import json
import time

from evolife.genome import Genome
from evolife.phase3 import MemoryEcologyWorld


NEG_GRID = [-3.0, -1.0, 0.0]


def run_one(mode, seed, ticks, warm, neg_e, init_e=80.0, repro_t=128.0,
            init_pop=50, season_len=8000):
    import evolife.config as cfg
    import evolife.phase3 as ph3
    import evolife.world as wmod
    cfg.PHASE3_INITIAL_ENERGY = init_e
    cfg.REPRODUCTION_THRESHOLD = repro_t
    cfg.INITIAL_POPULATION = init_pop
    cfg.SEASON_LENGTH = season_len
    cfg.FOOD_A_NEGATIVE_ENERGY = neg_e
    cfg.FOOD_B_NEGATIVE_ENERGY = neg_e
    ph3.PHASE3_INITIAL_ENERGY = init_e
    ph3.REPRODUCTION_THRESHOLD = repro_t
    ph3.INITIAL_POPULATION = init_pop
    ph3.SEASON_LENGTH = season_len
    ph3.FOOD_A_NEGATIVE_ENERGY = neg_e
    ph3.FOOD_B_NEGATIVE_ENERGY = neg_e
    wmod.REPRODUCTION_THRESHOLD = repro_t
    wmod.INITIAL_POPULATION = init_pop

    world = MemoryEcologyWorld(
        seed=seed, mode=mode,
        visible_season_sensor=(mode == "visible_season"),
        warm_genome=warm,
    )
    cum_pos = cum_neg = 0
    snapshots = []
    for t in range(ticks):
        world.step()
        for org in world.organisms:
            cum_pos += getattr(org, "positive_eats", 0)
            cum_neg += getattr(org, "negative_eats", 0)
        if t in (999, 1999, 3999, 5999, 7999):
            pop = sum(1 for o in world.organisms if o.alive)
            snapshots.append((world.tick, pop, world.season))
    pop = sum(1 for o in world.organisms if o.alive)
    max_gen = max((o.generation for o in world.organisms), default=0)
    return {
        "pop": pop, "max_gen": max_gen,
        "pos": cum_pos, "neg": cum_neg,
        "snapshots": snapshots,
    }


def main() -> None:
    seed = 1
    ticks = 8000  # one full season
    warm = None
    try:
        with open("evolife_regen_warm_parent.json") as fh:
            warm = Genome.from_dict(json.load(fh))
    except FileNotFoundError:
        print("WARNING: warm parent not found")

    print(f"warm: nodes={len(warm.nodes)} conns={len(warm.connections)}")
    print()
    rows = []
    for neg in NEG_GRID:
        for mode in ("static_dual", "hidden_season"):
            r = run_one(mode, seed, ticks, warm, neg)
            s = " ".join(f"t{tk}:p{p}(s{ss})" for tk, p, ss in r["snapshots"])
            print(
                f"  neg={neg:>5.1f}  {mode:<14} pop@end={r['pop']:>3} maxGen={r['max_gen']:>2} "
                f"pos={r['pos']:>7} neg={r['neg']:>5}  {s}",
                flush=True,
            )
            rows.append((neg, mode, r))

    print()
    print("Summary:")
    for neg, mode, r in rows:
        print(f"  neg={neg:>5.1f}  {mode:<14} pop@end={r['pop']:>3}")


if __name__ == "__main__":
    main()