"""Phase 3.2 demographic grid smoke.

Runs all 3 modes (static_dual / visible_season / hidden_season)
across a small grid of demographic overrides and reports a compact
summary table. The point is to find *some* configuration where all
three modes have stable populations long enough for evolution to
discover recurrence (i.e. at least one full season survives).

Run from project root.
"""
from __future__ import annotations

import json
import sys
import time

from evolife.genome import Genome
from evolife.phase3 import MemoryEcologyWorld


CONFIGS = [
    # (label, initial_energy, repro_threshold, initial_pop, season_length)
    ("baseline",       80.0,  64.0, 50, 2000),
    ("low_energy",     80.0,  64.0, 50, 8000),  # long season
    ("high_threshold", 80.0, 128.0, 50, 8000),
    ("low_pop",        80.0,  64.0, 20, 8000),
    ("original_400",  400.0,  64.0, 50, 2000),  # the broken one
]


def run_one(label, mode, seed, ticks, warm, cfg):
    label_local, init_e, repro_t, init_pop, season_len = cfg
    # Monkey-patch all relevant module bindings.
    import evolife.config as cfgmod
    import evolife.phase3 as ph3
    import evolife.world as wmod
    cfgmod.PHASE3_INITIAL_ENERGY = init_e
    cfgmod.REPRODUCTION_THRESHOLD = repro_t
    cfgmod.INITIAL_POPULATION = init_pop
    cfgmod.SEASON_LENGTH = season_len
    ph3.PHASE3_INITIAL_ENERGY = init_e
    ph3.REPRODUCTION_THRESHOLD = repro_t
    ph3.INITIAL_POPULATION = init_pop
    ph3.SEASON_LENGTH = season_len
    wmod.REPRODUCTION_THRESHOLD = repro_t
    wmod.INITIAL_POPULATION = init_pop

    world = MemoryEcologyWorld(
        seed=seed, mode=mode,
        visible_season_sensor=(mode == "visible_season"),
        warm_genome=warm,
    )
    t0 = time.time()
    cum_pos = cum_neg = 0
    snapshots: list[tuple[int, int]] = []  # (tick, pop)
    for t in range(ticks):
        world.step()
        for org in world.organisms:
            cum_pos += getattr(org, "positive_eats", 0)
            cum_neg += getattr(org, "negative_eats", 0)
        if t in (199, 999, 1999, 3999, 7999):
            pop = sum(1 for o in world.organisms if o.alive)
            snapshots.append((world.tick, pop))
    elapsed = time.time() - t0
    pop = sum(1 for o in world.organisms if o.alive)
    max_gen = max((o.generation for o in world.organisms), default=0)
    return {
        "label": label,
        "mode": mode,
        "seed": seed,
        "pop": pop,
        "max_gen": max_gen,
        "pos": cum_pos,
        "neg": cum_neg,
        "snapshots": snapshots,
        "elapsed_s": elapsed,
    }


def main() -> None:
    seed = 1
    ticks = 4000  # cover at least 1 full season (2000) plus runout
    warm = None
    try:
        with open("evolife_regen_warm_parent.json") as fh:
            warm = Genome.from_dict(json.load(fh))
    except FileNotFoundError:
        print("WARNING: warm parent not found, cold-start")

    print(f"warm: nodes={len(warm.nodes)} conns={len(warm.connections)}")
    print()
    rows = []
    for cfg in CONFIGS:
        label = cfg[0]
        print(f"=== {label} ===", flush=True)
        for mode in ("static_dual", "hidden_season"):
            r = run_one(label, mode, seed, ticks, warm, cfg)
            s = " ".join(f"t{tk}=p{p}" for tk, p in r["snapshots"])
            print(
                f"  {mode:<14} pop@end={r['pop']:>3} maxGen={r['max_gen']:>2} "
                f"pos={r['pos']:>6} neg={r['neg']:>5}  {s}",
                flush=True,
            )
            rows.append(r)
        print(flush=True)

    print()
    print("Summary table:")
    print(f"{'cfg':<18} {'mode':<14} {'pop':>4} {'maxGen':>6} {'pos':>7} {'neg':>6}")
    for r in rows:
        print(f"{r['label']:<18} {r['mode']:<14} {r['pop']:>4} {r['max_gen']:>6} {r['pos']:>7} {r['neg']:>6}")


if __name__ == "__main__":
    main()