"""Phase 3.2 smoke that prints population curves + cumulate eats.

Use to see whether the founder over-reproduces (overshoot wave) or
some other demographic problem is killing the population. Run from
the project root with a warm-start genome JSON path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from evolife.genome import Genome
from evolife.phase3 import MemoryEcologyWorld


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["static_dual", "visible_season", "hidden_season"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--warm-json", type=str, default="evolife_regen_warm_parent.json")
    p.add_argument("--initial-pop", type=int, default=None,
                   help="Override INITIAL_POPULATION for this run (debug only).")
    p.add_argument("--repro-threshold", type=float, default=None,
                   help="Override REPRODUCTION_THRESHOLD for this run (debug only).")
    p.add_argument("--initial-energy", type=float, default=None,
                   help="Override PHASE3_INITIAL_ENERGY for this run (debug only).")
    p.add_argument("--season-length", type=int, default=None,
                   help="Override SEASON_LENGTH for this run (debug only).")
    p.add_argument("--neg-energy", type=float, default=None,
                   help="Override FOOD_A/B_NEGATIVE_ENERGY for this run (debug only).")
    args = p.parse_args()

    warm = None
    try:
        with open(args.warm_json) as fh:
            warm = Genome.from_dict(json.load(fh))
        print(f"warm-start from {args.warm_json}: nodes={len(warm.nodes)} conns={len(warm.connections)}", flush=True)
    except FileNotFoundError:
        print(f"warm-json not found: {args.warm_json}; cold-start", flush=True)

    # Debug-only override: tweak INITIAL_POPULATION via world construction
    # by monkey-patching the module constant *before* world creation.
    overrides = []
    if args.initial_pop is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        import evolife.world as w
        cfg.INITIAL_POPULATION = args.initial_pop
        ph3.INITIAL_POPULATION = args.initial_pop
        w.INITIAL_POPULATION = args.initial_pop
        overrides.append(f"INITIAL_POPULATION={args.initial_pop}")
    if args.repro_threshold is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        import evolife.world as w
        cfg.REPRODUCTION_THRESHOLD = args.repro_threshold
        ph3.REPRODUCTION_THRESHOLD = args.repro_threshold
        w.REPRODUCTION_THRESHOLD = args.repro_threshold
        overrides.append(f"REPRODUCTION_THRESHOLD={args.repro_threshold}")
    if args.initial_energy is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        cfg.PHASE3_INITIAL_ENERGY = args.initial_energy
        ph3.PHASE3_INITIAL_ENERGY = args.initial_energy
        overrides.append(f"PHASE3_INITIAL_ENERGY={args.initial_energy}")
    if args.season_length is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        cfg.SEASON_LENGTH = args.season_length
        ph3.SEASON_LENGTH = args.season_length
        overrides.append(f"SEASON_LENGTH={args.season_length}")
    if args.neg_energy is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        cfg.FOOD_A_NEGATIVE_ENERGY = args.neg_energy
        cfg.FOOD_B_NEGATIVE_ENERGY = args.neg_energy
        ph3.FOOD_A_NEGATIVE_ENERGY = args.neg_energy
        ph3.FOOD_B_NEGATIVE_ENERGY = args.neg_energy
        overrides.append(f"NEG_ENERGY={args.neg_energy}")
    if overrides:
        print(f"DEBUG override: {', '.join(overrides)}", flush=True)

    world = MemoryEcologyWorld(
        seed=args.seed, mode=args.mode,
        visible_season_sensor=(args.mode == "visible_season"),
        warm_genome=warm,
    )

    t0 = time.time()
    cum_pos = cum_neg = 0
    for t in range(args.ticks):
        world.step()
        # Per-tick cumulative positive/negative eats.
        for org in world.organisms:
            cum_pos += getattr(org, "positive_eats", 0)
            cum_neg += getattr(org, "negative_eats", 0)
        # We must subtract per-tick contributions from the *previous*
        # log to get per-interval new counts. Track last snapshot.
        if t > 0 and t % args.log_every == 0:
            pop = sum(1 for o in world.organisms if o.alive)
            gen_counts: dict[int, int] = {}
            for o in world.organisms:
                if o.alive:
                    gen_counts[o.generation] = gen_counts.get(o.generation, 0) + 1
            top_gens = sorted(gen_counts.items(), key=lambda kv: -kv[1])[:5]
            print(
                f"t={world.tick:>5} pop={pop:>3} season={world.season} "
                f"foodA={len(world.food_a):>3} foodB={len(world.food_b):>3} "
                f"top_gens={top_gens} "
                f"pos_cum={cum_pos:>6} neg_cum={cum_neg:>5} "
                f"({(t+1)/(time.time()-t0):>5.1f}/s)",
                flush=True,
            )

    elapsed = time.time() - t0
    pop = sum(1 for o in world.organisms if o.alive)
    max_gen = max((o.generation for o in world.organisms), default=0)
    print(
        f"\n=== {args.mode} s={args.seed}: pop={pop} maxGen={max_gen} "
        f"pos_cum={cum_pos} neg_cum={cum_neg} "
        f"{args.ticks/elapsed:.1f}/s ===",
        flush=True,
    )


if __name__ == "__main__":
    main()