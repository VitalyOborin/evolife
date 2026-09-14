"""Phase 3.1 - controlled Memory Ecology runner.

Three worlds, identical mutation rates, identical warm-start founder:

  static_dual     : two food types, no season. Control: is recurrence
                    selected just by having two distinct food types?
  visible_season  : two food types + season flip + season sensor.
                    Control: is recurrence selected just by a visible
                    season?
  hidden_season   : two food types + season flip, no season sensor, only
                    post-eat feedback. Hypothesis test.

The runner prints per-seed summaries. Memory advantage + adaptation
latency metrics are computed in scripts/arena_memory_advantage.py.
"""
from __future__ import annotations

import argparse
import sys
import time

from evolife.metrics import Metrics
from evolife.phase3 import MemoryEcologyWorld


def run_branch(
    mode: str,
    seed: int,
    ticks: int,
    log_every: int,
    warm_genome=None,
    visible_season_sensor: bool = False,
) -> dict:
    world = MemoryEcologyWorld(
        seed=seed, mode=mode,
        visible_season_sensor=visible_season_sensor,
        warm_genome=warm_genome,
    )
    metrics = Metrics(f"evolife_phase31_{mode}_seed{seed}.sqlite")

    t0 = time.time()
    n_pos, n_neg = 0, 0
    for t in range(ticks):
        world.step()
        # Aggregate positive / negative eats over the live population.
        for org in world.organisms:
            n_pos += getattr(org, "positive_eats", 0)
            n_neg += getattr(org, "negative_eats", 0)
        metrics.record_archive_milestones(world.archive)
        metrics.record_cycle_carriers(world.archive)
        if t > 0 and t % log_every == 0:
            elapsed = time.time() - t0
            pop = sum(1 for o in world.organisms if o.alive)
            n_alive_with_recurrence = sum(
                1 for o in world.organisms
                if o.alive and any(
                    c.in_node in {n.id for n in o.genome.nodes.values() if n.type.value == "hidden"}
                    and c.out_node in {n.id for n in o.genome.nodes.values() if n.type.value == "hidden"}
                    for c in o.genome.connections.values()
                    if c.enabled
                )
            )
            rec_frac = (n_alive_with_recurrence / pop) if pop else 0.0
            print(
                f"[{mode} s={seed}] t={world.tick:>5} pop={pop:>3} "
                f"season={world.season} cycles={len(world.archive.cycle_carriers):>2} "
                f"rec_frac={rec_frac:.2f} "
                f"({(t+1)/elapsed:>5.1f}/s)",
                flush=True,
            )

    elapsed = time.time() - t0
    pop = sum(1 for o in world.organisms if o.alive)
    max_gen = max((o.generation for o in world.organisms), default=0)
    pos_total = sum(getattr(o, "positive_eats", 0) for o in world.organisms)
    neg_total = sum(getattr(o, "negative_eats", 0) for o in world.organisms)
    metrics.close()
    return {
        "mode": mode,
        "seed": seed,
        "ticks": world.tick,
        "elapsed_s": elapsed,
        "pop": pop,
        "max_gen": max_gen,
        "positive_eats": pos_total,
        "negative_eats": neg_total,
        "n_cycles": len(world.archive.cycle_carriers),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ticks", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--warm-json", type=str,
                   default="evolife_phase31_warm_parent.json",
                   help="Warm-start genome JSON (default: parent of first "
                        "cycle_carrier in seed 1, no recurrent edges)")
    p.add_argument("--neg-energy", type=float, default=None,
                   help="Override PHASE3_FOOD_A_NEGATIVE_ENERGY and "
                        "PHASE3_FOOD_B_NEGATIVE_ENERGY for this run. "
                        "Use -1 to soften the season punishment so the "
                        "founder has more runway for evolution to find "
                        "recurrence.")
    p.add_argument("--repro-threshold", type=float, default=None,
                   help="Override PHASE3_REPRODUCTION_THRESHOLD for this "
                        "run. Higher values slow reproduction.")
    args = p.parse_args()

    import json
    from evolife.genome import Genome
    warm = None
    if args.warm_json:
        try:
            with open(args.warm_json) as fh:
                warm = Genome.from_dict(json.load(fh))
            print(f"warm-start from {args.warm_json}: "
                  f"nodes={len(warm.nodes)} conns={len(warm.connections)}")
        except FileNotFoundError:
            print(f"warm-json not found: {args.warm_json}, using cold-start")

    print(f"Phase 3.1 sweep: ticks={args.ticks} seeds={args.seeds}")

    # Apply optional overrides to the Phase 3 demographic constants.
    if args.neg_energy is not None or args.repro_threshold is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        overrides = []
        if args.neg_energy is not None:
            cfg.PHASE3_FOOD_A_NEGATIVE_ENERGY = args.neg_energy
            cfg.PHASE3_FOOD_B_NEGATIVE_ENERGY = args.neg_energy
            ph3.PHASE3_FOOD_A_NEGATIVE_ENERGY = args.neg_energy
            ph3.PHASE3_FOOD_B_NEGATIVE_ENERGY = args.neg_energy
            overrides.append(f"PHASE3_NEG={args.neg_energy}")
        if args.repro_threshold is not None:
            cfg.PHASE3_REPRODUCTION_THRESHOLD = args.repro_threshold
            ph3.PHASE3_REPRODUCTION_THRESHOLD = args.repro_threshold
            overrides.append(f"PHASE3_REPRO_THRESH={args.repro_threshold}")
        print(f"overrides applied: {', '.join(overrides)}", flush=True)
    print("Three worlds (static_dual / visible_season / hidden_season):")
    print()

    summaries = []
    for mode in ("static_dual", "visible_season", "hidden_season"):
        for seed in args.seeds:
            vss = (mode == "visible_season")
            print(f"=== Mode {mode}, seed {seed} ===", flush=True)
            summary = run_branch(
                mode, seed, args.ticks, args.log_every,
                warm_genome=warm, visible_season_sensor=vss,
            )
            summaries.append(summary)
            print(
                f"=== {mode} s={seed}: pop={summary['pop']} maxGen={summary['max_gen']} "
                f"pos={summary['positive_eats']} neg={summary['negative_eats']} "
                f"cycles={summary['n_cycles']} "
                f"{summary['ticks']/summary['elapsed_s']:.1f}/s ===",
                flush=True,
            )
            print(flush=True)

    print()
    print("Final summary:")
    print(f"{'mode':<14} {'seed':>4} {'pop':>4} {'maxGen':>6} "
          f"{'pos':>6} {'neg':>6} {'cycles':>6}")
    for s in summaries:
        print(f"{s['mode']:<14} {s['seed']:>4} {s['pop']:>4} {s['max_gen']:>6} "
              f"{s['positive_eats']:>6} {s['negative_eats']:>6} {s['n_cycles']:>6}")


if __name__ == "__main__":
    sys.exit(main())
