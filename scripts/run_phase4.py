"""Phase 4 sweep runner -- lifetime synaptic plasticity + reward modulation.

Same 3-mode setup as run_phase3_1.py, but with PLASTICITY_ALPHA / BETA
non-zero so each organism's brain updates within its lifetime based
on the intake_feedback signal.

Expected outcome: in hidden_season, the brain should learn to track
which season it's in (via reward-modulated Hebbian updates on the
intake_feedback sensor). Recurrence that helps with this task
should now be selected for, where Phase 3 showed only drift-level
recurrence.
"""
from __future__ import annotations

import argparse
import sys
import time

# Phase 4 plasticity constants -- importable so the brain picks them
# up at forward time.
from evolife.brain import PLASTICITY_ALPHA, PLASTICITY_BETA  # noqa: F401
from evolife.metrics import Metrics
from evolife.phase3 import MemoryEcologyWorld


def run_branch(
    mode: str, seed: int, ticks: int, log_every: int, warm_genome=None,
    visible_season_sensor: bool = False,
) -> dict:
    world = MemoryEcologyWorld(
        seed=seed, mode=mode,
        visible_season_sensor=visible_season_sensor,
        warm_genome=warm_genome,
    )
    metrics = Metrics(f"evolife_phase4_{mode}_seed{seed}.sqlite")
    t0 = time.time()
    pos_total = 0
    neg_total = 0
    for t in range(ticks):
        world.step()
        for org in world.organisms:
            pos_total += getattr(org, "positive_eats", 0)
            neg_total += getattr(org, "negative_eats", 0)
        metrics.record_archive_milestones(world.archive)
        metrics.record_cycle_carriers(world.archive)
        if t > 0 and t % log_every == 0:
            elapsed = time.time() - t0
            pop = sum(1 for o in world.organisms if o.alive)
            n_rec = sum(
                1 for o in world.organisms
                if o.alive and any(
                    c.in_node in {n.id for n in o.genome.nodes.values() if n.type.value == "hidden"}
                    and c.out_node in {n.id for n in o.genome.nodes.values() if n.type.value == "hidden"}
                    for c in o.genome.connections.values()
                    if c.enabled
                )
            )
            rec_frac = (n_rec / pop) if pop else 0.0
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
    metrics.close()
    return {
        "mode": mode, "seed": seed, "ticks": world.tick,
        "elapsed_s": elapsed, "pop": pop, "max_gen": max_gen,
        "positive_eats": pos_total, "negative_eats": neg_total,
        "n_cycles": len(world.archive.cycle_carriers),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ticks", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--warm-json", type=str,
                   default="evolife_regen_warm_parent.json")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Local Hebbian learning rate (default 0.01).")
    p.add_argument("--beta", type=float, default=0.05,
                   help="Reward-modulated learning rate (default 0.05).")
    p.add_argument("--neg-energy", type=float, default=None,
                   help="Override PHASE3_FOOD_*_NEGATIVE_ENERGY for this run.")
    args = p.parse_args()

    # Set plasticity constants in the brain module.
    import evolife.brain as br
    br.PLASTICITY_ALPHA = args.alpha
    br.PLASTICITY_BETA = args.beta

    # Optional demographic overrides (same pattern as run_phase3_1.py).
    if args.neg_energy is not None:
        import evolife.config as cfg
        import evolife.phase3 as ph3
        cfg.PHASE3_FOOD_A_NEGATIVE_ENERGY = args.neg_energy
        cfg.PHASE3_FOOD_B_NEGATIVE_ENERGY = args.neg_energy
        ph3.PHASE3_FOOD_A_NEGATIVE_ENERGY = args.neg_energy
        ph3.PHASE3_FOOD_B_NEGATIVE_ENERGY = args.neg_energy
        print(f"override PHASE3_NEG={args.neg_energy}", flush=True)

    warm = None
    if args.warm_json:
        import json
        from evolife.genome import Genome
        try:
            with open(args.warm_json) as fh:
                warm = Genome.from_dict(json.load(fh))
            print(f"warm-start from {args.warm_json}: "
                  f"nodes={len(warm.nodes)} conns={len(warm.connections)}")
        except FileNotFoundError:
            print(f"warm-json not found: {args.warm_json}, using cold-start")

    print(f"Phase 4 sweep: ticks={args.ticks} seeds={args.seeds} "
          f"alpha={args.alpha} beta={args.beta}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())