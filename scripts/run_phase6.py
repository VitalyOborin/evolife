"""Phase 6 sweep -- social marker field (colonies).

Like run_phase3_1.py, but enables Phase 6 COLONY_MARKER_* via a
single --markers flag. The founder gets 3 extra sensor nodes wired
into the hidden node, and organisms emit a marker at every positive
eat. Markers diffuse and decay over time.

This is the test arm for the hypothesis: if colonies solve the
hidden_season memory task without recurrence, the answer to "does
evolution grow memory under minimal assumptions?" becomes "not in
the brain -- it grows in the environment".

Usage:
    python scripts/run_phase6.py --markers --seeds 1 2 3 \\
        --ticks 8000 --warm-json evolife_regen_warm_parent.json

Use --no-markers (default) for a Phase 3.1 baseline control arm.
"""
from __future__ import annotations

import argparse
import sys
import time

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
    metrics = Metrics(f"evolife_phase6_{mode}_seed{seed}.sqlite")
    t0 = time.time()
    pos_total = 0
    neg_total = 0
    marker_total = 0.0
    for t in range(ticks):
        world.step()
        for org in world.organisms:
            pos_total += getattr(org, "positive_eats", 0)
            neg_total += getattr(org, "negative_eats", 0)
        # Track the marker field's total intensity as a coarse
        # indicator of social information density.
        marker_total += float(world.marker_field.grid.sum())
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
            marker_count = int((world.marker_field.grid > 0.01).sum())
            print(
                f"[{mode} s={seed}] t={world.tick:>5} pop={pop:>3} "
                f"season={world.season} cycles={len(world.archive.cycle_carriers):>2} "
                f"rec_frac={rec_frac:.2f} markers={marker_count:>5} "
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
        "marker_cumsum": marker_total,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ticks", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--warm-json", type=str,
                   default="evolife_regen_warm_parent.json")
    p.add_argument("--markers", action="store_true",
                   help="Enable Phase 6 colony markers (default off).")
    p.add_argument("--marker-emit", type=float, default=1.0,
                   help="Marker amount per positive eat (default 1.0).")
    p.add_argument("--marker-decay", type=float, default=0.95,
                   help="Per-tick decay factor (default 0.95).")
    p.add_argument("--marker-diffusion", type=float, default=0.3,
                   help="Diffusion blend (default 0.3).")
    p.add_argument("--neg-energy", type=float, default=None,
                   help="Override PHASE3_FOOD_*_NEGATIVE_ENERGY.")
    p.add_argument("--grace-ticks", type=int, default=None,
                   help="Override PHASE3_GRACE_TICKS (Phase 5).")
    args = p.parse_args()

    # Configure colony markers.
    import evolife.colony as col_mod
    import evolife.config as cfg
    import evolife.phase3 as ph3
    cfg.COLONY_MARKER_OFF = not args.markers
    col_mod.COLONY_MARKER_OFF = not args.markers
    cfg.COLONY_MARKER_EMIT = args.marker_emit
    cfg.COLONY_MARKER_DECAY = args.marker_decay
    cfg.COLONY_MARKER_DIFFUSION = args.marker_diffusion
    col_mod.COLONY_MARKER_DECAY = args.marker_decay
    col_mod.COLONY_MARKER_DIFFUSION = args.marker_diffusion
    col_mod.COLONY_MARKER_EMIT = args.marker_emit
    print(
        f"COLONY_MARKER_OFF={cfg.COLONY_MARKER_OFF} "
        f"(emit={args.marker_emit} decay={args.marker_decay} "
        f"diff={args.marker_diffusion})"
    )

    overrides = []
    if args.neg_energy is not None:
        cfg.PHASE3_FOOD_A_NEGATIVE_ENERGY = args.neg_energy
        cfg.PHASE3_FOOD_B_NEGATIVE_ENERGY = args.neg_energy
        ph3.PHASE3_FOOD_A_NEGATIVE_ENERGY = args.neg_energy
        ph3.PHASE3_FOOD_B_NEGATIVE_ENERGY = args.neg_energy
        overrides.append(f"PHASE3_NEG={args.neg_energy}")
    if args.grace_ticks is not None:
        cfg.PHASE3_GRACE_TICKS = args.grace_ticks
        ph3.PHASE3_GRACE_TICKS = args.grace_ticks
        overrides.append(f"PHASE3_GRACE_TICKS={args.grace_ticks}")
    if overrides:
        print(f"overrides applied: {', '.join(overrides)}", flush=True)

    warm = None
    if args.warm_json:
        import json
        from evolife.genome import Genome
        try:
            with open(args.warm_json) as fh:
                warm = Genome.from_dict(json.load(fh))
            print(
                f"warm-start from {args.warm_json}: "
                f"nodes={len(warm.nodes)} conns={len(warm.connections)}"
            )
        except FileNotFoundError:
            print(f"warm-json not found: {args.warm_json}, using cold-start")

    print(
        f"Phase 6 sweep: ticks={args.ticks} seeds={args.seeds} "
        f"markers={'ON' if args.markers else 'OFF'}"
    )
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
                f"=== {mode} s={seed}: pop={summary['pop']} "
                f"maxGen={summary['max_gen']} pos={summary['positive_eats']} "
                f"neg={summary['negative_eats']} cycles={summary['n_cycles']} "
                f"marker_total={summary['marker_cumsum']:.0f} "
                f"{summary['ticks']/summary['elapsed_s']:.1f}/s ===",
                flush=True,
            )
            print(flush=True)

    print()
    print("Final summary:")
    print(
        f"{'mode':<14} {'seed':>4} {'pop':>4} {'maxGen':>6} "
        f"{'pos':>7} {'neg':>7} {'cycles':>6} {'marker_total':>14}"
    )
    for s in summaries:
        print(
            f"{s['mode']:<14} {s['seed']:>4} {s['pop']:>4} {s['max_gen']:>6} "
            f"{s['positive_eats']:>7} {s['negative_eats']:>7} {s['n_cycles']:>6} "
            f"{s['marker_cumsum']:>14.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
