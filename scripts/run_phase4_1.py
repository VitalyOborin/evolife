"""Phase 4.1 sweep runner -- eligibility-trace rHebb (Miconi 2017).

Same 3-mode setup as run_phase4.py, but enables PLASTICITY_TRACE
in brain.py and uses the Miconi-style 3-factor rule:

  e_ij(t) = lambda * e_ij(t-1) + S(pre_i * (post_j - <pre*post>_ema))
  Delta_w_ij = eta * (R - R_b) * e_ij(T)   [committed at episode end]

Episodes end at starvation or reproduction. R is the lifetime reward
(positive eats - |negative eats|); R_b is a per-organism running-mean
baseline (EMA) so the rule learns from reward prediction error rather
than absolute reward.

This addresses the sparse-credit-assignment gap identified in
PHASE3_4_RESULTS and PHASE4_LIT: Phase 4's per-tick 2-factor rule
applied reward directly without an eligibility trace, so the Hebbian
signal could not propagate back through time. rHebb does, by
construction.

Defaults chosen for fast convergence in a closed-loop simulation:
  ELIGIBILITY_DECAY      = 0.95   # traces survive ~20 ticks
  ELIGIBILITY_BASELINE_EMA = 0.99
  SUPERLINEAR_POWER      = 3      # Miconi's x^3
  PLASTICITY_RATE        = 0.005  # per-episode commit
  REWARD_BASELINE_EMA    = 0.99
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
    metrics = Metrics(f"evolife_phase4_1_{mode}_seed{seed}.sqlite")
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
            # Average reward baseline across living organisms -- a useful
            # proxy for whether learning is actually happening (R_b
            # tracks running reward).
            r_b_avg = 0.0
            r_b_n = 0
            for o in world.organisms:
                if o.alive and o.brain is not None and hasattr(o.brain, "_r_baseline"):
                    r_b_avg += o.brain._r_baseline
                    r_b_n += 1
            r_b_avg = r_b_avg / r_b_n if r_b_n else 0.0
            print(
                f"[{mode} s={seed}] t={world.tick:>5} pop={pop:>3} "
                f"season={world.season} cycles={len(world.archive.cycle_carriers):>2} "
                f"rec_frac={rec_frac:.2f} R_b={r_b_avg:+.2f} "
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
    p.add_argument("--eligibility-decay", type=float, default=0.95,
                   help="Trace decay lambda (default 0.95).")
    p.add_argument("--baseline-ema", type=float, default=0.99,
                   help="Per-edge <pre*post> EMA coefficient (default 0.99).")
    p.add_argument("--superlinear-power", type=int, default=3,
                   help="Exponent for superlinear non-linearity S(x)=sign(x)|x|^k (default 3).")
    p.add_argument("--plasticity-rate", type=float, default=0.005,
                   help="eta in Delta_w = eta (R - Rb) e (default 0.005).")
    p.add_argument("--reward-baseline-ema", type=float, default=0.99,
                   help="EMA for organism-level R_b (default 0.99).")
    p.add_argument("--neg-energy", type=float, default=None,
                   help="Override PHASE3_FOOD_*_NEGATIVE_ENERGY for this run.")
    args = p.parse_args()

    # Apply Phase 4.1 plasticity constants.
    import evolife.brain as br
    br.PLASTICITY_TRACE = True
    br.ELIGIBILITY_DECAY = args.eligibility_decay
    br.ELIGIBILITY_BASELINE_EMA = args.baseline_ema
    br.SUPERLINEAR_POWER = args.superlinear_power
    br.PLASTICITY_RATE = args.plasticity_rate
    br.REWARD_BASELINE_EMA = args.reward_baseline_ema

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

    print(f"Phase 4.1 sweep: ticks={args.ticks} seeds={args.seeds} "
          f"lam={args.eligibility_decay} k={args.superlinear_power} "
          f"eta={args.plasticity_rate} Rb_ema={args.reward_baseline_ema}")
    print("PLASTICITY_TRACE=ON (Miconi 2017 rHebb)")
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
