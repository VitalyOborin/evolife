"""Run the v2 (Natural Selection Baseline) experiment.

Five independent seeds × 50 000 ticks. Writes each run's metrics to
evolife_metrics_seed<N>.sqlite and a combined summary to stdout.
"""

from __future__ import annotations

import argparse
import time

from evolife.metrics import Metrics
from evolife.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife v2 experiment")
    parser.add_argument("--ticks", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    summaries = []
    for seed in args.seeds:
        path = f"evolife_metrics_seed{seed}.sqlite"
        world = World(seed=seed)
        metrics = Metrics(path=path)
        t0 = time.perf_counter()
        try:
            for _ in range(args.ticks):
                world.step()
                metrics.record_world(world)
                metrics.record_organisms(world, world.organisms)
            metrics.flush_events(world.events)
        finally:
            elapsed = time.perf_counter() - t0
            rate = args.ticks / elapsed if elapsed > 0 else float("inf")
            by_kind = world.events.by_kind()
            n_births = len(by_kind["birth"])
            n_deaths = len(by_kind["death"])
            n_eats = len(by_kind["eat"])
            n_repros = len(by_kind["reproduction"])
            summary = (
                f"seed={seed:>3}  ticks={args.ticks}  elapsed={elapsed:.1f}s  "
                f"rate={rate:.1f}/s  pop={world.population():>3}  "
                f"meanE={world.mean_energy():>6.1f}  "
                f"births={n_births:>4}  deaths={n_deaths:>4}  "
                f"eats={n_eats:>4}  reproductions={n_repros:>4}"
            )
            print(summary, flush=True)
            summaries.append(summary)
            metrics.close()


if __name__ == "__main__":
    main()
