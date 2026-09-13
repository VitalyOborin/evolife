"""Run the v2.2 (Viable Replicator) experiment.

Default: 20 independent seeds × 50 000 ticks. Writes each run's metrics
to evolife_metrics_seed<N>.sqlite and a combined summary to stdout.

Success for this stage is not "smart navigation". It is:
  - median run does not go extinct
  - median max_generation > 30
  - thousands of births
  - population neither pinned to the cap nor collapsing to 0
"""

from __future__ import annotations

import argparse
import statistics
import time

from evolife.metrics import Metrics
from evolife.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife v2.2 experiment")
    parser.add_argument("--ticks", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    args = parser.parse_args()

    pops: list[int] = []
    max_gens: list[int] = []
    births: list[int] = []
    repros: list[int] = []
    eats: list[int] = []

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
            ran = max(world.tick, 1)
            rate = ran / elapsed if elapsed > 0 else float("inf")
            by_kind = world.events.by_kind()
            n_births = len(by_kind["birth"])
            n_deaths = len(by_kind["death"])
            n_eats = len(by_kind["eat"])
            n_repros = len(by_kind["reproduction"])
            pop = world.population()
            max_gen = world.max_generation()
            pops.append(pop)
            max_gens.append(max_gen)
            births.append(n_births)
            repros.append(n_repros)
            eats.append(n_eats)
            print(
                f"seed={seed:>3}  ticks={world.tick}/{args.ticks}  elapsed={elapsed:.1f}s  "
                f"rate={rate:.1f}/s  pop={pop:>3}  "
                f"maxGen={max_gen:>3}  lin={world.n_lineages():>3}  "
                f"meanE={world.mean_energy():>6.1f}  "
                f"births={n_births:>5}  deaths={n_deaths:>5}  "
                f"eats={n_eats:>5}  reproductions={n_repros:>5}",
                flush=True,
            )
            metrics.close()

    def _med(xs: list[int]) -> float:
        return float(statistics.median(xs)) if xs else 0.0

    n_extinct = sum(1 for p in pops if p == 0)
    print(
        f"\nsummary  seeds={len(args.seeds)}  extinct={n_extinct}/{len(args.seeds)}  "
        f"median_pop={_med(pops):.0f}  median_maxGen={_med(max_gens):.0f}  "
        f"median_births={_med(births):.0f}  median_repro={_med(repros):.0f}  "
        f"median_eats={_med(eats):.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
