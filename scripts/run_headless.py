"""Run EvoLife headless for a fixed number of ticks and write metrics."""

from __future__ import annotations

import argparse
import time

from evolife.metrics import Metrics
from evolife.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife headless run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=10_000)
    args = parser.parse_args()

    world = World(seed=args.seed)
    metrics = Metrics()
    start = time.perf_counter()
    try:
        for _ in range(args.ticks):
            world.step()
            metrics.record_world(world)
            metrics.record_organisms(world, world.organisms)
    finally:
        elapsed = time.perf_counter() - start
        rate = args.ticks / elapsed if elapsed > 0 else float("inf")
        print(
            f"ticks={args.ticks}  elapsed={elapsed:.2f}s  "
            f"rate={rate:.1f} ticks/s  "
            f"final_pop={world.population()}  "
            f"final_mean_energy={world.mean_energy():.2f}"
        )
        metrics.close()


if __name__ == "__main__":
    main()
