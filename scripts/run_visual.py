"""Run EvoLife with Pygame visualisation."""

from __future__ import annotations

import argparse

from evolife.metrics import Metrics
from evolife.visualization import Visualizer
from evolife.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife visual run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=10_000,
                        help="Cap on ticks; 0 means run until window is closed.")
    args = parser.parse_args()

    world = World(seed=args.seed)
    viz = Visualizer(world)
    metrics = Metrics()

    try:
        cap = args.ticks if args.ticks > 0 else None
        while True:
            if cap is not None and world.tick >= cap:
                break
            world.step()
            metrics.record_world(world)
            metrics.record_organisms(world, world.organisms)
            metrics.record_species(world)
            metrics.record_behavior(world, world.organisms)
            viz.render()
    finally:
        metrics.close()
        viz.close()


if __name__ == "__main__":
    main()
