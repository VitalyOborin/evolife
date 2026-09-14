"""Run EvoLife with Pygame visualisation.

Supports both CPU (default) and GPU (`--device cuda`) world. On GPU
the renderer is `evolife.gpu_visualization.GpuVisualizer`, which draws
from GPU tensors and uses lineage-based colouring (no species
inspector — GpuWorld has no SpeciesManager).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife visual run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=10_000,
                        help="Cap on ticks; 0 means run until window is closed.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu = World + Visualizer (full species colouring). "
        "cuda = GpuWorld + GpuVisualizer (lineage colouring, no inspector).",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() "
                "is False"
            )
        from evolife.gpu_metrics import GpuMetrics
        from evolife.gpu_visualization import GpuVisualizer
        from evolife.gpu_world import GpuWorld

        world = GpuWorld(seed=args.seed, device=torch.device("cuda"))
        viz = GpuVisualizer(world)
        metrics = GpuMetrics()
    else:
        from evolife.metrics import Metrics
        from evolife.visualization import Visualizer
        from evolife.world import World

        world = World(seed=args.seed)
        viz = Visualizer(world)
        metrics = Metrics()

    try:
        cap = args.ticks if args.ticks > 0 else None
        while True:
            if cap is not None and world.tick >= cap:
                break
            world.step()
            if args.device == "cuda":
                metrics.record_gpu_world(world)
                metrics.record_gpu_organisms(world)
                metrics.record_gpu_species(world)
            else:
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
