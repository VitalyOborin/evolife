"""Run EvoLife with Pygame visualisation.

Supports four worlds:
  --mode classic         : legacy single-resource World (Phase 1.5 baseline)
  --mode static_dual     : MemoryEcologyWorld, two food types, no season
                           (Phase 3.1 control)
  --mode visible_season  : MemoryEcologyWorld + season flip + season sensor
                           (Phase 3.1 control)
  --mode hidden_season   : MemoryEcologyWorld + season flip, no sensor
                           (Phase 3.1 test -- Phase 3.0 hot path)

Warm-start founder is supported for MemoryEcologyWorld via
--warm-json (a Phase 1.5 genome JSON). The brain keeps navigating
and evolution can then specialise A vs B.

`--device` is `cpu` only for Phase 3 modes (GpuWorld has no
MemoryEcology counterpart yet).
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife visual run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ticks", type=int, default=10_000,
        help="Cap on ticks; 0 means run until window is closed.",
    )
    parser.add_argument(
        "--mode",
        choices=("classic", "static_dual", "visible_season", "hidden_season"),
        default="classic",
        help="classic = legacy single-resource World. "
        "The other three are MemoryEcologyWorld variants (Phase 3.1).",
    )
    parser.add_argument(
        "--warm-json", type=str, default=None,
        help="Path to a Phase 1.5 genome JSON for warm-start "
        "(MemoryEcologyWorld modes only).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu = World + Visualizer (full species colouring). "
        "cuda = GpuWorld + GpuVisualizer (lineage colouring, no inspector). "
        "MemoryEcologyWorld modes force cpu.",
    )
    args = parser.parse_args()

    if args.mode != "classic" and args.device == "cuda":
        print("MemoryEcologyWorld has no GPU backend yet; forcing --device cpu")
        args.device = "cpu"

    warm_genome = None
    if args.warm_json:
        from evolife.genome import Genome
        with open(args.warm_json) as fh:
            warm_genome = Genome.from_dict(json.load(fh))
        print(
            f"warm-start from {args.warm_json}: "
            f"nodes={len(warm_genome.nodes)} "
            f"conns={len(warm_genome.connections)}"
        )

    if args.mode == "classic":
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
    else:
        from evolife.metrics import Metrics
        from evolife.phase3 import MemoryEcologyWorld
        from evolife.phase3_visualization import Phase3Visualizer
        visible_sensor = (args.mode == "visible_season")
        world = MemoryEcologyWorld(
            seed=args.seed,
            mode=args.mode,
            visible_season_sensor=visible_sensor,
            warm_genome=warm_genome,
        )
        viz = Phase3Visualizer(world)
        metrics = Metrics()

    try:
        cap = args.ticks if args.ticks > 0 else None
        while True:
            if cap is not None and world.tick >= cap:
                break
            world.step()
            if args.mode == "classic" and args.device == "cuda":
                metrics.record_gpu_world(world)
                metrics.record_gpu_organisms(world)
                metrics.record_gpu_species(world)
            elif args.mode == "classic":
                metrics.record_world(world)
                metrics.record_organisms(world, world.organisms)
                metrics.record_species(world)
                metrics.record_behavior(world, world.organisms)
            else:
                # Phase 3 metrics: archive + cycle carriers.
                metrics.record_archive_milestones(world.archive)
                metrics.record_cycle_carriers(world.archive)
            viz.render()
    finally:
        metrics.close()
        viz.close()


if __name__ == "__main__":
    main()
