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

`--device cuda` (default) runs classic mode on GpuWorld. Phase 3 still
steps on CPU; the smell heatmap uses CUDA when a GPU is present.
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
        "(MemoryEcologyWorld modes only). WITHOUT this the founder is "
        "cold-start and likely goes extinct within 1000 ticks.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="cuda = GpuWorld when mode=classic (falls back to cpu if "
        "CUDA is missing). Smell colormap uses CUDA whenever torch "
        "sees a GPU, including Phase 3 CPU worlds. "
        "MemoryEcologyWorld simulation itself is still CPU.",
    )
    parser.add_argument(
        "--plasticity-alpha", type=float, default=None,
        help="Phase 4 per-tick local Hebbian rate. If None, no Phase 4 "
        "plasticity. Use 0.01 to match run_phase4.py.",
    )
    parser.add_argument(
        "--plasticity-beta", type=float, default=None,
        help="Phase 4 per-tick reward-modulated rate. If None, no Phase 4 "
        "plasticity. Use 0.05 to match run_phase4.py.",
    )
    parser.add_argument(
        "--plasticity-trace", action="store_true",
        help="Enable Phase 4.1 eligibility-trace rHebb (Miconi 2017). "
        "Overrides --plasticity-alpha/beta when set.",
    )
    parser.add_argument(
        "--plasticity-rate", type=float, default=0.005,
        help="Phase 4.1 per-episode commit rate (default 0.005).",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False
        if not cuda_ok:
            print("CUDA unavailable; falling back to --device cpu")
            args.device = "cpu"
        elif args.mode != "classic":
            print(
                "MemoryEcologyWorld has no GPU step yet; "
                "simulating on CPU, colourizing smell on CUDA"
            )

    from evolife.viz_lab import LabVisualizer

    # Configure brain plasticity BEFORE any Brain is constructed.
    # Phase 4.1 (eligibility-trace rHebb) takes precedence over Phase 4.
    import evolife.brain as br
    if args.plasticity_trace:
        br.PLASTICITY_TRACE = True
        br.PLASTICITY_RATE = args.plasticity_rate
        print(
            f"Phase 4.1 plasticity ON (trace=True eta={args.plasticity_rate})"
        )
    elif args.plasticity_alpha is not None or args.plasticity_beta is not None:
        br.PLASTICITY_ALPHA = args.plasticity_alpha or 0.0
        br.PLASTICITY_BETA = args.plasticity_beta or 0.0
        print(
            f"Phase 4 plasticity ON (alpha={br.PLASTICITY_ALPHA} "
            f"beta={br.PLASTICITY_BETA})"
        )

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
            from evolife.gpu_world import GpuWorld
            world = GpuWorld(seed=args.seed, device=torch.device("cuda"))
            viz = LabVisualizer(world)
            metrics = GpuMetrics()
        else:
            from evolife.metrics import Metrics
            from evolife.world import World
            world = World(seed=args.seed)
            viz = LabVisualizer(world)
            metrics = Metrics()
    else:
        from evolife.metrics import Metrics
        from evolife.phase3 import MemoryEcologyWorld
        visible_sensor = (args.mode == "visible_season")
        world = MemoryEcologyWorld(
            seed=args.seed,
            mode=args.mode,
            visible_season_sensor=visible_sensor,
            warm_genome=warm_genome,
        )
        viz = LabVisualizer(world)
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
