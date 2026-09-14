"""Run EvoLife headless for a fixed number of ticks and write metrics.

Supports both CPU (default) and GPU (`--device cuda`) world. On GPU
the world is `evolife.gpu_world.GpuWorld` and metrics are written via
`evolife.gpu_metrics.GpuMetrics`, which mirrors the same schema but
pulls rows from GPU tensors. GPU runs are ~2x faster per tick but
produce coarser events (only Births are recorded).
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoLife headless run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=10_000)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu = World (full metrics); cuda = GpuWorld (~2x faster, "
        "no Death/Eat/Reproduction events).",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="SQLite path for metrics (default: evolife_metrics.sqlite on "
        "CPU, evolife_metrics_gpu.sqlite on GPU).",
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
        from evolife.gpu_world import GpuWorld

        world = GpuWorld(seed=args.seed, device=torch.device("cuda"))
        path = args.metrics_path or "evolife_metrics_gpu.sqlite"
        metrics = GpuMetrics(path=path)
        start = time.perf_counter()
        try:
            for _ in range(args.ticks):
                world.step()
                metrics.record_gpu_world(world)
                metrics.record_gpu_organisms(world)
                metrics.record_gpu_species(world)
            metrics.flush_gpu_events(world)
        finally:
            elapsed = time.perf_counter() - start
            rate = args.ticks / elapsed if elapsed > 0 else float("inf")
            by_kind = world.events.by_kind()
            print(
                f"[gpu]  ticks={args.ticks}  elapsed={elapsed:.2f}s  "
                f"rate={rate:.1f} ticks/s  "
                f"final_pop={world.population()}  "
                f"max_gen={world.max_generation()}  "
                f"lineages={world.n_lineages()}  "
                f"final_mean_energy={world.mean_energy():.2f}  "
                f"events={len(world.events.events)}  "
                f"births={len(by_kind.get('birth', []))}"
            )
            metrics.close()
    else:
        from evolife.metrics import Metrics
        from evolife.world import World

        world = World(seed=args.seed)
        path = args.metrics_path or "evolife_metrics.sqlite"
        metrics = Metrics(path=path)
        start = time.perf_counter()
        try:
            for _ in range(args.ticks):
                world.step()
                metrics.record_world(world)
                metrics.record_organisms(world, world.organisms)
                metrics.record_species(world)
                metrics.record_behavior(world, world.organisms)
            metrics.flush_events(world.events)
            metrics.flush_species_events(world.species_manager)
        finally:
            elapsed = time.perf_counter() - start
            rate = args.ticks / elapsed if elapsed > 0 else float("inf")
            by_kind = world.events.by_kind()
            print(
                f"[cpu]  ticks={args.ticks}  elapsed={elapsed:.2f}s  "
                f"rate={rate:.1f} ticks/s  "
                f"final_pop={world.population()}  "
                f"max_gen={world.max_generation()}  "
                f"lineages={world.n_lineages()}  "
                f"final_mean_energy={world.mean_energy():.2f}  "
                f"events={len(world.events.events)}  "
                f"births={len(by_kind['birth'])}  "
                f"deaths={len(by_kind['death'])}  "
                f"eats={len(by_kind['eat'])}"
            )
            metrics.close()


if __name__ == "__main__":
    main()
