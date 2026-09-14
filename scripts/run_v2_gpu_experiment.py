"""Run a multi-seed GpuWorld experiment and collect archive milestones.

Mirrors `run_v2_experiment.py` but uses `evolife.gpu_world.GpuWorld` so we
can compare Phase 0 GPU vs CPU behaviour on the same seeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from evolife.archive import Archive, MilestoneKind
from evolife.gpu_world import GpuWorld


def run_one_seed(
    seed: int,
    n_ticks: int,
    width: int = 512,
    height: int = 512,
    log_every: int = 5000,
) -> dict:
    world = GpuWorld(seed=seed, width=width, height=height)
    archive: Archive = world.archive
    pop_history: list[tuple[int, int]] = []
    t0 = time.perf_counter()
    last = t0
    for _ in range(n_ticks):
        world.step()
        # Periodic pop snapshot.
        pop = int(world.alive_mask.sum().item())
        # Track max nodes seen.
        # Approximate: enabled connections count + 3 sensors + 2 motors.
        # For v2.2 default this is 8 (=5 nodes + 3 conns + 2 motors?).
        # We use archive's first/max as it triggers itself.
        now = time.perf_counter()
        if now - last >= log_every or _ == n_ticks - 1:
            rate = (world.tick) / (now - t0)
            print(
                f"seed={seed:>3}  ticks={world.tick}/{n_ticks}  "
                f"elapsed={now - t0:6.1f}s  rate={rate:6.1f}/s  pop={pop}"
            )
            last = now
        pop_history.append((world.tick, pop))

    elapsed = time.perf_counter() - t0
    # Aggregate archive milestones by kind.
    counts: dict[str, int] = {}
    first_tick_by_kind: dict[str, list[int]] = {}
    for rec in archive.milestones:
        kind_name = rec.kind.name
        counts[kind_name] = counts.get(kind_name, 0) + 1
        first_tick_by_kind.setdefault(kind_name, []).append(rec.tick)
    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "elapsed_s": elapsed,
        "rate_per_s": n_ticks / elapsed,
        "final_pop": int(world.alive_mask.sum().item()),
        "milestone_kind_counts": counts,
        "milestone_first_tick_by_kind": first_tick_by_kind,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ticks", type=int, default=50_000)
    p.add_argument("--out", type=Path, default=Path("evolife_phase0_gpu_archive.json"))
    args = p.parse_args()

    rows: list[dict] = []
    for s in range(1, args.seeds + 1):
        rows.append(run_one_seed(seed=s, n_ticks=args.ticks))

    n_extinct = sum(1 for r in rows if r["final_pop"] == 0)
    pops = sorted(r["final_pop"] for r in rows)
    summary = {
        "seeds": args.seeds,
        "ticks": args.ticks,
        "extinct": n_extinct,
        "median_pop": pops[len(pops) // 2],
    }
    print(f"summary  seeds={args.seeds}  extinct={n_extinct}/{args.seeds}  "
          f"median_pop={summary['median_pop']}")

    payload = {
        "summary": summary,
        "runs": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"archive written to {args.out}")


if __name__ == "__main__":
    main()
