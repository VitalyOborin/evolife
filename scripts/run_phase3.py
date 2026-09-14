"""Phase 3 - CONTROL vs MEMORY_ECOLOGY comparison.

Two branches run with identical mutation rates, population caps and
seed numbers. They differ only in the world:

  CONTROL          : legacy single-resource World (Phase 1.5 world)
  MEMORY_ECOLOGY   : two-resource + hidden-season MemoryEcologyWorld

The script records per-tick summary lines plus per-seed metrics to
SQLite. Aggregate report is left for the user; this runner just
produces the data.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import numpy as np

from evolife.config import (
    ADD_CONNECTION_RATE,
    ADD_NODE_RATE,
    BIAS_MAX,
    BIAS_MUTATION_RATE,
    BIAS_PERTURB_SIGMA,
    CONNECTION_METABOLIC_COST,
    FOOD_ENERGY,
    FOOD_TARGET,
    IDLE_ENERGY_COST,
    INITIAL_LOCOMOTION_BIAS_SIGMA,
    INITIAL_POPULATION,
    INITIAL_WEIGHT_SIGMA,
    MAX_LINEAR_SPEED,
    MAX_TURN_RATE,
    MOVE_DEADZONE,
    MOVE_ENERGY_COST,
    NEURON_METABOLIC_COST,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    SEASON_LENGTH,
    TOGGLE_CONNECTION_RATE,
    TURN_ENERGY_COST,
    WEIGHT_MAX,
    WEIGHT_MUTATION_RATE,
    WEIGHT_PERTURB_RATE,
    WEIGHT_PERTURB_SIGMA,
)
from evolife.metrics import Metrics
from evolife.phase3 import MemoryEcologyWorld
from evolife.world import World


def run_branch(
    branch: str,
    seed: int,
    ticks: int,
    log_every: int,
) -> dict:
    """Run one (branch, seed) to completion. Returns summary dict."""
    if branch == "control":
        world = World(seed=seed)
    elif branch == "memory":
        world = MemoryEcologyWorld(seed=seed)
    else:
        raise ValueError(f"unknown branch: {branch}")

    metrics = Metrics(f"evolife_phase3_{branch}_seed{seed}.sqlite")

    n_births = 0
    n_deaths = 0
    n_eats = 0
    n_repros = 0
    t0 = time.time()
    for t in range(ticks):
        world.step()
        # Count events so far (cheap; world.events grows incrementally).
        n_births = sum(1 for e in world.events.events if e.kind.value == "birth")
        n_deaths = sum(1 for e in world.events.events if e.kind.value == "death")
        n_eats = sum(1 for e in world.events.events if e.kind.value == "eat")
        n_repros = sum(1 for e in world.events.events if e.kind.value == "reproduction")
        # Record archive milestones + cycle carriers each tick.
        metrics.record_archive_milestones(world.archive)
        metrics.record_cycle_carriers(world.archive)
        if t > 0 and t % log_every == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            pop = sum(1 for o in world.organisms if o.alive)
            max_gen = max((o.generation for o in world.organisms), default=0)
            if branch == "memory":
                line = (
                    f"[{branch} s={seed}] t={world.tick:>5} pop={pop:>3} "
                    f"maxGen={max_gen:>3} season={world.season} "
                    f"cycles={len(world.archive.cycle_carriers):>2} "
                    f"({(t+1)/elapsed:>5.1f}/s)"
                )
            else:
                line = (
                    f"[{branch} s={seed}] t={world.tick:>5} pop={pop:>3} "
                    f"maxGen={max_gen:>3} "
                    f"cycles={len(world.archive.cycle_carriers):>2} "
                    f"({(t+1)/elapsed:>5.1f}/s)"
                )
            print(line, flush=True)

    elapsed = time.time() - t0
    pop = sum(1 for o in world.organisms if o.alive)
    max_gen = max((o.generation for o in world.organisms), default=0)
    n_cycles = len(world.archive.cycle_carriers)
    mean_energy = (
        float(np.mean([o.energy for o in world.organisms])) if world.organisms else 0.0
    )
    metrics.close()
    return {
        "branch": branch,
        "seed": seed,
        "ticks": world.tick,
        "elapsed_s": elapsed,
        "pop": pop,
        "max_gen": max_gen,
        "mean_energy": mean_energy,
        "n_births": n_births,
        "n_deaths": n_deaths,
        "n_eats": n_eats,
        "n_repros": n_repros,
        "n_cycles": n_cycles,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--ticks", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=2000)
    args = p.parse_args()

    print(f"Phase 3 sweep: ticks={args.ticks} seeds={args.seeds}")
    print(f"  Season length: {SEASON_LENGTH}")
    print(f"  Mutation rates: ADD_NODE={ADD_NODE_RATE}, ADD_CONN={ADD_CONNECTION_RATE}, TOGGLE_CONN={TOGGLE_CONNECTION_RATE}")
    print()

    summaries = []
    for branch in ("control", "memory"):
        for seed in args.seeds:
            print(f"=== Branch {branch}, seed {seed} ===", flush=True)
            summary = run_branch(branch, seed, args.ticks, args.log_every)
            summaries.append(summary)
            print(
                f"=== {branch} s={seed}: pop={summary['pop']}, "
                f"maxGen={summary['max_gen']}, eats={summary['n_eats']}, "
                f"cycles={summary['n_cycles']}, "
                f"{summary['ticks'] / summary['elapsed_s']:.1f}/s ===",
                flush=True,
            )
            print(flush=True)

    # Final summary table.
    print()
    print("Final summary:")
    print(
        f"{'branch':<8} {'seed':>4} {'pop':>4} {'maxGen':>6} {'eats':>6} "
        f"{'repros':>6} {'cycles':>6} {'meanE':>7}"
    )
    for s in summaries:
        print(
            f"{s['branch']:<8} {s['seed']:>4} {s['pop']:>4} {s['max_gen']:>6} "
            f"{s['n_eats']:>6} {s['n_repros']:>6} {s['n_cycles']:>6} "
            f"{s['mean_energy']:>7.1f}"
        )


if __name__ == "__main__":
    sys.exit(main())
