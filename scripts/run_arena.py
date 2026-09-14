"""Run the Behavioral Arena on a living CPU world.

Workflow:
  1. Spin up a CPU `World` and run it for N evolutionary ticks.
  2. After evolution finishes, sample one frozen genome per requested
     generation (default 0, 5, 10, 20, 30, ..., up to max_gen).
  3. For each (genome, scenario) pair, run N_SEEDS episodes of
     N_TICKS ticks each.
  4. Write per-episode results to SQLite and a CSV summary grouped by
     (generation, scenario).

This script does NOT modify the living world — it only reads genomes.
It is observational: no fitness function, no selection.

Usage:
  python scripts/run_arena.py \\
      --ticks 50000 --seed 42 \\
      --arena-ticks 1000 --arena-seeds 20 \\
      --generations 0,5,10,20,30,40,50 \\
      --scenarios uniform,ahead,behind,left_right,sparse,dense,relocating,smell_blanked \\
      --out evolife_arena.sqlite
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

from evolife.arena import (
    ARENA_N_SEEDS_DEFAULT,
    ARENA_N_TICKS_DEFAULT,
    SCENARIO_BUILDERS,
    aggregate,
    brain_shape,
    run_episode,
    sample_genomes_by_generation,
)
from evolife.world import World


_SCHEMA = """
CREATE TABLE IF NOT EXISTS arena_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario TEXT NOT NULL,
    seed INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    brain_nodes INTEGER NOT NULL,
    brain_connections INTEGER NOT NULL,
    food_eaten INTEGER NOT NULL,
    ticks_to_first_food INTEGER,
    ticks_alive INTEGER NOT NULL,
    final_energy REAL NOT NULL,
    distance_traveled REAL NOT NULL,
    mean_speed REAL NOT NULL,
    mean_speed_while_moving REAL NOT NULL,
    moving_fraction REAL NOT NULL,
    movement_transitions INTEGER NOT NULL,
    mean_rest_bout REAL NOT NULL,
    mean_move_bout REAL NOT NULL,
    longest_rest INTEGER NOT NULL,
    longest_move INTEGER NOT NULL,
    mean_abs_turn REAL NOT NULL,
    turns_per_distance REAL NOT NULL,
    exploration_rate REAL NOT NULL,
    steering_alignment REAL NOT NULL,
    state_dependence REAL NOT NULL,
    bias_over_weights REAL NOT NULL,
    energy_efficiency REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS arena_gen_idx ON arena_episodes(generation);
CREATE INDEX IF NOT EXISTS arena_scn_idx ON arena_episodes(scenario);
"""


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def default_generations(max_gen: int) -> list[int]:
    """Sample by generation: 0, 5, 10, 20, 30, ..., up to max_gen.

    Uses a denser spacing at the start (where structural evolution is
    rare) and coarser spacing later.
    """
    gens = [0]
    for g in (5, 10, 20, 30, 40, 50, 75, 100, 150, 200):
        if g <= max_gen:
            gens.append(g)
    if max_gen > 0 and max_gen not in gens:
        gens.append(max_gen)
    return sorted(set(gens))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=50_000)
    p.add_argument("--generations", type=str, default=None,
                   help="comma-separated generations to sample; default = "
                   "auto (0, 5, 10, 20, 30, ..., up to max_gen).")
    p.add_argument("--scenarios", type=str,
                   default="uniform,ahead,behind,left_right,sparse,"
                           "dense,relocating,smell_blanked")
    p.add_argument("--arena-seeds", type=int, default=ARENA_N_SEEDS_DEFAULT)
    p.add_argument("--arena-ticks", type=int, default=ARENA_N_TICKS_DEFAULT)
    p.add_argument("--out", type=Path, default=Path("evolife_arena.sqlite"))
    p.add_argument("--csv", type=Path, default=Path("evolife_arena_summary.csv"))
    args = p.parse_args()

    # 1. Living evolution.
    print(f"[arena] running evolution: ticks={args.ticks}, seed={args.seed}")
    world = World(seed=args.seed)
    t0 = time.perf_counter()
    for _ in range(args.ticks):
        world.step()
    evo_elapsed = time.perf_counter() - t0
    max_gen = world.max_generation()
    print(
        f"[arena] evolution done: elapsed={evo_elapsed:.1f}s "
        f"final_pop={world.population()} maxGen={max_gen}"
    )

    # 2. Sample genomes by generation.
    if args.generations is not None:
        gens = parse_int_list(args.generations)
    else:
        gens = default_generations(max_gen)
    print(f"[arena] sampling generations: {gens}")
    by_gen = sample_genomes_by_generation(world, gens)
    if not by_gen:
        print("[arena] no living genomes to sample, exiting")
        return

    scenarios = parse_str_list(args.scenarios)
    for s in scenarios:
        if s not in SCENARIO_BUILDERS:
            raise SystemExit(f"unknown scenario: {s}")

    # 3. Open SQLite.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.out)
    conn.executescript(_SCHEMA)

    # 4. Run all (genome, scenario, seed) episodes.
    total = len(by_gen) * len(scenarios) * args.arena_seeds
    done = 0
    t1 = time.perf_counter()
    rows = []
    summary_rows = []
    for g, genome in sorted(by_gen.items()):
        nodes, conns = brain_shape(genome)
        for sc in scenarios:
            per_seed = []
            for s in range(args.arena_seeds):
                res = run_episode(
                    genome, sc, seed=s, n_ticks=args.arena_ticks
                )
                per_seed.append(res)
                rows.append(
                    (
                        sc,
                        s,
                        g,
                        nodes,
                        conns,
                        res.food_eaten,
                        res.ticks_to_first_food,
                        res.ticks_alive,
                        res.final_energy,
                        res.distance_traveled,
                        res.mean_speed,
                        res.mean_speed_while_moving,
                        res.moving_fraction,
                        res.movement_transitions,
                        res.mean_rest_bout,
                        res.mean_move_bout,
                        res.longest_rest,
                        res.longest_move,
                        res.mean_abs_turn,
                        res.turns_per_distance,
                        res.exploration_rate,
                        res.steering_alignment,
                        res.state_dependence,
                        res.bias_over_weights,
                        res.energy_efficiency,
                    )
                )
                done += 1
            agg = aggregate(per_seed)
            summary_rows.append(
                {
                    "generation": g,
                    "scenario": sc,
                    "n_seeds": args.arena_seeds,
                    **agg,
                }
            )
            print(
                f"  gen={g:>3}  sc={sc:<14}  "
                f"food={agg['food_eaten_mean']:.2f}±{agg['food_eaten_std']:.2f}  "
                f"steer={agg['steering_alignment_mean']:.3f}  "
                f"state_dep={agg['state_dependence_mean']:.3f}  "
                f"eff={agg['energy_efficiency_mean']:.2f}  "
                f"({done}/{total})"
            )
    arena_elapsed = time.perf_counter() - t1
    print(f"[arena] arena done: elapsed={arena_elapsed:.1f}s "
          f"({done} episodes, {done/arena_elapsed:.1f}/s)")

    # 5. Write SQLite.
    conn.executemany(
        "INSERT INTO arena_episodes ("
        "scenario, seed, generation, brain_nodes, brain_connections, "
        "food_eaten, ticks_to_first_food, ticks_alive, final_energy, "
        "distance_traveled, mean_speed, mean_speed_while_moving, "
        "moving_fraction, movement_transitions, mean_rest_bout, "
        "mean_move_bout, longest_rest, longest_move, mean_abs_turn, "
        "turns_per_distance, exploration_rate, steering_alignment, "
        "state_dependence, bias_over_weights, energy_efficiency"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"[arena] wrote {args.out}")

    # 6. Write CSV summary.
    if summary_rows:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[arena] wrote {args.csv}")


if __name__ == "__main__":
    main()
