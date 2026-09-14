"""Phase 2.5 — Behavioral Arena on cycle_carriers from prior evolution.

Reads hidden<->hidden cycle carriers + their parent genomes (which are
guaranteed cycle-free by construction) from the cycle_carriers table of
a Phase 1.5-style run, then runs both genomes on every Arena scenario
for N_SEEDS episodes each. Writes per-(carrier, scenario, side)
metrics to a fresh SQLite plus a CSV summary keyed by carrier_id.

This is the qualitative payoff of Phase 1.5: did evolution's hidden
cycle do anything observable behaviourally?

Usage:
  python scripts/run_cycle_arena.py \\
      --sqlite evolife_metrics_seed4.sqlite \\
      --arena-ticks 200 --arena-seeds 10 \\
      --out evolife_cycle_arena.sqlite
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from evolife.arena import (
    ARENA_N_SEEDS_DEFAULT,
    ARENA_N_TICKS_DEFAULT,
    SCENARIO_BUILDERS,
    aggregate,
    brain_shape,
    run_episode,
)
from evolife.genome import Genome


_SCENARIOS_DEFAULT = "uniform,ahead,behind,left_right,sparse,dense," \
                     "relocating,smell_blanked"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycle_arena_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    carrier_id       INTEGER NOT NULL,
    side             TEXT    NOT NULL,    -- 'parent' or 'cycle'
    scenario         TEXT    NOT NULL,
    seed             INTEGER NOT NULL,
    brain_nodes      INTEGER NOT NULL,
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
CREATE INDEX IF NOT EXISTS cycar_idx ON cycle_arena_episodes(carrier_id);
CREATE INDEX IF NOT EXISTS cycar_side_idx ON cycle_arena_episodes(side);
"""


def _load_carriers(sqlite_path: Path) -> list[dict]:
    """Load all cycle_carriers from the SQLite, deserialise genomes."""
    c = sqlite3.connect(str(sqlite_path))
    rows = list(c.execute(
        "SELECT id, tick, child_id, parent_id, parent_genome_json, "
        "cycle_genome_json FROM cycle_carriers ORDER BY tick"
    ))
    out = []
    for cid, tick, child_id, parent_id, pj, cj in rows:
        out.append({
            "id": cid,
            "tick": tick,
            "child_id": child_id,
            "parent_id": parent_id,
            "parent_genome": Genome.from_dict(json.loads(pj)),
            "cycle_genome": Genome.from_dict(json.loads(cj)),
        })
    c.close()
    return out


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, required=True,
                   help="Per-seed Phase 1.5 SQLite with cycle_carriers")
    p.add_argument("--arena-ticks", type=int, default=ARENA_N_TICKS_DEFAULT)
    p.add_argument("--arena-seeds", type=int, default=ARENA_N_SEEDS_DEFAULT)
    p.add_argument("--scenarios", type=str, default=_SCENARIOS_DEFAULT)
    p.add_argument("--out", type=Path, default=Path("evolife_cycle_arena.sqlite"))
    p.add_argument("--csv", type=Path, default=Path("evolife_cycle_arena.csv"))
    args = p.parse_args()

    carriers = _load_carriers(args.sqlite)
    if not carriers:
        print(f"[cycle_arena] no cycle_carriers in {args.sqlite}, exiting")
        return
    print(f"[cycle_arena] loaded {len(carriers)} carriers from {args.sqlite}")
    for cc in carriers:
        pn, pc = brain_shape(cc["parent_genome"])
        cn, cc2 = brain_shape(cc["cycle_genome"])
        print(f"  carrier id={cc['id']} tick={cc['tick']} "
              f"parent(brain={pn}/{pc}) cycle(brain={cn}/{cc2})")

    scenarios = parse_str_list(args.scenarios)
    for s in scenarios:
        if s not in SCENARIO_BUILDERS:
            raise SystemExit(f"unknown scenario: {s}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.out))
    conn.executescript(_SCHEMA)

    rows = []
    summary_rows = []
    total = len(carriers) * len(scenarios) * 2 * args.arena_seeds
    done = 0
    for cc in carriers:
        for sc in scenarios:
            per_side: dict[str, list] = {"parent": [], "cycle": []}
            for side in ("parent", "cycle"):
                genome = cc["parent_genome"] if side == "parent" else cc["cycle_genome"]
                nodes, conns = brain_shape(genome)
                for s in range(args.arena_seeds):
                    res = run_episode(genome, sc, seed=s, n_ticks=args.arena_ticks)
                    per_side[side].append(res)
                    rows.append((
                        cc["id"], side, sc, s, nodes, conns,
                        res.food_eaten, res.ticks_to_first_food, res.ticks_alive,
                        res.final_energy, res.distance_traveled, res.mean_speed,
                        res.mean_speed_while_moving, res.moving_fraction,
                        res.movement_transitions, res.mean_rest_bout,
                        res.mean_move_bout, res.longest_rest, res.longest_move,
                        res.mean_abs_turn, res.turns_per_distance,
                        res.exploration_rate, res.steering_alignment,
                        res.state_dependence, res.bias_over_weights,
                        res.energy_efficiency,
                    ))
                    done += 1
            for side in ("parent", "cycle"):
                agg = aggregate(per_side[side])
                summary_rows.append({
                    "carrier_id": cc["id"],
                    "side": side,
                    "scenario": sc,
                    "n_seeds": args.arena_seeds,
                    **agg,
                })
            p_agg = aggregate(per_side["parent"])
            c_agg = aggregate(per_side["cycle"])
            print(
                f"  carrier={cc['id']:>2}  sc={sc:<14}  "
                f"P food={p_agg['food_eaten_mean']:>5.2f} "
                f"steer={p_agg['steering_alignment_mean']:>+6.3f} "
                f"sd={p_agg['state_dependence_mean']:>5.3f}  |  "
                f"C food={c_agg['food_eaten_mean']:>5.2f} "
                f"steer={c_agg['steering_alignment_mean']:>+6.3f} "
                f"sd={c_agg['state_dependence_mean']:>5.3f}  "
                f"({done}/{total})",
                flush=True,
            )

    conn.executemany(
        "INSERT INTO cycle_arena_episodes ("
        "carrier_id, side, scenario, seed, brain_nodes, brain_connections, "
        "food_eaten, ticks_to_first_food, ticks_alive, final_energy, "
        "distance_traveled, mean_speed, mean_speed_while_moving, "
        "moving_fraction, movement_transitions, mean_rest_bout, "
        "mean_move_bout, longest_rest, longest_move, mean_abs_turn, "
        "turns_per_distance, exploration_rate, steering_alignment, "
        "state_dependence, bias_over_weights, energy_efficiency"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"[cycle_arena] wrote {args.out}")

    if summary_rows:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[cycle_arena] wrote {args.csv}")


if __name__ == "__main__":
    main()
