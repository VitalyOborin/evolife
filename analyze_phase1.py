"""Quick summary of Phase 1 per-seed SQLite metrics.

Reads world_snapshots table from each of seed 1, 2, 3 sqlite files
written by run_v2_experiment.py and prints first / last row plus a
few checkpoint rows. This is enough to see whether each seed
stayed alive or went extinct.
"""
import sqlite3

for seed in (1, 2, 3):
    path = f"evolife_metrics_seed{seed}.sqlite"
    c = sqlite3.connect(path)
    n = c.execute("SELECT COUNT(*) FROM world_snapshots").fetchone()[0]
    first = c.execute(
        "SELECT tick, population, mean_energy, food_count "
        "FROM world_snapshots ORDER BY tick ASC LIMIT 1"
    ).fetchone()
    last = c.execute(
        "SELECT tick, population, mean_energy, food_count "
        "FROM world_snapshots ORDER BY tick DESC LIMIT 1"
    ).fetchone()
    print(f"seed {seed}: {n} world_snapshots")
    print(f"  first: tick={first[0]} pop={first[1]} meanE={first[2]:.2f} food={first[3]}")
    print(f"  last:  tick={last[0]} pop={last[1]} meanE={last[2]:.2f} food={last[3]}")
    c.close()
