"""Aggregate Phase 1 archive milestones from per-seed SQLite.

Reports first_hidden_node / first_recurrent_cycle / max_brain_nodes
counts per seed so we can answer whether structural evolution reaches
hidden<->hidden cycles within 50k ticks under the bumped rates.
"""
import sqlite3
import json
from collections import Counter

seeds = (1, 2, 3, 4, 5)
print("Phase 1 archive milestones by seed")
print("=" * 70)
total_first_hidden = 0
total_first_recurrent = 0
for seed in seeds:
    path = f"evolife_metrics_seed{seed}.sqlite"
    try:
        c = sqlite3.connect(path)
    except sqlite3.OperationalError:
        print(f"seed {seed}: missing sqlite ({path})")
        continue
    counts = Counter()
    first_tick_by_kind = {}
    for kind, tick in c.execute(
        "SELECT kind, MIN(tick) FROM archive_milestones GROUP BY kind"
    ).fetchall():
        first_tick_by_kind[kind] = tick
    for kind, n in c.execute(
        "SELECT kind, COUNT(*) FROM archive_milestones GROUP BY kind"
    ).fetchall():
        counts[kind] = n
    print(f"seed {seed}:")
    for k, n in sorted(counts.items()):
        first = first_tick_by_kind.get(k, "?")
        print(f"  {k:30s} n={n:4d}  first_tick={first}")
    total_first_hidden += counts.get("FIRST_HIDDEN_NODE", 0)
    total_first_recurrent += counts.get("FIRST_RECURRENT_CYCLE", 0)
    c.close()
print("=" * 70)
print(f"TOTAL across seeds {seeds}:")
print(f"  first_hidden_node events:    {total_first_hidden}")
print(f"  first_recurrent_cycle events: {total_first_recurrent}")
print()
if total_first_recurrent == 0:
    print("VERDICT: No hidden<->hidden cycle emerged in any of the")
    print(f"{len(seeds)} seeds within 50k ticks under the bumped rates.")
    print("Phase 1 lifted rates 5x but a recurrent cycle between two")
    print("hidden nodes did not appear in this sample.")
else:
    print(f"VERDICT: Recurrent cycle DID emerge in {total_first_recurrent}")
    print("of the seeds. Hidden<->hidden wiring appeared within 50k ticks.")
