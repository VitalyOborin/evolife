import sqlite3
for seed in (1, 2, 3):
    c = sqlite3.connect(f"evolife_metrics_seed{seed}.sqlite")
    rows = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [r[0] for r in rows]
    print(f"seed {seed} tables:", tables)
