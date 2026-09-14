import sqlite3
c = sqlite3.connect('evolife_metrics_gpu_smoke.sqlite')
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('tables:', tables)
for t in ('world_snapshots', 'organism_snapshots', 'species_snapshots', 'events'):
    if t in tables:
        n = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        print(f'  {t}: {n} rows')
    else:
        print(f'  {t}: MISSING')
print('--- world_snapshots sample ---')
for row in c.execute('SELECT * FROM world_snapshots ORDER BY tick LIMIT 3').fetchall():
    print(' ', row)
print('--- organism_snapshots sample ---')
for row in c.execute('SELECT tick, organism_id, parent_id, age, energy, generation, food_eaten, alive FROM organism_snapshots ORDER BY tick LIMIT 3').fetchall():
    print(' ', row)
print('--- species_snapshots sample ---')
for row in c.execute('SELECT * FROM species_snapshots LIMIT 3').fetchall():
    print(' ', row)
