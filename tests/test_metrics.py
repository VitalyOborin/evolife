from evolife.metrics import Metrics
from evolife.world import World


def test_metrics_writes_species_and_behavior_tables(tmp_path):
    w = World(seed=1)
    w.step()
    w.tick = 100
    path = tmp_path / "m.sqlite"
    m = Metrics(path=str(path))
    try:
        m.record_species(w)
        m.record_behavior(w, w.organisms)
        m.flush_species_events(w.species_manager)
        n_species = m._conn.execute(
            "SELECT COUNT(*) FROM species_snapshots"
        ).fetchone()[0]
        n_behavior = m._conn.execute(
            "SELECT COUNT(*) FROM behavior_snapshots"
        ).fetchone()[0]
        n_events = m._conn.execute(
            "SELECT COUNT(*) FROM species_events WHERE kind = 'ORIGIN'"
        ).fetchone()[0]
    finally:
        m.close()
    assert n_species >= 1
    assert n_behavior == len(w.organisms)
    assert n_events >= 1
