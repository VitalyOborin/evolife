"""Metrics — per-organism and per-world snapshots into SQLite.

Even though we explicitly avoid computing a fitness function, we record
metrics from tick 1. Without them we cannot later answer the v0 question:
"is complexity actually growing?"
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

from .config import METRICS_DB_PATH, METRICS_ORGANISM_EVERY, METRICS_WORLD_EVERY
from .organism import Organism
from .world import World


_SCHEMA = """
CREATE TABLE IF NOT EXISTS world_snapshots (
    tick        INTEGER PRIMARY KEY,
    population  INTEGER NOT NULL,
    mean_energy REAL    NOT NULL,
    food_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS organism_snapshots (
    tick         INTEGER NOT NULL,
    organism_id  INTEGER NOT NULL,
    parent_id    INTEGER,
    age          INTEGER NOT NULL,
    energy       REAL    NOT NULL,
    peak_energy  REAL    NOT NULL,
    children     INTEGER NOT NULL,
    genome_nodes INTEGER NOT NULL,
    genome_conns INTEGER NOT NULL,
    alive        INTEGER NOT NULL,
    PRIMARY KEY (tick, organism_id)
);
"""


class Metrics:
    """SQLite-backed metrics writer. Safe to call from the main thread."""

    def __init__(self, path: str = METRICS_DB_PATH) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_world(self, world: World) -> None:
        if world.tick % METRICS_WORLD_EVERY != 0:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO world_snapshots VALUES (?, ?, ?, ?)",
            (
                world.tick,
                world.population(),
                world.mean_energy(),
                len(world.food),
            ),
        )
        self._conn.commit()

    def record_organisms(
        self, world: World, organisms: Iterable[Organism]
    ) -> None:
        if world.tick % METRICS_ORGANISM_EVERY != 0:
            return
        rows = []
        for o in organisms:
            assert o.genome is not None
            rows.append(
                (
                    world.tick,
                    o.id,
                    o.parent_id,
                    o.age,
                    o.energy,
                    o.peak_energy,
                    o.children,
                    len(o.genome.nodes),
                    len(o.genome.connections),
                    int(o.alive),
                )
            )
        self._conn.executemany(
            "INSERT OR REPLACE INTO organism_snapshots VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
