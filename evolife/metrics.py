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
from .genome import Genome


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
    generation   INTEGER NOT NULL,
    founder_lineage_id INTEGER NOT NULL,
    food_eaten   INTEGER NOT NULL,
    movement_transitions INTEGER NOT NULL,
    ticks_moving INTEGER NOT NULL,
    speed_sum    REAL    NOT NULL,
    longest_rest INTEGER NOT NULL,
    longest_move INTEGER NOT NULL,
    bias_over_weights REAL NOT NULL,
    alive        INTEGER NOT NULL,
    PRIMARY KEY (tick, organism_id)
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tick         INTEGER NOT NULL,
    kind         TEXT    NOT NULL,
    org_id       INTEGER,
    parent_id    INTEGER,
    child_id     INTEGER,
    cause        TEXT,
    genome_hash  TEXT,
    x            REAL,
    y            REAL
);
CREATE INDEX IF NOT EXISTS events_tick_idx ON events(tick);
CREATE INDEX IF NOT EXISTS events_kind_idx ON events(kind);
"""


class Metrics:
    """SQLite-backed metrics writer. Safe to call from the main thread."""

    def __init__(self, path: str = METRICS_DB_PATH) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._migrate_organism_snapshots()
        self._conn.commit()

    def _migrate_organism_snapshots(self) -> None:
        """Add v2.2 columns to organism_snapshots if an older DB is reused."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(organism_snapshots)")
        }
        for name, spec in (
            ("generation", "INTEGER NOT NULL DEFAULT 0"),
            ("founder_lineage_id", "INTEGER NOT NULL DEFAULT 0"),
            ("food_eaten", "INTEGER NOT NULL DEFAULT 0"),
            ("movement_transitions", "INTEGER NOT NULL DEFAULT 0"),
            ("ticks_moving", "INTEGER NOT NULL DEFAULT 0"),
            ("speed_sum", "REAL NOT NULL DEFAULT 0"),
            ("longest_rest", "INTEGER NOT NULL DEFAULT 0"),
            ("longest_move", "INTEGER NOT NULL DEFAULT 0"),
            ("bias_over_weights", "REAL NOT NULL DEFAULT 0"),
        ):
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE organism_snapshots ADD COLUMN {name} {spec}"
                )

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
                    o.generation,
                    o.founder_lineage_id,
                    o.food_eaten,
                    o.movement_transitions,
                    o.ticks_moving,
                    o.speed_sum,
                    o.longest_rest,
                    o.longest_move,
                    _locomotion_bias_over_weights(o.genome),
                    int(o.alive),
                )
            )
        self._conn.executemany(
            "INSERT OR REPLACE INTO organism_snapshots ("
            "tick, organism_id, parent_id, age, energy, peak_energy, "
            "children, genome_nodes, genome_conns, generation, "
            "founder_lineage_id, food_eaten, movement_transitions, "
            "ticks_moving, speed_sum, longest_rest, longest_move, "
            "bias_over_weights, alive"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def flush_events(self, events) -> None:
        """Persist events from an EventLog to the events table."""
        rows = []
        for e in events.events:
            f = e.fields
            rows.append(
                (
                    e.tick,
                    e.kind.value,
                    f.get("org_id"),
                    f.get("parent_id"),
                    f.get("child_id"),
                    f.get("cause"),
                    f.get("genome_hash") or f.get("child_genome_hash"),
                    f.get("x"),
                    f.get("y"),
                )
            )
        if not rows:
            return
        self._conn.executemany(
            "INSERT INTO events (tick, kind, org_id, parent_id, child_id, "
            "cause, genome_hash, x, y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()


def _locomotion_bias_over_weights(genome: Genome) -> float:
    """|locomotion_bias| / sum(|weights into locomotion motor|).

    Values >> 1 mean the gait is a locked genetic type; values ~1 mean
    smell can still flip rest ↔ move.
    """
    motors = genome.motors()
    if len(motors) < 2:
        return 0.0
    loc = motors[1]
    wsum = sum(
        abs(c.weight)
        for c in genome.active_connections()
        if c.out_node == loc.id
    )
    if wsum < 1e-9:
        return abs(loc.bias) * 1e9
    return abs(loc.bias) / wsum
