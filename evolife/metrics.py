"""Metrics — per-organism and per-world snapshots into SQLite.

Even though we explicitly avoid computing a fitness function, we record
metrics from tick 1. Without them we cannot later answer the v0 question:
"is complexity actually growing?"
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

from .behavior import descriptors
from .config import (
    METRICS_BEHAVIOR_EVERY,
    METRICS_DB_PATH,
    METRICS_ORGANISM_EVERY,
    METRICS_SPECIES_EVERY,
    METRICS_WORLD_EVERY,
)
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

CREATE TABLE IF NOT EXISTS species_snapshots (
    tick                       INTEGER NOT NULL,
    species_id                 INTEGER NOT NULL,
    member_count               INTEGER NOT NULL,
    peak_population            INTEGER NOT NULL,
    born_tick                  INTEGER NOT NULL,
    last_seen_tick             INTEGER NOT NULL,
    parent_species_id          INTEGER,
    extinct                    INTEGER NOT NULL,
    established                INTEGER NOT NULL,
    representative_genome_hash TEXT,
    PRIMARY KEY (tick, species_id)
);

CREATE TABLE IF NOT EXISTS behavior_snapshots (
    tick                     INTEGER NOT NULL,
    organism_id              INTEGER NOT NULL,
    species_id               INTEGER NOT NULL,
    moving_fraction          REAL    NOT NULL,
    transition_rate          REAL    NOT NULL,
    mean_speed               REAL    NOT NULL,
    mean_speed_while_moving  REAL    NOT NULL,
    mean_rest_bout           REAL    NOT NULL,
    mean_move_bout           REAL    NOT NULL,
    longest_rest             REAL    NOT NULL,
    longest_move             REAL    NOT NULL,
    food_rate                REAL    NOT NULL,
    distance_per_food        REAL    NOT NULL,
    time_to_first_food       REAL    NOT NULL,
    mean_abs_turn            REAL    NOT NULL,
    turns_per_distance       REAL    NOT NULL,
    exploration_rate         REAL    NOT NULL,
    steering_alignment       REAL    NOT NULL,
    state_dependence         REAL    NOT NULL,
    PRIMARY KEY (tick, organism_id)
);

CREATE TABLE IF NOT EXISTS species_events (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    tick                         INTEGER NOT NULL,
    species_id                   INTEGER NOT NULL,
    kind                         TEXT    NOT NULL,
    parent_species_id            INTEGER,
    founder_organism_id          INTEGER,
    representative_genome_hash   TEXT
);
CREATE INDEX IF NOT EXISTS species_events_tick_idx ON species_events(tick);

CREATE TABLE IF NOT EXISTS archive_milestones (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tick         INTEGER NOT NULL,
    kind         TEXT    NOT NULL,
    payload_json  TEXT
);
CREATE INDEX IF NOT EXISTS archive_milestones_kind_idx ON archive_milestones(kind);
CREATE INDEX IF NOT EXISTS archive_milestones_tick_idx ON archive_milestones(tick);

CREATE TABLE IF NOT EXISTS cycle_carriers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    tick                INTEGER NOT NULL,
    child_id            INTEGER NOT NULL,
    parent_id           INTEGER NOT NULL,
    parent_genome_json  TEXT    NOT NULL,
    cycle_genome_json   TEXT    NOT NULL,
    parent_hash         TEXT    NOT NULL,
    cycle_hash          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS cycle_carriers_tick_idx ON cycle_carriers(tick);
"""


class Metrics:
    """SQLite-backed metrics writer. Safe to call from the main thread."""

    def __init__(self, path: str = METRICS_DB_PATH) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._migrate_organism_snapshots()
        self._migrate_species_snapshots()
        self._species_event_offset = 0
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

    def _migrate_species_snapshots(self) -> None:
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(species_snapshots)")
        }
        if "established" not in cols:
            self._conn.execute(
                "ALTER TABLE species_snapshots ADD COLUMN established "
                "INTEGER NOT NULL DEFAULT 0"
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

    def record_species(self, world: World) -> None:
        if world.tick % METRICS_SPECIES_EVERY != 0:
            return
        rows = []
        for sp in world.species_manager.species.values():
            rows.append(
                (
                    world.tick,
                    sp.id,
                    sp.member_count,
                    sp.peak_population,
                    sp.born_tick,
                    sp.last_seen_tick,
                    sp.parent_species_id,
                    int(sp.extinct),
                    int(sp.established),
                    sp.representative.fingerprint(),
                )
            )
        if not rows:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO species_snapshots ("
            "tick, species_id, member_count, peak_population, born_tick, "
            "last_seen_tick, parent_species_id, extinct, established, "
            "representative_genome_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def record_behavior(
        self, world: World, organisms: Iterable[Organism]
    ) -> None:
        if world.tick % METRICS_BEHAVIOR_EVERY != 0:
            return
        rows = []
        for o in organisms:
            d = descriptors(o)
            rows.append(
                (
                    world.tick,
                    o.id,
                    o.species_id,
                    d["moving_fraction"],
                    d["transition_rate"],
                    d["mean_speed"],
                    d["mean_speed_while_moving"],
                    d["mean_rest_bout"],
                    d["mean_move_bout"],
                    d["longest_rest"],
                    d["longest_move"],
                    d["food_rate"],
                    d["distance_per_food"],
                    d["time_to_first_food"],
                    d["mean_abs_turn"],
                    d["turns_per_distance"],
                    d["exploration_rate"],
                    d["steering_alignment"],
                    d["state_dependence"],
                )
            )
        if not rows:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO behavior_snapshots ("
            "tick, organism_id, species_id, moving_fraction, transition_rate, "
            "mean_speed, mean_speed_while_moving, mean_rest_bout, "
            "mean_move_bout, longest_rest, longest_move, food_rate, "
            "distance_per_food, time_to_first_food, mean_abs_turn, "
            "turns_per_distance, exploration_rate, steering_alignment, "
            "state_dependence"
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

    def flush_species_events(self, manager) -> None:
        """Persist SpeciesManager origin/extinction events."""
        events = manager.events[self._species_event_offset :]
        if not events:
            return
        rows = [
            (
                e.tick,
                e.species_id,
                e.kind,
                e.parent_species_id,
                e.founder_organism_id,
                e.representative_genome_hash,
            )
            for e in events
        ]
        self._conn.executemany(
            "INSERT INTO species_events ("
            "tick, species_id, kind, parent_species_id, "
            "founder_organism_id, representative_genome_hash"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._species_event_offset = len(manager.events)

    def record_archive_milestones(self, archive) -> int:
        """Persist any new Archive milestones to the archive_milestones
        table. Returns the number of rows written on this call.

        Each call records only milestones after the high-water mark so
        callers can invoke it every N ticks without re-inserting the
        same milestones.
        """
        import json as _json

        if not hasattr(self, "_archive_milestone_offset"):
            self._archive_milestone_offset = 0
        new = archive.milestones[self._archive_milestone_offset :]
        if not new:
            return 0
        rows = []
        for m in new:
            rows.append(
                (m.tick, m.kind.name, _json.dumps(m.payload, default=str))
            )
        self._conn.executemany(
            "INSERT INTO archive_milestones (tick, kind, payload_json) "
            "VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._archive_milestone_offset = len(archive.milestones)
        return len(rows)

    def record_cycle_carriers(self, archive) -> int:
        """Persist any new CycleCarrier entries (hidden<->hidden cycle
        born in a child) to the cycle_carriers table. Each entry
        contains the parent genome and the cycle genome as JSON so the
        Behavioral Arena can replay the pair without re-running
        evolution. Returns the number of rows written on this call.
        """
        import json as _json

        if not hasattr(self, "_cycle_carrier_offset"):
            self._cycle_carrier_offset = 0
        new = archive.cycle_carriers[self._cycle_carrier_offset :]
        if not new:
            return 0
        rows = []
        for cc in new:
            rows.append(
                (
                    cc.tick,
                    cc.child_id,
                    cc.parent_id,
                    _json.dumps(cc.parent_genome.to_dict()),
                    _json.dumps(cc.cycle_genome.to_dict()),
                    cc.parent_genome.fingerprint(),
                    cc.cycle_genome.fingerprint(),
                )
            )
        self._conn.executemany(
            "INSERT INTO cycle_carriers ("
            "tick, child_id, parent_id, parent_genome_json, "
            "cycle_genome_json, parent_hash, cycle_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._cycle_carrier_offset = len(archive.cycle_carriers)
        return len(rows)
        return len(rows)


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
