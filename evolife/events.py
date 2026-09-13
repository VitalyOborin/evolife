"""Event log — append-only Birth / Death / Reproduction / Eat events.

In v0/v1 we only had periodic snapshots, which means short-lived
organisms could disappear from the lineage entirely. v2 records every
meaningful event so the evolutionary tree can be reconstructed from
the log alone.

The log is held in memory and flushed to SQLite by Metrics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    BIRTH = "birth"
    DEATH = "death"
    REPRODUCTION = "reproduction"
    EAT = "eat"


@dataclass
class Event:
    kind: EventKind
    tick: int
    fields: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """In-memory append-only event log.

    For long runs the in-memory list can grow large; flushing to SQLite
    is left to Metrics.record_events. We keep the simple list here so
    tests can introspect without touching the DB.
    """

    def __init__(self) -> None:
        self.events: list[Event] = []

    def record_birth(
        self,
        tick: int,
        *,
        org_id: int,
        parent_id: int | None,
        genome_hash: str,
    ) -> None:
        self.events.append(
            Event(
                kind=EventKind.BIRTH,
                tick=tick,
                fields={
                    "org_id": org_id,
                    "parent_id": parent_id,
                    "genome_hash": genome_hash,
                },
            )
        )

    def record_death(
        self, tick: int, *, org_id: int, cause: str
    ) -> None:
        self.events.append(
            Event(
                kind=EventKind.DEATH,
                tick=tick,
                fields={"org_id": org_id, "cause": cause},
            )
        )

    def record_reproduction(
        self,
        tick: int,
        *,
        parent_id: int,
        child_id: int,
        child_genome_hash: str,
    ) -> None:
        self.events.append(
            Event(
                kind=EventKind.REPRODUCTION,
                tick=tick,
                fields={
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "child_genome_hash": child_genome_hash,
                },
            )
        )

    def record_eat(
        self, tick: int, *, org_id: int, x: float, y: float
    ) -> None:
        self.events.append(
            Event(
                kind=EventKind.EAT,
                tick=tick,
                fields={"org_id": org_id, "x": x, "y": y},
            )
        )

    def by_kind(self) -> dict[EventKind, list[Event]]:
        out: dict[EventKind, list[Event]] = defaultdict(list)
        for e in self.events:
            out[e.kind].append(e)
        return out
