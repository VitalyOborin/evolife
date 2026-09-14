"""Archive — evolutionary milestones for observability, not selection.

The Archive is NOT a Hall of Fame. It does not influence fitness,
reproduction, or survival. It is a passive log of notable events:

  - "first organism with hidden node"
  - "first recurrent cycle"
  - "first lineage to survive N ticks"
  - "largest brain in population"
  - "oldest surviving genome"

This lets us read evolutionary history after the fact: which
structural innovations appeared, when, and in which lineage. We can
also store the actual genomes of milestone organisms so their
fingerprint can be re-checked later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MilestoneKind(str, Enum):
    FIRST_HIDDEN_NODE = "first_hidden_node"
    FIRST_RECURRENT_CYCLE = "first_recurrent_cycle"
    FIRST_LIG_AND_NOG = "first_lineage_sustained_10k"  # placeholder
    MAX_BRAIN_NODES = "max_brain_nodes"
    MAX_LINEAGE_SIZE = "max_lineage_size"
    MAX_GENERATION_REACHED = "max_generation_reached"


@dataclass
class Milestone:
    kind: MilestoneKind
    tick: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleCarrier:
    """A child genome that contains a hidden<->hidden cycle plus its
    parent genome. Stored in the archive so the cycle carrier and its
    parent (which, by construction, has no cycle) can be replayed
    side-by-side in the Behavioral Arena to ask "what does the cycle
    do?".

    The pair is observational only — neither genome is "fitter".
    """

    tick: int
    child_id: int
    parent_id: int
    parent_genome: Any  # evolife.genome.Genome
    cycle_genome: Any   # evolife.genome.Genome


class Archive:
    """Append-only milestone log."""

    def __init__(self) -> None:
        self.milestones: list[Milestone] = []
        # Track which milestones have fired to avoid duplicate entries.
        self._fired: set[MilestoneKind] = set()
        # Numeric maxima: keep only the best value seen.
        self._max_values: dict[MilestoneKind, tuple[int, dict[str, Any]]] = {}
        # Hidden<->hidden cycle carriers with their parent genomes, for
        # later arena comparison. Captured every time a cycle appears
        # in a newborn organism, not just the first time.
        self.cycle_carriers: list[CycleCarrier] = []

    def maybe_fire(
        self,
        kind: MilestoneKind,
        tick: int,
        value: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Fire a milestone if it has not fired before (one-shot).

        For "first_X" milestones we want exactly one entry: the first
        time X happened. Subsequent occurrences are not news.
        """
        if kind in self._fired:
            return
        self._fired.add(kind)
        self.milestones.append(
            Milestone(kind=kind, tick=tick, payload=payload or {"value": value})
        )

    def record_max(
        self,
        kind: MilestoneKind,
        tick: int,
        value: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Update a numeric maximum if `value` exceeds the prior best.

        Each update appends a Milestone with the new value. This is
        intended for milestones like MAX_BRAIN_NODES or
        MAX_GENERATION_REACHED where the running record is itself
        interesting.
        """
        prev = self._max_values.get(kind)
        if prev is None or value > prev[0]:
            self._max_values[kind] = (value, payload or {})
            self.milestones.append(
                Milestone(
                    kind=kind, tick=tick, payload=payload or {"value": value}
                )
            )

    def summary(self) -> dict[str, Any]:
        """Compact snapshot of current archive state for metrics."""
        return {
            "milestone_count": len(self.milestones),
            "firsts": sorted(
                m.kind.value for m in self.milestones
                if m.kind in self._fired
            ),
            "max_values": {
                kind.value: val[0] for kind, val in self._max_values.items()
            },
        }
