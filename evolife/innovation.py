"""Innovation numbers (NEAT).

When two organisms independently evolve the same structural mutation —
the same connection between the same two node ids — they should receive
the SAME innovation number. This is what lets us later align genomes for
crossover and measure compatibility distance for speciation.

In v1 we don't crossover and don't speciate, but we wire this in now so
v2 and v3 are config flips, not rewrites.

The InnovationDatabase is shared by ALL organisms in the world. In a
multi-process setting this would have to be process-shared (Redis, file,
etc.). For v1 it's a single object held by World.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InnovationDatabase:
    """Maps (in_node, out_node) -> innovation_number.

    Both structural mutations (add_node, add_connection) consult this
    database. Weight mutations do not.
    """

    _next: int = 0
    _by_pair: dict[tuple[int, int], int] = field(default_factory=dict)

    def innovation_for(self, in_node: int, out_node: int) -> int:
        """Return the innovation number for a connection in_node -> out_node.

        Creates a new entry if this pair has never been seen before.
        """
        key = (in_node, out_node)
        existing = self._by_pair.get(key)
        if existing is not None:
            return existing
        fresh = self._next
        self._next += 1
        self._by_pair[key] = fresh
        return fresh

    def reset(self) -> None:
        self._next = 0
        self._by_pair.clear()
