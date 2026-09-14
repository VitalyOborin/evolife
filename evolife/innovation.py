"""Innovation numbers (NEAT).

Connections are identified by (in_node, out_node). Node IDs for hidden
neurons created by add_node are allocated globally from the historical
connection that was split, so two lineages that independently split the
same ancestral gene receive the same new node ID. That is what makes
compatibility distance meaningful.

The InnovationDatabase is shared by all organisms in a World.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SplitRecord:
    new_node_id: int
    in_innovation: int
    out_innovation: int


@dataclass
class InnovationDatabase:
    """Global connection and split-node registry."""

    _next_conn: int = 0
    _next_node: int = 0
    _by_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    _splits: dict[int, SplitRecord] = field(default_factory=dict)

    def innovation_for(self, in_node: int, out_node: int) -> int:
        """Return the innovation number for in_node -> out_node."""
        self._touch_nodes(in_node, out_node)
        key = (in_node, out_node)
        existing = self._by_pair.get(key)
        if existing is not None:
            return existing
        fresh = self._next_conn
        self._next_conn += 1
        self._by_pair[key] = fresh
        return fresh

    def split_for(
        self, original_innovation: int, in_node: int, out_node: int
    ) -> SplitRecord:
        """Allocate (or reuse) the node+edges produced by splitting a gene.

        Two independent splits of the same historical connection share
        new_node_id and the two new connection innovations.
        """
        existing = self._splits.get(original_innovation)
        if existing is not None:
            self._touch_nodes(existing.new_node_id)
            return existing
        self._touch_nodes(in_node, out_node)
        new_node_id = self._next_node
        self._next_node += 1
        in_innov = self.innovation_for(in_node, new_node_id)
        out_innov = self.innovation_for(new_node_id, out_node)
        rec = SplitRecord(
            new_node_id=new_node_id,
            in_innovation=in_innov,
            out_innovation=out_innov,
        )
        self._splits[original_innovation] = rec
        return rec

    def _touch_nodes(self, *node_ids: int) -> None:
        for nid in node_ids:
            if nid + 1 > self._next_node:
                self._next_node = nid + 1

    def reset(self) -> None:
        self._next_conn = 0
        self._next_node = 0
        self._by_pair.clear()
        self._splits.clear()
