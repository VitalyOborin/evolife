"""NEAT-shaped genome.

Even though v0 only mutates weights, the genome is structured as if it
were NEAT from day one: nodes are keyed by integer ids, connections carry
innovation numbers, and all four mutations are defined. This way v1
(structural mutations enabled) is a config flip, not a rewrite.

Concepts (kept minimal, no species/innovation DB yet — those come when
speciation turns on in v1):

- NodeGene: id, type (sensor / hidden / motor), activation, bias.
- ConnectionGene: innovation_number, in_node, out_node, weight, enabled.
- Genome: list of NodeGenes + dict of ConnectionGenes keyed by innovation.

The genome is the **source of truth for topology and weights**. The Brain
is just a thin executable view over a frozen genome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class NodeType(str, Enum):
    SENSOR = "sensor"
    HIDDEN = "hidden"
    MOTOR = "motor"


class Activation(str, Enum):
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    RELU = "relu"


@dataclass
class NodeGene:
    id: int
    type: NodeType
    activation: Activation = Activation.TANH
    bias: float = 0.0


@dataclass
class ConnectionGene:
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    """A NEAT-shaped genome. Pure data, no behaviour."""

    nodes: dict[int, NodeGene] = field(default_factory=dict)
    connections: dict[int, ConnectionGene] = field(default_factory=dict)
    # Counter local to this genome: highest innovation it has seen. Not the
    # global innovation database — that lives elsewhere and is only needed
    # when structural mutations are enabled.
    max_innovation: int = 0

    def sensors(self) -> list[NodeGene]:
        return [n for n in self.nodes.values() if n.type is NodeType.SENSOR]

    def motors(self) -> list[NodeGene]:
        return [n for n in self.nodes.values() if n.type is NodeType.MOTOR]

    def active_connections(self) -> Iterable[ConnectionGene]:
        return (c for c in self.connections.values() if c.enabled)

    def fingerprint(self) -> str:
        """A stable hash of topology + weights for event log dedup."""
        import hashlib

        node_part = ",".join(
            f"{n.id}:{n.type.value}:{n.activation.value}:{n.bias:.6f}"
            for n in sorted(self.nodes.values(), key=lambda x: x.id)
        )
        conn_part = ",".join(
            f"{c.innovation}:{c.in_node}>{c.out_node}:{c.weight:.6f}:{int(c.enabled)}"
            for c in sorted(self.connections.values(), key=lambda x: x.innovation)
        )
        h = hashlib.sha1(f"{node_part}|{conn_part}".encode()).hexdigest()
        return h[:16]

    def to_dict(self) -> dict:
        """Plain-dict JSON-friendly representation.

        Used by the archive to persist carrier genomes for later
        arena benchmarking. Enum fields are stored as their string
        value so the JSON is human-readable.
        """
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "activation": n.activation.value,
                    "bias": n.bias,
                }
                for n in sorted(self.nodes.values(), key=lambda x: x.id)
            ],
            "connections": [
                {
                    "innovation": c.innovation,
                    "in_node": c.in_node,
                    "out_node": c.out_node,
                    "weight": c.weight,
                    "enabled": c.enabled,
                }
                for c in sorted(self.connections.values(), key=lambda x: x.innovation)
            ],
            "max_innovation": self.max_innovation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        """Inverse of `to_dict`. Builds a Genome from a JSON-ready dict."""
        g = cls()
        for nd in data.get("nodes", []):
            g.nodes[nd["id"]] = NodeGene(
                id=nd["id"],
                type=NodeType(nd["type"]),
                activation=Activation(nd["activation"]),
                bias=nd["bias"],
            )
        for cd in data.get("connections", []):
            g.connections[cd["innovation"]] = ConnectionGene(
                innovation=cd["innovation"],
                in_node=cd["in_node"],
                out_node=cd["out_node"],
                weight=cd["weight"],
                enabled=cd["enabled"],
            )
        g.max_innovation = data.get("max_innovation", 0)
        return g
