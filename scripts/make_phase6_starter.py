"""Create a Phase 6 starter genome from the Phase 1.5 warm-start parent.

The warm-start genome has 6 sensor nodes (3 food_a + 3 food_b) and
no marker sensor. For Phase 6 we need 3 additional sensor nodes for
the marker field probes (left, front, right). We add them as new
HIDDEN-ish sensor slots and wire them into the existing hidden node
with small initial weights, so the brain can immediately use them.

Output: evolife_regen_warm_parent_phase6.json
"""
from __future__ import annotations

import json
import sys

from evolife.genome import (
    Activation,
    ConnectionGene,
    Genome,
    NodeGene,
    NodeType,
)


def make_phase6_starter(warm_path: str, out_path: str) -> None:
    with open(warm_path) as fh:
        parent = Genome.from_dict(json.load(fh))

    # Find existing nodes.
    sensor_ids = [n.id for n in parent.nodes.values() if n.type is NodeType.SENSOR]
    motor_ids = [n.id for n in parent.nodes.values() if n.type is NodeType.MOTOR]
    hidden_ids = [n.id for n in parent.nodes.values() if n.type is NodeType.HIDDEN]

    if len(sensor_ids) != 6:
        sys.exit(
            f"Expected 6 sensor nodes in warm-start, found {len(sensor_ids)}. "
            f"warm-start genome may not be Phase 1.5 parent."
        )

    # Add 3 new sensor nodes for marker probes (left, front, right).
    next_node_id = max(parent.nodes.keys()) + 1
    marker_sensors = []
    for _ in range(3):
        nid = next_node_id
        next_node_id += 1
        parent.nodes[nid] = NodeGene(
            id=nid, type=NodeType.SENSOR, activation=Activation.TANH
        )
        marker_sensors.append(nid)

    # Wire each marker sensor to the (single) hidden node, if any, with
    # small initial weight so evolution can amplify it. If no hidden node
    # exists, wire directly to motors instead.
    next_innov = max(parent.connections.keys()) + 1
    targets = hidden_ids if hidden_ids else motor_ids
    for ms in marker_sensors:
        for tgt in targets:
            parent.connections[next_innov] = ConnectionGene(
                innovation=next_innov,
                in_node=ms,
                out_node=tgt,
                weight=0.1,
                enabled=True,
            )
            next_innov += 1

    with open(out_path, "w") as fh:
        json.dump(parent.to_dict(), fh, indent=2)
    print(
        f"Wrote {out_path}: nodes={len(parent.nodes)} "
        f"(sensor={len(sensor_ids) + 3} incl 3 marker) "
        f"conns={len(parent.connections)}"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--warm",
        default="evolife_regen_warm_parent.json",
        help="Path to warm-start genome JSON.",
    )
    p.add_argument(
        "--out",
        default="evolife_regen_warm_parent_phase6.json",
        help="Path to write Phase 6 starter genome.",
    )
    args = p.parse_args()
    make_phase6_starter(args.warm, args.out)
