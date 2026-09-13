"""Mutation operators.

All four mutations are defined here. v1 enables:
- mutate_weights (always)
- mutate_add_connection (rate from config)
- mutate_add_node (rate from config)
- mutate_toggle_connection (rate from config)

Structural mutations consult a shared InnovationDatabase so that two
organisms which independently evolve the same (in_node, out_node)
connection receive the SAME innovation number. This is what enables
crossover and compatibility-distance speciation in v2/v3.

Each operator takes a genome and returns a NEW genome. Genomes are
treated as immutable values; mutation produces a copy.
"""

from __future__ import annotations

import copy

import numpy as np

from .config import (
    ADD_CONNECTION_RATE,
    ADD_NODE_RATE,
    TOGGLE_CONNECTION_RATE,
    WEIGHT_MAX,
    WEIGHT_MUTATION_RATE,
    WEIGHT_PERTURB_RATE,
    WEIGHT_PERTURB_SIGMA,
)
from .genome import (
    Activation,
    ConnectionGene,
    Genome,
    NodeGene,
    NodeType,
)
from .innovation import InnovationDatabase


# --- weights ----------------------------------------------------------------


def mutate_weights(
    genome: Genome,
    rng: np.random.Generator,
    rate: float = WEIGHT_MUTATION_RATE,
    perturb_rate: float = WEIGHT_PERTURB_RATE,
    sigma: float = WEIGHT_PERTURB_SIGMA,
) -> Genome:
    """Return a new genome with perturbed weights.

    Each enabled connection is independently perturbed with probability
    `perturb_rate`. With probability `rate` the whole operator is applied
    at all; otherwise the genome is returned unchanged.
    """
    if rng.random() >= rate:
        return genome

    new = copy.deepcopy(genome)
    for conn in new.connections.values():
        if not conn.enabled:
            continue
        if rng.random() < perturb_rate:
            conn.weight = float(
                np.clip(
                    conn.weight + rng.normal(0.0, sigma),
                    -WEIGHT_MAX,
                    WEIGHT_MAX,
                )
            )
    return new


# --- structural -------------------------------------------------------------


def mutate_add_connection(
    genome: Genome,
    rng: np.random.Generator,
    innovations: InnovationDatabase,
    rate: float = ADD_CONNECTION_RATE,
) -> Genome:
    """Add a new connection between two previously unconnected nodes.

    We try up to N_ATTEMPTS random (in_node, out_node) pairs. We accept
    the first pair that:
    - both nodes exist in the genome,
    - there is no existing enabled connection between them,
    - the pair is not (sensor, sensor) or (motor, motor) — sensors are
      inputs, motors are outputs, so we forbid those intra-layer pairs.
    The new connection starts with a small random weight in [-1, 1].

    If no valid pair is found in N_ATTEMPTS, the genome is returned
    unchanged.
    """
    if rng.random() >= rate:
        return genome

    new = copy.deepcopy(genome)
    existing = {
        (c.in_node, c.out_node)
        for c in new.connections.values()
        if c.enabled
    }

    nodes = list(new.nodes.values())
    if len(nodes) < 2:
        return new

    n_attempts = 20
    for _ in range(n_attempts):
        a = nodes[int(rng.integers(0, len(nodes)))]
        b = nodes[int(rng.integers(0, len(nodes)))]
        if a.id == b.id:
            continue
        if a.type is NodeType.SENSOR and b.type is NodeType.SENSOR:
            continue
        if a.type is NodeType.MOTOR and b.type is NodeType.MOTOR:
            continue
        if (a.id, b.id) in existing:
            continue
        innov = innovations.innovation_for(a.id, b.id)
        new.connections[innov] = ConnectionGene(
            innovation=innov,
            in_node=a.id,
            out_node=b.id,
            weight=float(rng.uniform(-1.0, 1.0)),
            enabled=True,
        )
        new.max_innovation = max(new.max_innovation, innov)
        return new

    return new


def mutate_add_node(
    genome: Genome,
    rng: np.random.Generator,
    innovations: InnovationDatabase,
    rate: float = ADD_NODE_RATE,
) -> Genome:
    """Split a random enabled connection with a new hidden node.

    The original connection is disabled (not deleted — that keeps the
    innovation lineage). Two new connections are added: in -> new_node
    with weight 1.0, and new_node -> out with the old connection's
    weight. The new node is a tanh hidden node with bias 0.
    """
    if rng.random() >= rate:
        return genome

    enabled = [c for c in genome.connections.values() if c.enabled]
    if not enabled:
        return genome

    target = enabled[int(rng.integers(0, len(enabled)))]
    new = copy.deepcopy(genome)

    # Disable the old connection.
    for c in new.connections.values():
        if c.innovation == target.innovation:
            c.enabled = False
            break

    # Allocate a new hidden node id.
    new_node_id = max(new.nodes.keys()) + 1 if new.nodes else 0
    new.nodes[new_node_id] = NodeGene(
        id=new_node_id,
        type=NodeType.HIDDEN,
        activation=Activation.TANH,
        bias=0.0,
    )

    in_innov = innovations.innovation_for(target.in_node, new_node_id)
    out_innov = innovations.innovation_for(new_node_id, target.out_node)

    new.connections[in_innov] = ConnectionGene(
        innovation=in_innov,
        in_node=target.in_node,
        out_node=new_node_id,
        weight=1.0,
        enabled=True,
    )
    new.connections[out_innov] = ConnectionGene(
        innovation=out_innov,
        in_node=new_node_id,
        out_node=target.out_node,
        weight=target.weight,
        enabled=True,
    )
    new.max_innovation = max(new.max_innovation, in_innov, out_innov)
    return new


def mutate_toggle_connection(
    genome: Genome,
    rng: np.random.Generator,
    rate: float = TOGGLE_CONNECTION_RATE,
) -> Genome:
    """Disable (or re-enable) a random connection."""
    if rng.random() >= rate:
        return genome
    if not genome.connections:
        return genome
    new = copy.deepcopy(genome)
    keys = list(new.connections.keys())
    pick = keys[int(rng.integers(0, len(keys)))]
    new.connections[pick].enabled = not new.connections[pick].enabled
    return new
