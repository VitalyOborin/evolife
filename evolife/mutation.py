"""Mutation operators.

All four mutations are defined here, even though v0 only uses
`mutate_weights`. v1 will flip rates in config.py and the rest light up
without code changes.

Each operator takes a genome and returns a NEW genome. Genomes are
treated as immutable values; mutation produces a copy.
"""

from __future__ import annotations

import copy

import numpy as np

from .config import (
    WEIGHT_MAX,
    WEIGHT_MUTATION_RATE,
    WEIGHT_PERTURB_RATE,
    WEIGHT_PERTURB_SIGMA,
)
from .genome import ConnectionGene, Genome, NodeGene, NodeType


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


def mutate_add_node(genome: Genome, rng: np.random.Generator) -> Genome:
    """Stub. Splits a random enabled connection with a new hidden node.

    Implemented in v1. Kept here so the API surface is stable.
    """
    raise NotImplementedError("mutate_add_node lands in v1.")


def mutate_add_connection(genome: Genome, rng: np.random.Generator) -> Genome:
    """Stub. Adds a new connection between two previously unconnected nodes.

    Implemented in v1. Kept here so the API surface is stable.
    """
    raise NotImplementedError("mutate_add_connection lands in v1.")


def mutate_toggle_connection(
    genome: Genome, rng: np.random.Generator
) -> Genome:
    """Stub. Disables (or re-enables) a random connection.

    Implemented in v1. Kept here so the API surface is stable.
    """
    raise NotImplementedError("mutate_toggle_connection lands in v1.")
